# -*- coding: utf-8 -*-
"""
ml_detector.py - Stage 2 ML detection engine for the Hybrid NIDS pipeline.

Loads the trained Random Forest model and classifies any flows that weren't
already caught by the Stage 1 rule-based detector. Keeping ML detection in
its own module means I can swap the model out (e.g. try a different
classifier) without touching any of the rule-based or reporting code.

Uses predict_proba() rather than predict() so a confidence threshold can be
applied — this makes it easy to tune how aggressive the detector is without
retraining.
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

import log_parser  # shared preprocessing — same function as used in training


# MITRE ATT&CK mapping for every CICIoT2023 attack label.
# Added as an upgrade to make ML detections more informative — instead of
# just "attack detected", the incident report now includes the specific
# ATT&CK tactic and technique so results are easier to interpret.
# Mappings taken from MITRE ATT&CK Enterprise Matrix v14 (attack.mitre.org).
MITRE_MAPPING = {
    # --- DDoS ---
    "DDoS-PSHACK_Flood":    {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (TCP PSH/ACK)"},
    "DDoS-SYN_Flood":       {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (SYN Flood)"},
    "DDoS-UDP_Flood":       {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (UDP Flood)"},
    "DDoS-ICMP_Flood":      {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (ICMP Flood)"},
    "DDoS-SlowLoris":       {"tactic": "TA0040 — Impact",
                              "technique": "T1499.002 — Service Exhaustion Flood (SlowLoris)"},
    "DDoS-HTTP_Flood":      {"tactic": "TA0040 — Impact",
                              "technique": "T1499.002 — Service Exhaustion Flood (HTTP Flood)"},
    "DDoS-RSTFIN_Flood":    {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (RST/FIN Flood)"},
    "DDoS-ACK_Fragmentation": {"tactic": "TA0040 — Impact",
                                "technique": "T1498.001 — Direct Network Flood (ACK Frag)"},
    "DDoS-UDP_Fragmentation": {"tactic": "TA0040 — Impact",
                                "technique": "T1498.001 — Direct Network Flood (UDP Frag)"},
    "DDoS-ICMP_Fragmentation": {"tactic": "TA0040 — Impact",
                                 "technique": "T1498.001 — Direct Network Flood (ICMP Frag)"},
    "DDoS-TCP_Flood":       {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (TCP Flood)"},

    # --- DoS (single source) ---
    "DoS-UDP_Flood":        {"tactic": "TA0040 — Impact",
                              "technique": "T1499.002 — Service Exhaustion Flood (DoS UDP)"},
    "DoS-SYN_Flood":        {"tactic": "TA0040 — Impact",
                              "technique": "T1499.002 — Service Exhaustion Flood (DoS SYN)"},
    "DoS-TCP_Flood":        {"tactic": "TA0040 — Impact",
                              "technique": "T1499.002 — Service Exhaustion Flood (DoS TCP)"},
    "DoS-HTTP_Flood":       {"tactic": "TA0040 — Impact",
                              "technique": "T1499.002 — Service Exhaustion Flood (DoS HTTP)"},

    # --- Mirai botnet ---
    "Mirai-greeth_flood":   {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (Mirai GRE-ETH)"},
    "Mirai-greip_flood":    {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (Mirai GRE-IP)"},
    "Mirai-udpplain":       {"tactic": "TA0040 — Impact",
                              "technique": "T1498.001 — Direct Network Flood (Mirai UDP Plain)"},

    # --- Reconnaissance ---
    "Recon-HostDiscovery":  {"tactic": "TA0043 — Reconnaissance",
                              "technique": "T1595.001 — Active Scanning: Scanning IP Blocks"},
    "Recon-OSScan":         {"tactic": "TA0043 — Reconnaissance",
                              "technique": "T1595.001 — Active Scanning: OS Fingerprinting"},
    "Recon-PortScan":       {"tactic": "TA0043 — Reconnaissance",
                              "technique": "T1595.001 — Active Scanning: Port Scanning"},
    "Recon-PingSweep":      {"tactic": "TA0043 — Reconnaissance",
                              "technique": "T1595.001 — Active Scanning: Ping Sweep"},
    "VulnerabilityScan":    {"tactic": "TA0043 — Reconnaissance",
                              "technique": "T1595.002 — Active Scanning: Vulnerability Scanning"},

    # --- MITM / Spoofing ---
    "DNS_Spoofing":         {"tactic": "TA0006 — Credential Access",
                              "technique": "T1557 — Adversary-in-the-Middle (DNS Spoofing)"},
    "MITM-ArpSpoofing":     {"tactic": "TA0009 — Collection",
                              "technique": "T1557.002 — ARP Cache Poisoning"},

    # --- Web application attacks ---
    "BrowserHijacking":     {"tactic": "TA0001 — Initial Access",
                              "technique": "T1189 — Drive-by Compromise"},
    "CommandInjection":     {"tactic": "TA0002 — Execution",
                              "technique": "T1059 — Command and Scripting Interpreter"},
    "SqlInjection":         {"tactic": "TA0001 — Initial Access",
                              "technique": "T1190 — Exploit Public-Facing Application (SQLi)"},
    "XSS":                  {"tactic": "TA0001 — Initial Access",
                              "technique": "T1189 — Drive-by Compromise (Cross-Site Scripting)"},
    "Backdoor_Malware":     {"tactic": "TA0003 — Persistence",
                              "technique": "T1505 — Server Software Component (Backdoor)"},
    "Uploading_Attack":     {"tactic": "TA0001 — Initial Access",
                              "technique": "T1190 — Exploit Public-Facing Application (Upload)"},

    # --- Brute force ---
    "DictionaryBruteForce": {"tactic": "TA0006 — Credential Access",
                              "technique": "T1110.001 — Brute Force: Password Guessing"},

    # Benign traffic — no MITRE mapping applicable.
    "BenignTraffic":        {"tactic": "N/A — Benign",
                              "technique": "N/A"},
}

# Fallback entry used when the detected label is not in MITRE_MAPPING.
_MITRE_UNKNOWN = {
    "tactic":    "TA0040 — Impact (inferred)",
    "technique": "T1498 — Network Denial of Service (unclassified variant)",
}


class MLDetectionEngine:
    """Stage 2 detector — classifies flows using a pre-trained Random Forest.

    Designed to catch stealthy attacks that the rate-based Stage 1 rule
    misses. The RF looks at all 39 statistical features rather than just
    the packet rate, so it can pick up things like slow scans and protocol
    abuse that wouldn't trigger a simple threshold.

    Usage::

        engine = MLDetectionEngine(Path("models/rf_nids_brain.pkl"))
        incidents = engine.predict_anomalies(df_unflagged)
    """

    def __init__(self, model_path: Path) -> None:
        """Loads the trained RF model from disk.

        Model is loaded once at init and kept in self.model — loading it
        fresh on every prediction call would add several seconds of delay.

        Args:
            model_path: Path to the .pkl file produced by train_model.py.

        Raises:
            FileNotFoundError: If the model file doesn't exist yet (i.e.
                               training hasn't been run).
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"\n[ML] ERROR: Model file not found at: {model_path}\n"
                f"     Solution: Run the training script first:\n"
                f"               python train_model.py\n"
            )

        print(f"[ML] Loading model from : {model_path}")
        self.model = joblib.load(model_path)

        # Store the feature names the model was trained on so we can
        # reorder input columns to match exactly — column order matters
        # for tree splits and getting it wrong causes silent wrong predictions.
        if hasattr(self.model, "feature_names_in_"):
            self.expected_features: Optional[List[str]] = list(
                self.model.feature_names_in_
            )
        else:
            # Older scikit-learn versions don't record feature names.
            self.expected_features = None

        print(
            f"[ML] Model ready.  "
            f"Trees: {self.model.n_estimators}  |  "
            f"Features expected: {self.model.n_features_in_}"
        )

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def predict_anomalies(
        self,
        df_unflagged: pd.DataFrame,
        confidence_threshold: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """Runs the RF model over flows that passed Stage 1 and returns incidents.

        Uses predict_proba() rather than predict() so the confidence threshold
        can be tuned without retraining — set in config.py as
        ML_CONFIDENCE_THRESHOLD.

        Args:
            df_unflagged:         Flows that were not caught by Stage 1.
            confidence_threshold: Min attack probability to raise an alert (0–1).

        Returns:
            List of incident dicts using the same schema as detector.py so
            reporter.py can handle both without any changes.
        """
        if df_unflagged is None or df_unflagged.empty:
            print("[ML] Stage 2: No unflagged traffic to analyse — skipping.")
            return []

        print(
            f"[ML] Stage 2: Analysing {len(df_unflagged):,} flows that "
            f"passed Stage 1 rule-based detection..."
        )

        # Step A — same preprocessing as training (shared via log_parser).
        # y will be None here since live traffic has no Label column.
        X, _ = log_parser.prepare_ml_data(df_unflagged)

        # Step B — reorder columns to match the training feature order.
        # Column order matters for tree splits; if the dataset version has
        # extra/missing columns we fill with 0 rather than crashing.
        if self.expected_features is not None:
            missing_cols = set(self.expected_features) - set(X.columns)
            if missing_cols:
                print(
                    f"[ML] Warning: {len(missing_cols)} feature(s) missing from "
                    f"input — filling with 0: {missing_cols}"
                )
                for col in missing_cols:
                    X[col] = 0
            X = X[self.expected_features]

        # Step C — run inference.
        # predict_proba() gives [P(benign), P(attack)] per row;
        # we take column index 1 (P(attack)) and apply the confidence threshold.
        attack_probs = self.model.predict_proba(X)[:, 1]
        predictions = (attack_probs >= confidence_threshold).astype(int)

        n_flagged = int(predictions.sum())
        print(
            f"[ML] Stage 2 complete: {n_flagged:,} potential attack(s) detected "
            f"from {len(df_unflagged):,} unflagged flows."
        )

        # Step D — build incident list.
        # CICIoT2023 can be 90%+ attack traffic, so a per-row loop would
        # produce tens of thousands of dicts and freeze the GUI. Instead:
        # one aggregated summary + the TOP_N highest-confidence individual
        # samples for detail.

        TOP_N = 20
        flagged_mask = predictions == 1
        n_flagged = int(flagged_mask.sum())

        if n_flagged == 0:
            return []

        flagged_probs = attack_probs[flagged_mask]

        # Aggregated summary —
        incidents: List[Dict[str, Any]] = [{
            "rule": "ML Stage 2 — Stealthy Attack (Aggregated)",
            "timestamp": pd.Timestamp.now().isoformat(),
            "source_ip": "Multiple",
            "destination_ip": "N/A",
            "details": (
                f"Random Forest flagged {n_flagged:,} flows as stealthy attacks "
                f"(confidence >= {confidence_threshold:.0%}).  "
                f"Confidence range: {flagged_probs.min():.2%} – {flagged_probs.max():.2%}  "
                f"(mean: {flagged_probs.mean():.2%}).  "
                f"These flows passed all Stage 1 rate-based thresholds but show "
                f"anomalous statistical feature signatures."
            ),
            "ml_confidence": round(float(flagged_probs.mean()), 4),
            "flagged_flow_count": n_flagged,
        }]

        # Top N individual incidents sorted by confidence (highest first).
        flagged_indices = flagged_mask.nonzero()[0]
        top_indices = flagged_indices[
            flagged_probs.argsort()[::-1][:TOP_N]
        ]

        df_reset = df_unflagged.reset_index(drop=False)

        for row_idx in top_indices:
            prob = float(attack_probs[row_idx])
            row  = df_reset.iloc[row_idx]

            source_ip = self._get_field(row, ["src_ip", "Src IP", "SourceIP"], "N/A")
            dest_ip   = self._get_field(row, ["dst_ip", "Dst IP", "DestinationIP"], "N/A")
            raw_ts    = self._get_field(row, ["Timestamp", "timestamp", "index"], pd.Timestamp.now())
            ts_str    = raw_ts.isoformat() if hasattr(raw_ts, "isoformat") else str(raw_ts)

            # MITRE enrichment — if there's a Label column (CICIoT2023 test data)
            # look it up in the mapping; otherwise fall back to the generic entry.
            raw_label = self._get_field(row, ["Label", "label", "Category"], None)
            attack_label = str(raw_label) if raw_label is not None else "Unknown"
            mitre_entry = MITRE_MAPPING.get(attack_label, _MITRE_UNKNOWN)

            incidents.append({
                "rule": "ML Stage 2 — High-Confidence Anomaly (Top 20)",
                "timestamp": ts_str,
                "source_ip": str(source_ip),
                "destination_ip": str(dest_ip),
                "details": (
                    f"Random Forest flagged this flow with {prob:.2%} confidence.  "
                    f"Passed all Stage 1 thresholds but has an anomalous "
                    f"statistical signature (one of the {TOP_N} highest-confidence detections).  "
                    f"MITRE ATT&CK: {mitre_entry['tactic']} | {mitre_entry['technique']}."
                ),
                "ml_confidence": round(prob, 4),
                # MITRE ATT&CK fields — structured for downstream filtering/reporting
                "attack_label":    attack_label,
                "mitre_tactic":    mitre_entry["tactic"],
                "mitre_technique": mitre_entry["technique"],
            })

        return incidents

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _get_field(row: pd.Series, candidates: List[str], default: Any) -> Any:
        """Returns the first matching field value from a list of candidate names.

        Different CICIoT2023 file versions use different column names for the
        same field (e.g. 'src_ip' vs 'Src IP'), so we try a list of options.
        """
        for name in candidates:
            if name in row.index:
                return row[name]
        return default
