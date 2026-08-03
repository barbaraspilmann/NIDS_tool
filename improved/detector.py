# -*- coding: utf-8 -*-
"""
detector.py - Rule-based detection engine for the legacy NIDS pipeline.

Implements three rules (threshold, density, canary) as methods on a
DetectionEngine class. All thresholds come from config.py rather than
being hard-coded here, which makes testing different sensitivity levels easier.

Each rule produces one aggregated incident (the overall summary) plus up to
_TOP_N individual sample entries. I went with aggregated incidents because
generating one dict per flagged packet produced tens of thousands of entries
that completely froze the GUI's text box — not useful for anyone anyway.

Performance note: all rules use pandas boolean masking instead of iterrows().
On large CSVs the difference is significant — iterrows() on 10k rows takes
several seconds; boolean masking on the same data is nearly instant.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Set

# Maximum individual sample incidents to surface per rule.
# The aggregated summary always reports the full count and range statistics.
_TOP_N = 10


class DetectionEngine:
    """A vectorised, rule-based engine to detect anomalies in network traffic."""

    def __init__(self, pps_threshold: int, density_threshold: float, canary_ips: Set[str]):
        """Initialises the detection engine with specific rule parameters.

        Args:
            pps_threshold:     Packets-per-second threshold for Rule 1.
            density_threshold: Traffic-share threshold (0.0–1.0) for Rule 2.
            canary_ips:        Set of honeypot destination IPs for Rule 3.
        """
        self.pps_threshold     = pps_threshold
        self.density_threshold = density_threshold
        self.canary_ips        = canary_ips
        self.incidents: List[Dict[str, Any]] = []

    def run_detection(
        self,
        df_sec: pd.DataFrame,
        df_min: pd.DataFrame,
        df_raw: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Orchestrates all detection rules and returns the combined incident list.

        Args:
            df_sec: DataFrame grouped by 1-second windows.
            df_min: DataFrame grouped by 1-minute windows.
            df_raw: Original, ungrouped DataFrame (for canary checking).

        Returns:
            List of incident dicts: one aggregated entry per rule that fired,
            plus up to ``_TOP_N`` individual sample entries per rule.
        """
        print("Starting detection process...")
        self._check_threshold_rule(df_sec)
        self._check_density_rule(df_min)
        self._check_canary_rule(df_raw)
        print(f"Detection complete. Found {len(self.incidents)} incident(s).")
        return self.incidents

    # ------------------------------------------------------------------
    # Rule 1 — Threshold
    # ------------------------------------------------------------------

    def _check_threshold_rule(self, df_sec: pd.DataFrame):
        """Flags 1-second windows where a single IP's packet count exceeds pps_threshold."""
        flagged = df_sec[df_sec["PacketCount"] > self.pps_threshold]

        if flagged.empty:
            return

        pkt_col = flagged["PacketCount"]

        # One aggregated incident summarises the whole event rather than
        # creating thousands of individual entries that would flood the report.
        self.incidents.append({
            "rule":      "Threshold Violation — Rule 1 (Aggregated)",
            "timestamp": datetime.now().isoformat(),
            "source_ip": "Multiple",
            "details": (
                f"{len(flagged):,} second-window(s) exceeded the threshold of "
                f"{self.pps_threshold:,} pps.  "
                f"Packet range: {int(pkt_col.min()):,} - {int(pkt_col.max()):,} pps  "
                f"(mean: {pkt_col.mean():.1f} pps).  "
                f"Consistent with a volumetric flood / DDoS attack."
            ),
            "flagged_flow_count": int(len(flagged)),
        })

        # Also surface the top _TOP_N highest-rate samples for detail.
        top_idx = np.argsort(pkt_col.values)[::-1][:_TOP_N]
        for pos in top_idx:
            row = flagged.iloc[pos]
            ts  = row["Timestamp"]
            self.incidents.append({
                "rule":      "Threshold Violation — Rule 1 (Sample)",
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "source_ip": str(row["SourceIP"]),
                "details": (
                    f"Exceeded threshold of {self.pps_threshold:,} pps "
                    f"with {int(row['PacketCount']):,} packets/sec."
                ),
            })

    # ------------------------------------------------------------------
    # Rule 2 — Density
    # ------------------------------------------------------------------

    def _check_density_rule(self, df_min: pd.DataFrame):
        """Flags any source IP that monopolises more than density_threshold of
        traffic in a single 1-minute window."""
        total_per_minute = df_min.groupby("Timestamp")["PacketCount"].transform("sum")
        df_work = df_min.copy()
        df_work["TrafficShare"] = df_work["PacketCount"] / total_per_minute

        flagged = df_work[df_work["TrafficShare"] > self.density_threshold]

        if flagged.empty:
            return

        share_col = flagged["TrafficShare"]

        # --- Aggregated summary incident ---
        self.incidents.append({
            "rule":      "Density Violation — Rule 2 (Aggregated)",
            "timestamp": datetime.now().isoformat(),
            "source_ip": "Multiple",
            "details": (
                f"{len(flagged):,} source IP(s) exceeded the density threshold of "
                f"{self.density_threshold:.0%} in at least one 1-minute window.  "
                f"Traffic-share range: {share_col.min():.2%} - {share_col.max():.2%}  "
                f"(mean: {share_col.mean():.2%}).  "
                f"Consistent with a single-source flood or traffic monopolisation."
            ),
            "flagged_flow_count": int(len(flagged)),
        })

        # Top _TOP_N samples by traffic share for the detail view.
        top_idx = np.argsort(share_col.values)[::-1][:_TOP_N]
        for pos in top_idx:
            row = flagged.iloc[pos]
            ts  = row["Timestamp"]
            self.incidents.append({
                "rule":      "Density Violation — Rule 2 (Sample)",
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "source_ip": str(row["SourceIP"]),
                "details": (
                    f"Contributed {row['TrafficShare']:.2%} of traffic "
                    f"(threshold: {self.density_threshold:.0%}) in a 1-minute window."
                ),
            })

    # ------------------------------------------------------------------
    # Rule 3 — Canary
    # ------------------------------------------------------------------

    def _check_canary_rule(self, df_raw: pd.DataFrame):
        """Flags any traffic targeting a honeypot (canary) IP.

        These are decoy addresses that should never receive real traffic, so
        any hit is treated as a high-confidence indicator of scanning or
        lateral movement. Individual incidents are capped at _TOP_N to avoid
        flooding the report if a misconfigured canary gets hammered.
        """
        if not self.canary_ips:
            return  # no canary IPs configured — skip entirely

        canary_hits = df_raw[df_raw["DestinationIP"].isin(self.canary_ips)]

        if canary_hits.empty:
            return

        unique_dests = canary_hits["DestinationIP"].unique().tolist()

        # --- Aggregated summary incident ---
        self.incidents.append({
            "rule":      "Canary Hit — Rule 3 (Aggregated)",
            "timestamp": datetime.now().isoformat(),
            "source_ip": "Multiple",
            "details": (
                f"{len(canary_hits):,} packet(s) targeted honeypot canary IP(s): "
                f"{', '.join(str(ip) for ip in unique_dests)}.  "
                f"Any traffic to a canary IP is a high-confidence indicator of "
                f"reconnaissance or lateral movement."
            ),
            "flagged_flow_count": int(len(canary_hits)),
        })

        # Individual incidents — capped at _TOP_N.
        for pos in range(min(len(canary_hits), _TOP_N)):
            row = canary_hits.iloc[pos]
            ts  = row.name  # Timestamp is the DataFrame index in df_raw
            self.incidents.append({
                "rule":           "Canary Hit — Rule 3 (Individual)",
                "timestamp":      ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "source_ip":      str(row["SourceIP"]),
                "destination_ip": str(row["DestinationIP"]),
                "details": (
                    f"Attempted connection to canary IP {row['DestinationIP']}."
                ),
            })
