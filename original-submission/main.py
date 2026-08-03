# -*- coding: utf-8 -*-
# Limit OpenBLAS / MKL thread pool allocation BEFORE any numpy/pandas import.
# Without this, running two Python processes with numpy simultaneously can
# exhaust the OS memory allocator (OpenBLAS tries to pre-allocate per-thread
# buffers for every logical CPU core). Setting 1 thread per process is safe
# for a CLI tool and eliminates the "Memory allocation failed" crash.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

"""
main.py - Pipeline orchestrator for the Hybrid NIDS tool.

Two modes:
  Legacy (default): reads a raw packet-capture CSV and runs the rule-based
                    DetectionEngine (threshold, density, canary rules).

  Hybrid (--hybrid): two-stage pipeline for CICIoT2023-format data.
    Stage 1 — rate threshold rule catches obvious volumetric floods quickly.
    Stage 2 — Random Forest classifies anything Stage 1 didn't catch.
    Combined incidents go to reporter.py.

Usage:
  python main.py --logfile logs/sample_log.csv
  python main.py --logfile data/ciciot2023_train01.csv --hybrid
"""

import argparse
import json
import textwrap
import time                          # TIMING: used to profile each pipeline stage
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# --- Our own modules ---
import log_parser
import config
from detector import DetectionEngine
from reporter import generate_reports

# The ML engine is imported lazily inside run_hybrid_pipeline() so that the
# tool still starts correctly even if joblib / scikit-learn are not installed,
# as long as the user only runs legacy mode.
# (If you always have scikit-learn installed, you can move this to the top.)


# =====================================================================
# LEGACY PIPELINE  (original behaviour, unchanged)
# =====================================================================

def run_legacy_pipeline(
    log_file: Path,
    report_base_path: Path,
) -> None:
    """Runs the original rule-based pipeline on a raw packet-capture CSV.

    Args:
        log_file:         Path to the input CSV.
        report_base_path: Base path for output files (no extension).
    """
    # --- Stage 1: Ingest and prepare raw log data ---
    df_raw = log_parser.load_and_prepare_data(log_file)
    if df_raw is None or df_raw.empty:
        print("[Legacy] No data to analyse — exiting.")
        return

    # Aggregate into 1-second and 1-minute windows for the rule engine.
    df_sec_grouped = log_parser.group_by_time_window(df_raw, "1s")
    df_min_grouped = log_parser.group_by_time_window(df_raw, "1min")

    # --- Stage 2: Rule-based detection ---
    engine = DetectionEngine(
        pps_threshold=config.PPS_THRESHOLD,
        density_threshold=config.DENSITY_THRESHOLD,
        canary_ips=config.CANARY_IPS,
    )
    incidents = engine.run_detection(df_sec_grouped, df_min_grouped, df_raw)

    # --- Stage 3: Report ---
    generate_reports(incidents, df_raw, report_base_path)


# =====================================================================
# HYBRID PIPELINE  (new two-stage ML pipeline)
# =====================================================================

def run_hybrid_pipeline(
    log_file: Path,
    report_base_path: Path,
) -> None:
    """Runs the two-stage hybrid pipeline on a CICIoT2023-format CSV.

    Stage 1 flags high-rate flows via a simple threshold rule.
    Stage 2 passes the remaining flows to the Random Forest.
    Both incident lists are combined and passed to reporter.py.

    Args:
        log_file:         Path to the CICIoT2023 CSV.
        report_base_path: Base path for output files (no extension).
    """
    # Import here so legacy mode does not require scikit-learn to be installed.
    from ml_detector import MLDetectionEngine

    all_incidents: List[Dict[str, Any]] = []

    t_pipeline_start = time.time()

    # ------------------------------------------------------------------
    # STAGE: Data loading
    # ------------------------------------------------------------------
    print(f"[Hybrid] Loading dataset: {log_file}")
    t0 = time.time()
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"[Hybrid] ERROR: File not found: {log_file}")
        return
    except Exception as exc:
        print(f"[Hybrid] ERROR loading file: {exc}")
        return
    print(f"[TIMER] Data loading          : {time.time() - t0:.2f}s")

    if df.empty:
        print("[Hybrid] Dataset is empty — nothing to analyse.")
        return

    print(f"[Hybrid] Loaded {len(df):,} flows with {len(df.columns)} columns.")

    # Validate that the 'Rate' column exists (required for Stage 1).
    if "Rate" not in df.columns:
        print(
            "[Hybrid] WARNING: 'Rate' column not found.  "
            "Stage 1 rule-based check will be skipped."
        )
        df_flagged_s1 = pd.DataFrame()  # empty — nothing flagged at Stage 1
    else:
        # Stage 1 — flag obvious high-rate flows before bothering the ML model.
        # High-rate flows are characteristic of volumetric floods (SYN, UDP, etc.)
        # and a simple threshold catches them faster and more transparently than RF.
        print(
            f"\n[Stage 1 — Rule-Based] Flagging flows with Rate > "
            f"{config.ML_RATE_THRESHOLD:,.1f} pps..."
        )
        t0 = time.time()
        df_flagged_s1 = df[df["Rate"] > config.ML_RATE_THRESHOLD]
        print(f"[Stage 1] Flagged {len(df_flagged_s1):,} high-rate flows.")
        print(f"[TIMER] Stage 1 rule-based    : {time.time() - t0:.2f}s")

        # One aggregated incident for the whole flood — adding one dict per
        # flagged packet would generate tens of thousands of entries and
        # freeze the GUI's log box.
        if not df_flagged_s1.empty:
            rate_col = df_flagged_s1["Rate"]
            all_incidents.append({
                "rule": "Rule-Based Stage 1 — Volumetric Attack (Aggregated)",
                "timestamp": datetime.now().isoformat(),
                "source_ip": "Multiple",
                "destination_ip": "N/A",
                "details": (
                    f"{len(df_flagged_s1):,} flows exceeded the rate threshold of "
                    f"{config.ML_RATE_THRESHOLD:,.1f} pps.  "
                    f"Rate range: {rate_col.min():.1f} – {rate_col.max():.1f} pps  "
                    f"(mean: {rate_col.mean():.1f} pps).  "
                    f"Consistent with a volumetric DDoS or flood attack."
                ),
                "flagged_flow_count": int(len(df_flagged_s1)),
            })

    # Stage 2 — pass everything Stage 1 didn't catch to the RF model.
    # Index-based exclusion works regardless of whether a Label column is present.
    df_unflagged = df[~df.index.isin(df_flagged_s1.index)]

    print(
        f"\n[Stage 2 — ML] {len(df_unflagged):,} flows passed Stage 1.  "
        f"Submitting to Random Forest for deep inspection..."
    )

    # Instantiate the ML engine (loads the .pkl model from disk).
    t0 = time.time()
    ml_engine = MLDetectionEngine(config.ML_MODEL_PATH)
    print(f"[TIMER] Model load (.pkl)      : {time.time() - t0:.2f}s")

    # Run inference and collect ML incidents.
    t0 = time.time()
    ml_incidents = ml_engine.predict_anomalies(
        df_unflagged,
        confidence_threshold=config.ML_CONFIDENCE_THRESHOLD,
    )
    print(f"[TIMER] Stage 2 ML inference   : {time.time() - t0:.2f}s")
    all_incidents.extend(ml_incidents)

    # ==================================================================
    # SUMMARY
    # ==================================================================
    s1_count = len(df_flagged_s1)
    s2_count = len(ml_incidents)
    print(f"\n[Hybrid] Pipeline complete:")
    print(f"         Stage 1 (Rule-Based) incidents : {s1_count:,}")
    print(f"         Stage 2 (ML)         incidents : {s2_count:,}")
    print(f"         Total                incidents : {len(all_incidents):,}")

    # Reporting — pass an empty df_raw since we don't have raw packet-level
    # data in hybrid mode. reporter.py skips the traffic graph in this case.
    t0 = time.time()
    generate_reports(all_incidents, pd.DataFrame(), report_base_path)
    print(f"[TIMER] Reporting              : {time.time() - t0:.2f}s")
    print(f"[TIMER] Total pipeline         : {time.time() - t_pipeline_start:.2f}s")


# =====================================================================
# ENTRY POINT
# =====================================================================

def main() -> None:
    """Parses CLI arguments and dispatches to the correct pipeline."""

    cli_parser = argparse.ArgumentParser(
        description="Offline Hybrid Network Intrusion Detection Tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    cli_parser.add_argument(
        "-l", "--logfile",
        type=str,
        required=True,
        help=(
            "Path to the input CSV file.\n"
            "  Legacy mode : raw packet-capture CSV\n"
            "                (columns: Timestamp, SourceIP, DestinationIP, PacketCount)\n"
            "  Hybrid mode : CICIoT2023 pre-engineered flow CSV\n"
            "                (39 feature columns + Label)"
        ),
    )
    cli_parser.add_argument(
        "-o", "--output",
        type=str,
        default=r"reports/incident_report",
        help=(
            "Base path for output reports (no file extension).\n"
            "Default: reports/incident_report"
        ),
    )
    cli_parser.add_argument(
        "--hybrid",
        action="store_true",
        default=False,
        help=(
            "Activate the two-stage Hybrid pipeline.\n"
            "  Stage 1: Rule-based Rate threshold check.\n"
            "  Stage 2: Random Forest ML anomaly detection.\n"
            "  Requires: python train_model.py must have been run first."
        ),
    )
    args = cli_parser.parse_args()

    log_file = Path(args.logfile)
    report_base_path = Path(args.output)

    # Print startup banner
    print("=" * 65)
    print("  Offline NIDS — Network Intrusion Detection Tool")
    print("=" * 65)
    mode_label = "HYBRID (Stage 1: Rules + Stage 2: ML)" if args.hybrid else "LEGACY (Rule-Based only)"
    print(f"  Mode      : {mode_label}")
    print(f"  Input     : {log_file}")
    print(f"  Output    : {report_base_path}_<timestamp>.json")
    print("=" * 65 + "\n")

    # Append a timestamp to the report path so each run produces a unique file.
    # pathlib.Path handles Windows backslash / Unix forward slash transparently.
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_report_path = Path(f"{report_base_path}_{timestamp_str}")
    final_report_path.parent.mkdir(parents=True, exist_ok=True)

    # Dispatch to the appropriate pipeline.
    if args.hybrid:
        run_hybrid_pipeline(log_file, final_report_path)
    else:
        run_legacy_pipeline(log_file, final_report_path)

    print("\n" + "=" * 65)
    print("  Analysis complete.")
    print("=" * 65)

    # When running inside the GUI subprocess, skip the HTTP server entirely.
    # serve_forever() is a blocking loop — if we don't skip it, the subprocess
    # never exits and the GUI hangs waiting for it to finish.
    if os.environ.get("NIDS_GUI_MODE") == "1":
        return

    port = 5003 if args.hybrid else 5002

    # Load the JSON report that reporter.py just wrote so we can display it.
    json_path = final_report_path.with_suffix(".json")
    try:
        report_data = json.loads(json_path.read_text(encoding="utf-8"))
        incidents = report_data.get("incidents", [])
    except Exception:
        incidents = []

    # Build one HTML row per incident
    def _incident_row(inc: Dict[str, Any]) -> str:
        rule = inc.get("rule", "")
        ts   = inc.get("timestamp", "")
        src  = inc.get("source_ip", "N/A")
        det  = inc.get("details", "")
        conf = inc.get("ml_confidence", "")
        conf_cell = f"{float(conf)*100:.1f}%" if conf != "" else "—"
        colour = "#fef9c3" if "Stage 2" in rule else "#fce7f3"
        return (
            f"<tr style='background:{colour}'>"
            f"<td>{rule}</td><td>{ts}</td><td>{src}</td>"
            f"<td>{det}</td><td>{conf_cell}</td></tr>"
        )

    rows_html = "".join(_incident_row(i) for i in incidents) or (
        "<tr><td colspan='5' style='text-align:center;color:#4ade80'>"
        "No incidents detected — all traffic looks clean.</td></tr>"
    )

    mode_str = "Hybrid (Stage 1: Rules + Stage 2: ML)" if args.hybrid else "Legacy (Rule-Based)"
    html = textwrap.dedent(f"""
    <!DOCTYPE html><html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>NIDS - Incident Report</title>
      <style>
        body{{font-family:monospace;background:#0f172a;color:#e2e8f0;padding:2rem;}}
        h1{{color:#38bdf8;}} h2{{color:#7dd3fc;margin-top:1.5rem;}}
        .stat{{display:inline-block;background:#1e3a5f;border:1px solid #3b82f6;
               padding:.4rem 1.2rem;border-radius:6px;margin-right:1rem;}}
        table{{border-collapse:collapse;width:100%;margin-top:.5rem;font-size:.85rem;}}
        th,td{{padding:.4rem .7rem;border:1px solid #334155;text-align:left;
               color:#0f172a;}}
        th{{background:#1e293b;color:#e2e8f0;}}
      </style>
    </head>
    <body>
      <h1>NIDS Incident Report</h1>
      <p>Mode: <b>{mode_str}</b> &nbsp;|&nbsp; Input: <code>{log_file}</code></p>
      <span class="stat">Total incidents: <b>{len(incidents)}</b></span>
      <h2>Detected Incidents</h2>
      <table>
        <tr><th>Rule</th><th>Timestamp</th><th>Source IP</th>
            <th>Details</th><th>ML Confidence</th></tr>
        {rows_html}
      </table>
    </body></html>
    """).strip()

    _html = html

    class IncidentHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = _html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass  # suppress per-request console noise

    server = HTTPServer(("", port), IncidentHandler)
    print(f"\n[*] Incident report available at http://localhost:{port}")
    print("    Press Ctrl+C to stop.\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
