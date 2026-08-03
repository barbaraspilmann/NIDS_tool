# -*- coding: utf-8 -*-
"""
reporter.py - Output generation for the NIDS pipeline.

Takes the finished incident list and writes:
  - A console summary (capped at 20 lines so it doesn't spam the terminal)
  - A JSON file with the full incident list
  - A PNG traffic graph with incident markers (skipped if df_raw is empty,
    which is the case in hybrid mode since we don't have raw packet data)

Keeping this separate from detection means I can change the output format
without touching any of the detection logic.
"""

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend — safe for all platforms including Windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import List, Dict, Any, Union


def generate_reports(incidents: List[Dict[str, Any]], df_raw: pd.DataFrame, report_path: Union[str, Path]):
    """Orchestrates the generation of all reports.

    Args:
        incidents: A list of detected incident dictionaries.
        df_raw: The original, raw DataFrame for creating the graph.
        report_path: The base path for saving report files (JSON, PNG), without extension.
    """
    report_path = Path(report_path)

    if not incidents:
        # Used to return here without writing anything. That meant a clean
        # run (no incidents at all) left no JSON file behind, so the GUI's
        # "load latest report" would just keep showing results from the
        # previous run instead of reporting a clean pass. A clean result is
        # still a result, so it gets saved like any other.
        print("No incidents detected — traffic looks clean.")
    else:
        # Cap console output at 20 incidents to prevent flooding the terminal
        # (or the GUI log box) when thousands of flows are flagged.
        # The complete list is always saved to the JSON report regardless.
        MAX_CONSOLE = 20
        print(f"\n--- Intrusion Detection Alerts (showing {min(len(incidents), MAX_CONSOLE)} of {len(incidents)}) ---")
        for incident in incidents[:MAX_CONSOLE]:
            print_console_alert(incident)
        if len(incidents) > MAX_CONSOLE:
            print(f"  ... and {len(incidents) - MAX_CONSOLE} more incident(s). See the JSON report for the full list.\n")

    json_report_path = report_path.with_suffix(".json")
    save_json_report(incidents, json_report_path)

    graph_path = report_path.with_suffix(".png")
    generate_traffic_graph(incidents, df_raw, graph_path)


def print_console_alert(incident: Dict[str, Any]):
    """Prints a single formatted alert to the console."""
    print(f"[!] ALERT: {incident['rule']}")
    print(f"    - Timestamp:    {incident['timestamp']}")
    print(f"    - Source IP:    {incident['source_ip']}")
    if "destination_ip" in incident:
        print(f"    - Dest IP:      {incident['destination_ip']}")
    print(f"    - Details:      {incident['details']}\n")


def save_json_report(incidents: List[Dict[str, Any]], file_path: Path):
    """Saves the list of incidents to a JSON file as a structured incident report."""
    try:
        report = {
            "report_title": "NIDS Incident Report",
            "total_incidents": len(incidents),
            "incidents": incidents
        }
        # Use pathlib.Path.write_text for cross-platform file writing
        file_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
        print(f"Successfully saved JSON incident report to: {file_path}")
    except Exception as e:
        print(f"Error saving JSON report: {e}")


def generate_traffic_graph(incidents: List[Dict[str, Any]], df_raw: pd.DataFrame, file_path: Path):
    """Generates and saves a Matplotlib graph of traffic over time with incident markers."""
    if df_raw is None or df_raw.empty:
        print("Cannot generate graph: Raw data is empty.")
        return

    print("Generating traffic analysis graph...")

    # Resample raw data to get total packets per second for the overall traffic plot
    total_traffic = df_raw["PacketCount"].resample("1s").sum()

    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot the main traffic data as a filled area chart for clarity
    ax.fill_between(total_traffic.index, total_traffic.values, alpha=0.3, color="#007ACC")
    ax.plot(total_traffic.index, total_traffic.values, label="Total Packets/Second", color="#007ACC", linewidth=2)

    # Add vertical markers for each incident, de-duplicating the legend label
    legend_added = False
    for incident in incidents:
        ts = pd.to_datetime(incident["timestamp"])
        label = "Incident Detected" if not legend_added else None
        ax.axvline(x=ts, color="#D22B2B", linestyle="--", linewidth=1.5, label=label)
        legend_added = True

    # Formatting the plot
    ax.set_title("Network Traffic Analysis - Detected Incidents Highlighted", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=12)
    ax.set_ylabel("Total Packets per Second", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=11)

    # Improve date formatting on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    try:
        plt.savefig(str(file_path), dpi=150)  # str() ensures compatibility with older matplotlib on Windows
        print(f"Successfully saved traffic graph to: {file_path}")
    except Exception as e:
        print(f"Error saving graph: {e}")
    finally:
        plt.close(fig)  # Free up memory
