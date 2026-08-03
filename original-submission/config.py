# -*- coding: utf-8 -*-
"""
config.py - Central configuration for the NIDS tool.

All the tunable thresholds and file paths live here so they only need
changing in one place. Easier to adjust for different test runs without
digging through the actual detection logic.
"""

from pathlib import Path

# Rule 1 — how many packets/sec from one IP before we flag it as a flood.
# 100 pps is a reasonable default for a lab dataset; real deployments would
# tune this against baseline traffic profiles.
PPS_THRESHOLD = 100

# Rule 2 — if one IP accounts for more than this share of traffic in a
# 1-minute window, that's suspicious. 0.4 = 40%.
DENSITY_THRESHOLD = 0.40

# Rule 3 — honeypot IPs that should never receive real traffic.
# Any hit here is an immediate alert regardless of volume.
CANARY_IPS = {
    "10.0.0.100",
    "192.168.1.200"
}

# =============================================================
# Stage 2 — ML pipeline settings
# =============================================================

# Where the trained Random Forest model is stored.
# Using pathlib so the path works on both Windows and Linux without changes.
ML_MODEL_PATH = Path("models/rf_nids_brain.pkl")

# Flows with a 'Rate' above this are caught by the Stage 1 rule and never
# reach the ML model — keeps inference time down on obvious flood traffic.
ML_RATE_THRESHOLD = 1000.0  # packets per second

# The RF model needs to be at least this confident before raising an alert.
# 0.60 gave a good false-positive / false-negative trade-off during testing.
ML_CONFIDENCE_THRESHOLD = 0.60
