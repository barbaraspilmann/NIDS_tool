# -*- coding: utf-8 -*-
"""
log_parser.py - CSV ingestion and pre-processing for the NIDS pipeline.

Handles reading and cleaning the input log files so that the detection and
ML modules receive a consistent DataFrame to work with. Keeping this separate
means I only have to update one place if the input format ever changes.

Also contains prepare_ml_data(), which is shared between train_model.py and
ml_detector.py — using the exact same preprocessing in both places prevents
the model from seeing a different data format at inference than it trained on.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union


def load_and_prepare_data(log_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Loads a raw packet-capture CSV and returns a cleaned, indexed DataFrame.

    Args:
        log_path: Path to the CSV (str or pathlib.Path).

    Returns:
        DataFrame with a datetime index and standardised columns, or None on error.
    """
    log_path = Path(log_path)

    try:
        column_names = ["Timestamp", "SourceIP", "DestinationIP", "PacketCount"]
        df = pd.read_csv(log_path, header=None, names=column_names)

        if df.empty:
            print(f"Warning: Log file is empty: {log_path}")
            return None

        # Convert Unix epoch timestamps to proper datetime objects so pandas
        # time-based grouping (resample, floor) works correctly downstream.
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
        df.set_index("Timestamp", inplace=True)

        df["PacketCount"] = pd.to_numeric(df["PacketCount"])

        print(f"Successfully loaded and prepared {len(df)} records from {log_path}")
        return df

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred whilst loading the data: {e}")
        return None


def group_by_time_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """Aggregates packet counts by SourceIP within fixed time buckets.

    Used by the legacy detection engine to get per-second and per-minute
    traffic summaries for the threshold and density rules.

    Args:
        df:     DataFrame with a datetime index (from load_and_prepare_data).
        window: Pandas frequency string, e.g. '1s' or '1min'.

    Returns:
        DataFrame with columns [Timestamp, SourceIP, PacketCount].
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Reset the index so Timestamp becomes a regular column we can floor.
    # Using dt.floor() + groupby avoids the deprecated resample()/groupby combo
    # that was removed in newer pandas versions.
    df_reset = df.reset_index()
    df_reset["TimeBucket"] = df_reset["Timestamp"].dt.floor(window)

    grouped_df = (
        df_reset.groupby(["SourceIP", "TimeBucket"])["PacketCount"]
        .sum()
        .reset_index()
        .rename(columns={"TimeBucket": "Timestamp"})
    )

    return grouped_df


# =============================================================
# ML data preparation — shared between training and inference
# =============================================================

def prepare_ml_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Prepares a CICIoT2023 DataFrame for scikit-learn (training or inference).

    Does three things:
      1. Splits the 'Label' column off as y (0 = Benign, 1 = Attack).
         If there's no Label column (live traffic), y is returned as None.
      2. Casts features to float32 — roughly halves RAM usage vs float64
         with no meaningful accuracy loss for a tree-based model.
      3. Fills any NaN / inf values with 0 so sklearn doesn't throw errors.
         Inf values do appear in the CICIoT2023 dataset where a flow has
         zero inter-arrival time, causing division-by-zero in feature calc.

    Args:
        df: CICIoT2023 DataFrame (39 numerical feature columns + optional 'Label').

    Returns:
        Tuple of (X features DataFrame, y label Series or None).
    """
    # Split label off first before any type conversions — avoids triggering
    # pandas block-consolidation which allocates a large contiguous copy.
    if "Label" in df.columns:
        # Using startswith("BENIGN") handles both "BenignTraffic" and "BENIGN"
        # label variants found across different CICIoT2023 file versions.
        # int8 keeps memory usage low — only needs 0 or 1.
        y = (
            df["Label"]
            .apply(lambda label: 0 if str(label).upper().startswith("BENIGN") else 1)
            .astype("int8")
        )
        X = df.drop(columns=["Label"])
    else:
        y = None
        X = df

    # float32 cuts the feature matrix size roughly in half.
    # For 712k rows x 39 columns: float64 ~= 222 MB, float32 ~= 111 MB.
    X = X.astype("float32")

    # inplace=True avoids creating a second copy while cleaning up.
    X.fillna(0, inplace=True)
    X.replace([np.inf, -np.inf], 0, inplace=True)

    return X, y
