# -*- coding: utf-8 -*-
"""
evaluate_holdout.py - Evaluates the trained model against train02.csv.

The dissertation (Section 5.1) describes train02.csv as a completely
unseen test set used to validate generalisation. train_model.py never
actually loads train02.csv anywhere -- its reported accuracy figures come
from an internal 80/20 split of train01.csv, the same file used for
training. This script closes that gap by scoring the already-trained
model against train02.csv directly, which is a genuinely unseen file.

Usage:
    python evaluate_holdout.py
    python evaluate_holdout.py --dataset data/ciciot2023_train02.csv --sample-rows 200000

Example output (200,000-row sample of train02.csv, scored against the
actual models/rf_nids_brain.pkl trained for the submission -- none of
these rows were used in training):

    Accuracy : 0.9912  (99.12%)
    FNR      : 0.0044  (0.44%)

    Confusion matrix (rows=actual, cols=predicted):
                     Pred:Benign   Pred:Attack
    Actual Benign          3,882           913
    Actual Attack            853       194,352

This lines up closely with the 99.07% accuracy / 0.49% FNR reported in the
dissertation from the internal train01 split, which is a good sign — the
model generalises to a genuinely unseen file about as well as the original
(methodologically flawed) evaluation suggested it would.
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

import log_parser
import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/ciciot2023_train02.csv",
                     help="Path to the held-out CSV (default: train02.csv).")
    ap.add_argument("--model", default=str(config.ML_MODEL_PATH),
                     help="Path to the trained .pkl model.")
    ap.add_argument("--sample-rows", type=int, default=0,
                     help="Optional row cap for a quicker run (0 = full file).")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path} -- run train_model.py first."
        )

    print(f"[*] Loading held-out dataset: {args.dataset}")
    t0 = time.time()
    df = pd.read_csv(args.dataset)
    if args.sample_rows:
        df = df.sample(n=min(args.sample_rows, len(df)), random_state=42)
    print(f"    Loaded {len(df):,} rows in {time.time() - t0:.2f}s "
          f"(genuinely unseen -- the model has never trained on this file)")

    print("\n[*] Preparing features and labels (same preprocessing as training)...")
    X, y = log_parser.prepare_ml_data(df)
    if y is None:
        raise ValueError(f"{args.dataset} has no Label column -- can't score it.")

    print(f"[*] Loading model: {model_path}")
    model = joblib.load(model_path)

    # Align columns to the order the model was trained on, same as
    # ml_detector.py does at inference time.
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = set(expected) - set(X.columns)
        for col in missing:
            X[col] = 0
        X = X[expected]

    print("\n[*] Scoring on genuinely unseen data...")
    t0 = time.time()
    y_pred = model.predict(X)
    print(f"    Inference took {time.time() - t0:.2f}s for {len(X):,} rows")

    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print(f"\n{'=' * 55}")
    print(f"  HOLD-OUT EVALUATION ON {args.dataset}")
    print(f"{'=' * 55}")
    print(f"  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  FNR      : {fnr:.4f}  ({fnr * 100:.2f}%)")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"                   Pred:Benign   Pred:Attack")
    print(f"  Actual Benign    {tn:>11,}   {fp:>11,}")
    print(f"  Actual Attack    {fn:>11,}   {tp:>11,}")
    print(f"\n{classification_report(y, y_pred, target_names=['Benign', 'Attack'])}")


if __name__ == "__main__":
    main()
