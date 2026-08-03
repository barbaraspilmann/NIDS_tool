# -*- coding: utf-8 -*-
"""
train_model.py - Offline Random Forest training script for the Hybrid NIDS.

Run this once before using the detection tool:
    python train_model.py

The full CICIoT2023 dataset is too large to load all at once on a 16 GB machine,
so the script streams it in chunks of 50,000 rows, takes a random sample from
each chunk, and trains on the combined sample. This keeps peak RAM usage
manageable while still giving the model a representative slice of the dataset.

All random operations use RANDOM_SEED=42 so the output is reproducible.
The same preprocessing function (log_parser.prepare_ml_data) is used here
and in ml_detector.py to avoid any mismatch between training and inference.
"""

import io
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import multiprocessing
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import log_parser  # shared preprocessing — same function used in ml_detector.py


# =====================================================================
# Configuration — edit here or override via environment variables
# (env vars are how the GUI passes user-selected values at runtime)
# =====================================================================
import os as _os

# GUI override: NIDS_DATASET_PATH
DATASET_PATH = Path(_os.environ.get("NIDS_DATASET_PATH", "data/ciciot2023_train01.csv"))

MODEL_OUTPUT_PATH = Path("models/rf_nids_brain.pkl")

# 50k rows ≈ 15–20 MB RAM per chunk — safe on a 16 GB machine.
CHUNK_SIZE = 50_000

# GUI override: NIDS_SAMPLE_FRACTION (1.0 = keep everything from each chunk)
SAMPLE_FRACTION = float(_os.environ.get("NIDS_SAMPLE_FRACTION", "1.0"))

# GUI override: NIDS_MAX_CHUNKS (0 = read the whole file)
# Default of 4 chunks gives 200k training rows — enough for a good model
# without pushing RAM limits.
MAX_CHUNKS = int(_os.environ.get("NIDS_MAX_CHUNKS", "4"))

# Set by the GUI subprocess launcher so we know to skip serve_forever()
# at the end — that blocking loop would prevent the subprocess from ever
# exiting, causing the GUI to hang indefinitely.
_GUI_MODE = bool(_os.environ.get("NIDS_GUI_MODE", ""))

RANDOM_SEED = 42
N_ESTIMATORS = 100


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_sampled_dataset(
    csv_path: Path,
    chunksize: int,
    sample_frac: float,
    max_chunks: int = 0,
) -> pd.DataFrame:
    """Streams the CICIoT2023 CSV in chunks and returns a sampled training set.

    Reads at most one chunk at a time so RAM usage stays bounded — the full
    dataset is too large to load all at once on a 16 GB machine.

    Args:
        csv_path:    Path to the CSV.
        chunksize:   Rows per batch.
        sample_frac: Fraction of each batch to keep (0.0–1.0).
        max_chunks:  Stop after this many chunks (0 = read entire file).

    Returns:
        Concatenated DataFrame of all sampled rows.

    Raises:
        FileNotFoundError: If csv_path doesn't exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            f"Please copy ciciot2023_train01.csv into the 'data/' directory."
        )

    cap_str = f"capped at {max_chunks} chunks ({max_chunks * chunksize:,} rows)" if max_chunks else "full file"
    print(f"[*] Streaming dataset from : {csv_path}")
    print(f"    Chunk size              : {chunksize:,} rows")
    print(f"    Sample fraction         : {sample_frac:.0%} per chunk")
    print(f"    Chunk limit             : {cap_str}\n")

    sampled_chunks = []
    total_rows_read = 0

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        if max_chunks and i >= max_chunks:
            print(f"    [*] MAX_CHUNKS={max_chunks} reached — stopping early.")
            break

        total_rows_read += len(chunk)

        # random_state keeps sampling deterministic across runs.
        sampled = chunk.sample(frac=sample_frac, random_state=RANDOM_SEED)
        sampled_chunks.append(sampled)

        print(
            f"    Chunk {i + 1:>4}: "
            f"read {len(chunk):>7,} rows  ->  kept {len(sampled):>5,} rows  "
            f"(total read: {total_rows_read:,})"
        )

    print(f"\n[*] Finished streaming. Total rows read : {total_rows_read:,}")

    # ignore_index=True resets the row index so there are no duplicates
    # from rows that happened to share the same index across different chunks.
    df_sample = pd.concat(sampled_chunks, ignore_index=True)
    print(f"[*] Training sample size    : {len(df_sample):,} rows")
    return df_sample


# =====================================================================
# MAIN TRAINING PIPELINE
# =====================================================================

def main() -> None:
    """Orchestrates the full offline training pipeline."""

    # Ensure the output directories exist before we try to write to them.
    # parents=True  -> also creates parent directories if they do not exist.
    # exist_ok=True -> does not raise an error if the directory already exists.
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Hybrid NIDS - Offline Random Forest Training Script")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    df_raw = load_sampled_dataset(DATASET_PATH, CHUNK_SIZE, SAMPLE_FRACTION, MAX_CHUNKS)

    # ------------------------------------------------------------------
    # 2. PREPARE FEATURES AND LABELS
    # ------------------------------------------------------------------
    # Using log_parser.prepare_ml_data() here and in ml_detector.py ensures
    # the exact same preprocessing steps run at both training and inference.
    print("\n[*] Preparing features and labels...")
    X, y = log_parser.prepare_ml_data(df_raw)

    print(f"    Feature matrix   : {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"    Label distribution (0=Benign, 1=Attack):")
    print(y.value_counts().rename({0: "  Benign (0)", 1: "  Attack (1)"}).to_string())

    # ------------------------------------------------------------------
    # 3. TRAIN / TEST SPLIT
    # ------------------------------------------------------------------
    # 80/20 split. stratify=y keeps the attack/benign ratio consistent
    # across both sets — without it the test set could end up with very
    # few attack samples and give an artificially high accuracy score.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\n[*] Train / Test split:")
    print(f"    Training set : {len(X_train):,} rows  (80 %)")
    print(f"    Test set     : {len(X_test):,} rows  (20 %)")

    # ------------------------------------------------------------------
    # 4. TRAIN THE RANDOM FOREST
    # ------------------------------------------------------------------
    # class_weight="balanced" is important here — benign traffic heavily
    # outnumbers attacks in the dataset, so without it the model tends to
    # just predict "benign" for everything and still hit decent accuracy
    # while missing most of the actual attacks.
    #
    # n_jobs=1 rather than -1: when running inside the frozen .exe, joblib
    # spawns workers by calling sys.executable — which is NIDS-Tool.exe.
    # Each worker re-runs the exe entry point, falls through to the GUI
    # constructor (no --nids-train flag in the worker's argv), and opens
    # a new window. One per CPU core, simultaneously. This is bad.
    # n_jobs=1 avoids all of that; training is still fast enough for a demo.
    print(f"\n[*] Training RandomForestClassifier "
          f"({N_ESTIMATORS} trees, n_jobs=1 — PyInstaller safe mode)...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        n_jobs=1,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    print("    Training complete.")

    # ------------------------------------------------------------------
    # 5. EVALUATE THE MODEL
    # ------------------------------------------------------------------
    print("\n[*] Evaluating on held-out test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # --- Accuracy Score ---
    print(f"\n{'=' * 45}")
    print(f"  ACCURACY SCORE : {accuracy:.4f}  ({accuracy * 100:.2f} %)")
    print(f"{'=' * 45}")

    # Confusion matrix layout:
    #                  Pred Benign   Pred Attack
    # Actual Benign  [   TN            FP (false alarm)  ]
    # Actual Attack  [   FN (missed)   TP (caught)       ]
    # For a NIDS, FN (missed attacks) is the worst outcome to minimise.
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  CONFUSION MATRIX (rows=Actual, cols=Predicted):")
    print(f"                   Pred:Benign   Pred:Attack")
    print(f"  Actual Benign    {tn:>11,}   {fp:>11,}   <- False Positives")
    print(f"  Actual Attack    {fn:>11,}   {tp:>11,}   <- Missed (FN) / Caught (TP)")

    print(f"\n  FULL CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))

    # Feature importances — mean decrease in Gini impurity across all trees.
    # Useful for the dissertation to show which network features matter most.
    feature_names = list(X.columns)
    importances = model.feature_importances_
    top_n = 10
    top_indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n  TOP {top_n} MOST IMPORTANT FEATURES (Gini importance):")
    for rank, idx in enumerate(top_indices, start=1):
        bar = "#" * int(importances[idx] * 200)  # visual bar
        print(f"    {rank:>2}. {feature_names[idx]:<35} {importances[idx]:.4f}  {bar}")

    # ------------------------------------------------------------------
    # 5b. UPGRADE 2 — 5-FOLD CROSS-VALIDATION
    # ------------------------------------------------------------------
    # The single 80/20 hold-out from step 5 depends on which rows ended up
    # in the test set, which can vary. 5-fold CV trains and evaluates on 5
    # different partitions and reports the mean, which is a more reliable
    # performance estimate. Using 5 folds to meet the assessment criteria.
    #
    # Using a lightweight 10-tree proxy model rather than the full 100-tree
    # RF for CV — this gives 50 total tree fits vs 500, keeping the CV step
    # fast enough for a demo. max_depth=10 is critical here: without it,
    # trees on a 200k-row dataset can grow to depth 50+ and the CV step
    # takes 37+ minutes. Capping at depth 10 brings it under 2 minutes.
    print("\n[*] Upgrade 2: Running 5-Fold Cross-Validation (proxy RF)...")
    proxy_model = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        n_jobs=1,           # same PyInstaller safe-mode constraint as main model
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    cv_scores = cross_val_score(
        proxy_model, X, y,
        cv=5,
        scoring="accuracy",
        n_jobs=1,
    )

    print(f"\n  5-FOLD CROSS-VALIDATION RESULTS  (10-tree proxy RF)")
    print(f"  Fold scores : {' | '.join(f'{s:.4f}' for s in cv_scores)}")
    print(f"  Mean acc.   : {cv_scores.mean():.4f}  ({cv_scores.mean() * 100:.2f}%)")
    print(f"  Std dev     : {cv_scores.std():.4f}  (+/- {cv_scores.std() * 100:.2f}%)")
    print(f"  Verdict     : {'Stable generalisation (std < 1%)' if cv_scores.std() < 0.01 else 'Some variance across folds — consider more training data'}")

    # ------------------------------------------------------------------
    # 5c. UPGRADE 1 — SHAP EXPLAINABILITY ANALYSIS
    # ------------------------------------------------------------------
    # SHAP (Lundberg & Lee, 2017) explains individual predictions rather
    # than just giving a global feature importance. Positive SHAP values
    # push a prediction towards "Attack", negative towards "Benign".
    # The beeswarm summary plot makes this easy to present in the dissertation.
    #
    # TreeExplainer is used because it can compute exact Shapley values
    # for tree models in polynomial time, which is much faster than the
    # generic KernelExplainer.
    #
    # Only running SHAP on 100 samples — computing it on the full test set
    # would allocate a (45000, 39, 2) array that exceeds available RAM on
    # a 16 GB machine. 100 samples is enough for a meaningful summary plot.
    SHAP_SUMMARY_PATH = Path("reports/shap_summary.png")
    shap_generated = False  # flag used when building the HTML report

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — no GUI window
        import matplotlib.pyplot as plt

        print("\n[*] Upgrade 1: Running SHAP explainability analysis...")
        print("    (using 100-sample background + 100 test samples for memory safety)")

        # 100-row background dataset for the explainer — this sets the
        # baseline prediction that SHAP values are measured relative to.
        background_data = shap.sample(X_train, 100, random_state=RANDOM_SEED)
        test_data       = shap.sample(X_test,  100, random_state=RANDOM_SEED)

        explainer = shap.TreeExplainer(model, data=background_data)

        # check_additivity=False skips a post-computation validation pass
        # that verifies SHAP values sum to the model output. This check is
        # correct-by-construction for TreeExplainer so skipping it just
        # saves ~90 seconds of unnecessary computation.
        shap_values = explainer.shap_values(test_data, check_additivity=False)

        if isinstance(shap_values, list):
            # Old API: index 1 selects the "Attack" class SHAP values
            plot_values = shap_values[1]
        else:
            # New API: last axis is the class axis; slice index 1 (Attack)
            plot_values = shap_values[:, :, 1]

        # Beeswarm plot: one dot per sample per feature, sorted by mean
        # absolute SHAP value. max_display=15 keeps it readable.
        shap.summary_plot(
            plot_values,
            test_data,
            show=False,       # suppress interactive GUI window
            max_display=15,
        )
        plt.tight_layout()
        plt.savefig(str(SHAP_SUMMARY_PATH), dpi=150, bbox_inches="tight")
        plt.close()
        shap_generated = True
        print(f"    SHAP summary plot saved -> {SHAP_SUMMARY_PATH}")
        print(f"    Interpretation: features with rightward dots strongly predict 'Attack'.")

    except ImportError:
        print("\n[*] SHAP not installed -- skipping XAI analysis.")
        print("    To enable: pip install shap")
    except Exception as shap_err:
        print(f"\n[*] SHAP analysis failed: {shap_err}")
        print("    Continuing without SHAP output.")

    # ------------------------------------------------------------------
    # 6. SAVE THE MODEL
    # ------------------------------------------------------------------
    # joblib is recommended over pickle for sklearn models — faster for
    # objects that contain large numpy arrays (which RF models are).
    print(f"\n[*] Saving model -> {MODEL_OUTPUT_PATH}")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    size_kb = MODEL_OUTPUT_PATH.stat().st_size / 1024
    print(f"    Saved ({size_kb:,.1f} KB).")

    print("\n" + "=" * 65)
    print("  Training complete.  You can now run the NIDS tool:")
    print("  python main.py --logfile data/ciciot2023_train01.csv --hybrid")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 7. SERVE RESULTS PAGE (standalone mode only)
    # ------------------------------------------------------------------

    # Build feature-importance rows for the HTML table
    fi_rows = "".join(
        f"<tr><td>{rank}</td><td>{feature_names[idx]}</td>"
        f"<td>{importances[idx]:.4f}</td>"
        f"<td><div style='background:#3b82f6;height:14px;width:{int(importances[idx]*500)}px'></div></td></tr>"
        for rank, idx in enumerate(top_indices, start=1)
    )

    # Build CV results rows for the HTML table (Upgrade 2)
    cv_fold_cells = "".join(f"<td>{s:.4f}</td>" for s in cv_scores)
    cv_section_html = f"""
      <h2>Upgrade 2 - 5-Fold Cross-Validation (10-tree proxy RF)</h2>
      <p>CV measures generalisation across 5 independent data partitions.
         A low std dev (&lt;1%) confirms stable performance regardless of which
         rows land in the test set.</p>
      <table>
        <tr><th>Fold 1</th><th>Fold 2</th><th>Fold 3</th><th>Fold 4</th><th>Fold 5</th>
            <th>Mean</th><th>Std Dev</th></tr>
        <tr>{cv_fold_cells}
            <td class="ok"><b>{cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)</b></td>
            <td>+/- {cv_scores.std()*100:.2f}%</td></tr>
      </table>"""

    # Embed SHAP image as base64 if it was generated (Upgrade 1)
    shap_section_html = ""
    if shap_generated and SHAP_SUMMARY_PATH.exists():
        import base64
        img_b64 = base64.b64encode(SHAP_SUMMARY_PATH.read_bytes()).decode("ascii")
        shap_section_html = f"""
      <h2>Upgrade 1 - SHAP Explainability (Top 15 Features)</h2>
      <p>Each dot = one explained flow. Positive SHAP (rightward) = feature pushes
         prediction towards "Attack". Colour: red = high feature value, blue = low.</p>
      <img src="data:image/png;base64,{img_b64}"
           style="max-width:100%;border-radius:6px;margin-top:.5rem;">"""
    else:
        shap_section_html = """
      <h2>Upgrade 1 - SHAP Explainability</h2>
      <p class="warn">SHAP not installed or analysis failed.
         Run: <code>pip install shap</code> and re-train to generate this chart.</p>"""

    html = textwrap.dedent(f"""
    <!DOCTYPE html><html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>NIDS - Training Results</title>
      <style>
        body{{font-family:monospace;background:#0f172a;color:#e2e8f0;padding:2rem;}}
        h1{{color:#38bdf8;}} h2{{color:#7dd3fc;margin-top:2rem;}}
        .badge{{display:inline-block;background:#1e3a5f;border:1px solid #3b82f6;
               padding:.4rem 1.2rem;border-radius:6px;font-size:1.6rem;color:#38bdf8;}}
        table{{border-collapse:collapse;margin-top:.5rem;}}
        th,td{{padding:.35rem .9rem;border:1px solid #334155;text-align:left;}}
        th{{background:#1e293b;}} tr:hover{{background:#1e293b;}}
        .ok{{color:#4ade80;}} .warn{{color:#facc15;}}
        pre{{background:#1e293b;padding:1rem;border-radius:6px;overflow-x:auto;}}
      </style>
    </head>
    <body>
      <h1>Hybrid NIDS - Random Forest Training Results</h1>
      <p>Dataset: <code>{DATASET_PATH}</code> &nbsp;|&nbsp;
         Model: <code>{MODEL_OUTPUT_PATH}</code> ({size_kb:,.1f} KB)</p>

      <h2>Accuracy</h2>
      <span class="badge">{accuracy*100:.2f}%</span>

      <h2>Confusion Matrix</h2>
      <table>
        <tr><th></th><th>Predicted: Benign</th><th>Predicted: Attack</th></tr>
        <tr><td><b>Actual: Benign</b></td>
            <td class="ok">{tn} (TN)</td>
            <td class="warn">{fp} (FP - False Alarm)</td></tr>
        <tr><td><b>Actual: Attack</b></td>
            <td class="warn">{fn} (FN - Missed!)</td>
            <td class="ok">{tp} (TP - Caught)</td></tr>
      </table>

      <h2>Top 10 Feature Importances (Gini)</h2>
      <table>
        <tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Magnitude</th></tr>
        {fi_rows}
      </table>

      <h2>Full Classification Report</h2>
      <pre>{classification_report(y_test, y_pred, target_names=["Benign","Attack"])}</pre>

      {cv_section_html}

      {shap_section_html}
    </body></html>
    """).strip()

    # Thread-safe holder so the handler closure can reference the HTML string
    _html = html

    class ResultsHandler(BaseHTTPRequestHandler):
        """Minimal HTTP handler that serves the training results page."""
        def do_GET(self):
            body = _html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            # Suppress the default per-request console noise
            pass

    if _GUI_MODE:
        # Save to disk and exit — can't call serve_forever() here because
        # that's a blocking loop that would prevent the subprocess from
        # exiting, causing the GUI's process.wait() to hang indefinitely.
        html_path = Path("reports/training_results.html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        print(f"\n[*] Results page saved -> {html_path}")
        print("    Open it in a browser, or browse it in the Reports tab.")
        # Process exits normally here — the GUI subprocess wait() unblocks.
    else:
        # Standalone mode — keep the process alive to serve results in browser.
        PORT = 5001
        server = HTTPServer(("", PORT), ResultsHandler)
        print(f"\n[*] Results available at http://localhost:{PORT}")
        print("    Press Ctrl+C to stop.\n")
        server.serve_forever()


if __name__ == "__main__":
    # Required for PyInstaller on Windows — intercepts loky worker re-entry
    # so workers don't accidentally re-run main() and open a new GUI window.
    multiprocessing.freeze_support()
    main()
