# Hybrid NIDS — Synoptic Project

A desktop Network Intrusion Detection System built for my 3rd-year synoptic project at Manchester Metropolitan University (2026). The tool combines a rule-based detector with a Random Forest machine learning model to classify network traffic from the [CICIoT2023 dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

Report mark: 73/100. Creative Piece mark: 71/100.

---

## Repository layout

The code at the root of this folder is the current, working version — use this if you just want to run the tool. Two extra folders exist for transparency:

- **`original-submission/`** — a frozen snapshot of the code exactly as submitted and graded (10/05/2026). It contains three small display/reporting bugs, documented in its own README.
- **`improved/`** — a duplicate of the root's fixed code, kept side by side with `original-submission/` so the two can be diffed directly.

I'm keeping the graded version visible rather than quietly rewriting history — the bugs were real, they're now fixed, and the fixes are dated after submission.

---

## How it works

The detection pipeline has two stages:

**Stage 1 — Rule-Based**
Three hand-written rules inspect the traffic data:
- **Threshold rule**: flags any source IP sending more than N packets/second (volumetric flood)
- **Density rule**: flags any source IP that dominates more than X% of traffic in a 1-minute window
- **Canary rule**: flags any traffic directed at honeypot IP addresses

**Stage 2 — Machine Learning (Random Forest)**
Flows that pass Stage 1 are submitted to a 100-tree Random Forest classifier trained on 39 statistical network-flow features from the CICIoT2023 dataset. The model is trained offline using `train_model.py` and saved as a `.pkl` file. At inference time the model uses `predict_proba()` with a configurable confidence threshold (default 0.60) rather than a hard `predict()` call, which makes false-positive tuning easier without retraining.

The architecture is deliberately hybrid: the rule-based stage handles obvious high-volume attacks cheaply and quickly, while the ML stage catches stealthy, low-rate anomalies that a simple threshold would miss.

**Known limitation:** CICIoT2023's flow-feature CSVs are pre-engineered statistics (packet size, timing, protocol flags, etc.) — they don't retain the original source/destination IP addresses. Because of this, individual incidents raised by Stage 2 (ML) always show `N/A` for IP fields. This isn't a bug, it's a property of the dataset format; the Stage 1 rule-based engine still reports real IPs when the legacy packet-capture format is used, since that format includes them.

---

## Why these choices

A few of the decisions that shaped the design, and what I weighed against them:

**Random Forest over gradient boosting (XGBoost/LightGBM) or a neural net.** RF trains and infers fine on a CPU-only laptop, which the deployment target (a standalone `.exe` with no bundled Python environment) required. Gradient boosting would likely have squeezed out a bit more accuracy, but tuning it properly needed more iteration time than I had, and its `n_jobs` parallelism triggered the same PyInstaller re-entry problem described in Lessons Learned below — one more thing to fight rather than one less. A neural net was ruled out earlier still: CICIoT2023's features are already flat, pre-engineered statistics rather than raw sequences, so there's no temporal structure for a network like an LSTM to exploit, and it would have meant a much less interpretable model for no clear accuracy gain.

**39 pre-engineered features, used as-is, rather than hand-picking a subset.** I considered dropping visibly correlated pairs (e.g. packet count and byte count) to simplify the model, but Random Forest's split selection already handles correlated features reasonably well without manual pruning, and cutting features I hadn't rigorously tested felt riskier than leaving that decision to the training process. The SHAP output (`reports/shap_summary.png`) was partly a way of checking this after the fact — if a feature I expected to matter showed near-zero importance, that would have been a prompt to reconsider.

**`ML_CONFIDENCE_THRESHOLD = 0.60` rather than the default 0.50.** A lower threshold catches more borderline flows but raises more false positives; a higher one misses more real attacks. For a security tool, a missed attack (false negative) is worse than an extra alert, so I biased slightly toward catching more — 0.60 was chosen by eye after comparing false-positive counts at a few threshold values on the training sample, not from a formal sweep. It's exposed in `config.py` specifically so it can be retuned without touching detection code.

**Two-stage hybrid rather than ML on everything.** Running the Random Forest on all 700k+ rows works fine on my hardware, but the point of Stage 1 is that a packet count over a hard threshold is unambiguous — there's no reason to spend model inference time (and lose the ability to point at "this exact rule fired") on traffic a two-line comparison already answers with certainty. Stage 2 exists for exactly the flows where that certainty isn't there.

---

## Tech stack

| Component | Library / Tool |
|---|---|
| GUI | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) 5.2 |
| ML model | [scikit-learn](https://scikit-learn.org/) — RandomForestClassifier |
| Data processing | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Explainability | [SHAP](https://shap.readthedocs.io/) — TreeExplainer |
| Data Visualisation | [Matplotlib](https://matplotlib.org/) |
| Packaging | [PyInstaller](https://pyinstaller.org/) 6.x |
| Language | Python 3.11 |

---

## Running from source
```bash
# Clone and set up a virtual environment
git clone https://github.com/barbaraspilmann/NIDS_tool.git
cd NIDS_tool
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
# Download the CICIoT2023 dataset separately (see Dataset section below)
# and place the CSVs in a data/ folder before running the two lines below.
# Train the model first (only needs to be done once)
python train_model.py
# Launch the GUI
python gui.py
# Or run detection from the command line
python main.py --logfile data/ciciot2023_train01.csv --hybrid
# Evaluate the trained model against a genuinely unseen file (train02.csv)
python evaluate_holdout.py
```
A trained model is already included at `models/rf_nids_brain.pkl`, so `train_model.py` doesn't need to be re-run just to try detection — `gui.py` or `main.py --hybrid` will pick it up as-is.

**Building a standalone `.exe`** (optional — this is how the submitted version was packaged):
```bash
pyinstaller nids_tool.spec
```
---

2. **Launch the app** — double-click `NIDS-Tool.exe`. No Python installation required.

3. **Train the model** — open the **Model Training** tab, select the dataset CSV, adjust the chunk slider if needed (2 chunks ≈ 30–60 seconds on most machines), and click **Train Model**. The model is saved to `models/rf_nids_brain.pkl` automatically.

4. **Run detection** — open the **Detection Dashboard** tab, select **Hybrid (Rules + ML)** mode, point it at a CSV file, and click **Run Detection**. Results appear in the metric cards and pie chart, and a JSON report is saved to the `reports/` folder.

---

## Running from source

```bash
# Clone and set up a virtual environment
git clone <repo-url>
cd NIDS_tool
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt

# Train the model first (only needs to be done once)
python train_model.py

# Launch the GUI
python gui.py

# Or run detection from the command line
python main.py --logfile data/ciciot2023_train01.csv --hybrid

# Evaluate the trained model against a genuinely unseen file (train02.csv)
python evaluate_holdout.py
```

---

## Project structure

```
NIDS_tool/
├── gui.py                  # CustomTkinter GUI — main entry point
├── main.py                 # CLI pipeline orchestrator
├── train_model.py          # Offline RF training script
├── detector.py             # Rule-based detection engine (Stage 1)
├── ml_detector.py          # ML detection engine (Stage 2)
├── log_parser.py           # CSV ingestion and shared preprocessing
├── reporter.py             # JSON + PNG report generation
├── config.py               # All tunable thresholds and paths
├── evaluate_holdout.py     # Scores the model against a genuinely unseen file
├── nids_tool.spec          # PyInstaller build spec
├── rthook_set_cwd.py       # PyInstaller runtime hook (path setup)
├── original-submission/    # Frozen snapshot of the graded submission
├── improved/               # Copy of the fixed code, for side-by-side diffing
├── models/                 # Trained model (.pkl) stored here after training
├── reports/                # JSON incident reports and SHAP summary plot
├── data/                   # CICIoT2023 CSV files (not included — too large)
└── logs/                   # Sample legacy log files for testing
```

---

## Dataset

This project uses the [CICIoT2023 dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html) from the Canadian Institute for Cybersecurity. It contains 34 attack types across 7 categories (DDoS, DoS, Mirai, Reconnaissance, Web attacks, MITM, Brute force) plus benign traffic, with 39 pre-engineered statistical flow features per record.

**Testing the Software:**
For convenience during the assessment, a 140 MB sample of the dataset (`ciciot2023_train01.csv`) has been included in the `data/` directory. This allows the examiner to immediately test the software's training and detection pipelines without needing to download the full multi-gigabyte dataset from the official provider. 
To run a test, simply launch `NIDS-Tool.exe` and select this file from the `data/` folder.

It's released under CC BY 4.0, which is why it's cited here rather than bundled in this repo — the files themselves run into the tens of GB, and redistributing them isn't necessary when they're freely available from the source above.

**To try the software yourself:** download `train01.csv` (and `train02.csv` if you want to run `evaluate_holdout.py`) from the link above and place them in a `data/` folder at the project root. `logs/sample_log.csv` is included directly in this repo for testing the legacy rule-based mode without needing the full dataset.

---

## Limitations

- **Offline only.** This tool reads pre-captured CSV files; it doesn't do live packet capture or sit inline on a network. That was a deliberate scope decision from the start, not an oversight — adding real-time capture would be a substantial rewrite (a different ingestion layer, latency constraints the current batch-oriented pipeline doesn't have to think about), and it's listed under Future Work below rather than attempted here.
- **Tied to the CICIoT2023 feature set.** The Random Forest expects exactly the 39 columns that dataset provides. Pointing it at a different NIDS dataset (e.g. UNSW-NB15, CIC-IDS2017) would need a retrain at minimum, and possibly feature-mapping work if the column sets don't line up — this hasn't been tested.
- **Generalisation was checked, but only within CICIoT2023.** `evaluate_holdout.py` (added after submission) scores the actual submitted model against `train02.csv`, a file it never trained on, and gets a consistent result (99.12% accuracy, 0.44% FNR vs. 99.07%/0.49% from the original evaluation). That's a genuine unseen-data check, but it's still the same dataset and the same IoT testbed — it doesn't tell you how the model would behave on a different network's traffic patterns.

## Future work

- Live capture via a library like Scapy, feeding the same Stage 1/Stage 2 pipeline instead of a CSV.
- Retraining or fine-tuning against a second dataset to check the model isn't overfit to CICIoT2023's specific testbed.
- A small pytest suite for the pure functions in `detector.py` and `log_parser.prepare_ml_data()` — they don't need the GUI or a trained model to test, so there's no real reason they're untested.

---

## Notes

- Training on 2 chunks (~100,000 rows) takes roughly 30–60 seconds on a modern laptop with n_jobs=1.
- 5-fold cross-validation uses a lightweight 10-tree proxy model with `max_depth=10` to keep runtime reasonable — full CV on 100 trees took over 37 minutes.
- SHAP explainability analysis runs on a 100-sample subset to avoid out-of-memory errors on 16 GB machines.
- The tool runs entirely offline — no internet connection required after setup.
- Built and tested on Windows 11 x64.

---

## Lessons learned

A few things I ran into while building this that weren't obvious going in:

**Memory management sneaks up on you.** Running the GUI and a training subprocess at the same time (both importing numpy) came close to exhausting memory on my machine — OpenBLAS pre-allocates a thread buffer per logical CPU core, per process, so two processes doubled that overhead. Capping `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` to 1 per process fixed it, at the cost of slower training. For a demo tool that's a fine trade-off; I wouldn't make the same call for something that needed to train fast.

**PyInstaller and multiprocessing don't mix cleanly by default.** scikit-learn's `n_jobs=-1` spawns worker processes by re-invoking the executable — which, once bundled into a single `.exe`, means every worker re-runs the whole program from the top. Without a guard, each worker fell through to the GUI constructor and opened its own window, one per CPU core. Setting `n_jobs=1` sidesteps it entirely; slower per-fit, but training on a 100k-row sample is still well under a minute, so it didn't cost much in practice.

**Cross-validation is expensive in a way that's easy to underestimate.** My first attempt at 5-fold CV used the full 100-tree Random Forest and took well over half an hour. Swapping in a lighter 10-tree proxy model with a capped tree depth for the CV step specifically (not for the final model) brought that down to a couple of minutes while still giving a meaningful stability estimate. I hadn't planned for CV cost to be a separate design decision from the main model's hyperparameters, but it clearly needs to be.

**The dataset doesn't hand you a clean signal for free.** Because CICIoT2023's flow-feature format doesn't include timestamps or IP addresses, a fair chunk of the "detection dashboard" work was about being upfront when that data isn't there — showing `N/A` instead of quietly inventing something plausible-looking. It's tempting to fill gaps with placeholder values that make the output look more complete, but that's exactly the kind of thing that falls apart under scrutiny.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Manchester Metropolitan University — BSc Software Engineering — Synoptic Project 2026*
