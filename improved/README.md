# improved/

This is a copy of the current, working version of the tool — identical to
the files at the root of this repository. It exists so you can diff it
directly against [`../original-submission/`](../original-submission/)
without needing to know that the repo root also happens to be the "live"
version.

Three files changed since the version that was submitted and graded
(10/05/2026, Report: 73/100, Creative Piece: 71/100):

**`ml_detector.py`** — the MITRE ATT&CK label lookup was case-sensitive, so
it never matched the dataset's own uppercase labels and always fell back to
a generic "unclassified" tactic. Fixed with a case-insensitive lookup table
built once at import time. Also removed a fallback that printed a row's raw
numeric position as if it were a timestamp — it now says `N/A` when there's
genuinely no timestamp column, which is the honest answer for CICIoT2023's
flow-feature format.

**`main.py`** — a small number of CICIoT2023 flows have an unbounded `Rate`
value (division by a near-zero inter-arrival time), which was leaking
straight into the incident report as `Rate range: 1000.0 - inf pps`. The
Stage 1 summary now reports the finite range and separately notes how many
flows were unbounded.

**`reporter.py`** — a completely clean detection run (zero incidents) used
to write no report file at all, which meant the GUI's "load latest report"
view would keep showing results from whatever run happened before it. Now a
clean run is recorded like any other, just with `total_incidents: 0`.

None of these change the detection logic, thresholds, or the trained
model's behaviour — they're all display and reporting fixes for output that
was already being computed correctly underneath.

## Also added here

`evaluate_holdout.py` — the dissertation (Section 5.1) describes evaluating
the model against `train02.csv` as a genuinely unseen test set. The
training script never actually did this: it evaluates via an internal
80/20 split of the same `train01.csv` sample used for training. This script
closes that gap by loading the trained model and scoring it against
`train02.csv` directly. See the script's own header comment for the exact
numbers this produced.
