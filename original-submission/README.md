# original-submission/

This folder is a frozen snapshot of the code exactly as it was submitted to
Manchester Metropolitan University on 10/05/2026 for the 6G6Z0019 Synoptic
Project (Report mark: 73/100, ; Creative Piece mark: 71/100,
assessor). It's kept here, unmodified, for transparency — so
the code matches word-for-word what the dissertation describes and what was
graded.

It contains three known issues, found during a post-submission review:

1. **MITRE ATT&CK mapping never matched.** `ml_detector.py`'s label lookup
   was case-sensitive (`DDoS-PSHACK_Flood`), but the CICIoT2023 dataset's
   own labels are all uppercase (`DDOS-PSHACK_FLOOD`). Every lookup silently
   missed and fell back to a generic "unclassified" tactic. This is visible
   in Figure 9 of the report — the MITRE tactic bar chart is almost entirely
   the fallback category.
2. **Fake timestamps in ML incident samples.** A fallback field named
   `"index"` was used when no real timestamp column existed, which just
   printed the row's raw position (e.g. `712220`) instead of a real
   timestamp or an honest `N/A`.
3. **`inf` values leaking into report text.** A small number of CICIoT2023
   flows have an unbounded `Rate` value (near-zero inter-arrival time),
   which showed up as `Rate range: 1000.0 - inf pps` in the Stage 1 incident
   summary.

None of these affect the detection logic itself — the rules, thresholds,
and trained model behave identically either way. They're all display/
reporting issues: the wrong text was shown for correctly-flagged incidents.

The fixed versions of these three files (`ml_detector.py`, `main.py`,
`reporter.py`) are in [`../improved/`](../improved/), along with an
explanation of what changed and why. Everything else in this folder —
`config.py`, `detector.py`, `gui.py`, `log_parser.py`, `train_model.py`,
`rthook_set_cwd.py`, `requirements.txt` — is identical to the improved
version; only the three files above changed.
