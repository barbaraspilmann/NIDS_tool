# rthook_set_cwd.py
# PyInstaller runtime hook — executed BEFORE gui.py starts.
#
# PURPOSE 1 — Working directory
#   Guarantee CWD = the folder containing NIDS-Tool.exe so that relative
#   paths in config.py (e.g. Path("models/rf_nids_brain.pkl")) always
#   resolve correctly, regardless of how the user launched the exe.
#
# PURPOSE 2 — Asset bridge
#   PyInstaller 6 places all bundled support files in an "_internal"
#   subdirectory (sys._MEIPASS = dist\NIDS-Tool\_internal\).
#   Our app resolves writable paths relative to the EXE directory
#   (dist\NIDS-Tool\), which is one level up.
#   On first launch we copy the bundled asset folders (models, reports,
#   logs) from _internal\ to the exe directory so the application can
#   read and write them via the standard relative paths in config.py.
#
# This file is referenced in nids_tool.spec:
#   runtime_hooks=["rthook_set_cwd.py"]
# It is NOT a regular Python module — PyInstaller executes it automatically.

import os
import sys
import shutil

if getattr(sys, "frozen", False):
    _exe_dir  = os.path.dirname(sys.executable)   # dist\NIDS-Tool\
    _internal = sys._MEIPASS                        # dist\NIDS-Tool\_internal\

    # ── Step 1: set CWD to exe directory ────────────────────────────────
    os.chdir(_exe_dir)

    # ── Step 2: bridge bundled asset folders to the writable exe level ──
    # Only copies on the FIRST launch (dst doesn't exist yet).
    # After that, the user's own reports/models accumulate there untouched.
    for _folder in ("models", "reports", "logs"):
        _src = os.path.join(_internal, _folder)
        _dst = os.path.join(_exe_dir,  _folder)
        if os.path.isdir(_src) and not os.path.isdir(_dst):
            try:
                shutil.copytree(_src, _dst)
            except Exception:
                # Non-fatal — app will show its own "model not found" error
                os.makedirs(_dst, exist_ok=True)
