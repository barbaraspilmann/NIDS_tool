# -*- mode: python ; coding: utf-8 -*-
"""
nids_tool.spec  —  PyInstaller 6 build specification for NIDS Tool
===================================================================

HOW TO BUILD
------------
Open a terminal in the project folder, activate the venv, then run:

    .venv\\Scripts\\pyinstaller.exe nids_tool.spec

The finished folder will be at:
    dist\\NIDS-Tool\\NIDS-Tool.exe

IMPORTANT — data folder
-----------------------
The CICIoT2023 CSV training files are intentionally NOT bundled here
because they can be several hundred MB to multiple GB in size.
To enable training on the target machine, copy the data\\ folder
manually alongside the dist\\NIDS-Tool\\ folder after building.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ── 1. CustomTkinter: theme JSON files, window scaling images, etc. ─────
#    Without this, the app crashes on startup with "No theme file found".
ctk_datas = collect_data_files("customtkinter")

# ── 2. scikit-learn: many lazy C-extension sub-packages PyInstaller
#       misses during static analysis. collect_submodules captures all.
sklearn_hidden = collect_submodules("sklearn")

# ── 3. Our own project modules — imported at RUNTIME inside the
#       _dispatch_train() / _dispatch_detect() functions, so they are
#       invisible to PyInstaller's static analyser. Must be listed here.
project_modules = [
    "train_model", "main", "detector", "ml_detector",
    "reporter",    "log_parser", "config",
]

# ═══════════════════════════════════════════════════════════════════════
# Analysis — scan imports, collect data files
# ═══════════════════════════════════════════════════════════════════════

a = Analysis(
    ["gui.py"],       # single entry point (also the subprocess dispatcher)
    pathex=["."],     # add project root to sys.path inside the bundle
    binaries=[],
    datas=[
        # ── Third-party assets ──────────────────────────────────────────
        *ctk_datas,                            # CustomTkinter themes & images

        # ── Project assets ──────────────────────────────────────────────
        ("models",                 "models"),  # trained RF model (.pkl)
        ("reports",                "reports"), # existing reports / SHAP PNG
        ("logs",                   "logs"),    # log placeholder

        # ── Window icon ─────────────────────────────────────────────────
        ("cyber-security_ico.ico", "."),       # lands in dist root (same as .exe)
    ],
    hiddenimports=[
        # ── Our own runtime-imported modules ───────────────────────────
        *project_modules,

        # ── scikit-learn internal sub-packages ─────────────────────────
        *sklearn_hidden,

        # ── shap (XAI) — no top-level import in gui.py, found only at
        #    train time inside train_model.py ───────────────────────────
        "shap",
        "shap.explainers",
        "shap.explainers._tree",
        "shap.plots",
        "shap.plots._beeswarm",
        "shap.maskers",
        "shap._explanation",

        # ── matplotlib backends ─────────────────────────────────────────
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backends.backend_agg",
        "matplotlib.figure",

        # ── pandas internal C libs ──────────────────────────────────────
        "pandas._libs.tslibs.base",
        "pandas._libs.tslibs.np_datetime",
        "pandas._libs.tslibs.nattype",
        "pandas._libs.tslibs.timezones",
        "pandas._libs.tslibs.offsets",
        "pandas._libs.tslibs.strptime",
        "pandas._libs.tslibs.period",

        # ── joblib / loky (sklearn parallel execution) ──────────────────
        "joblib",
        "joblib.externals.loky",
        "joblib.externals.loky.backend.managers",

        # ── scipy sparse (used internally by sklearn) ───────────────────
        "scipy.sparse.csgraph",
        "scipy.special._cython_special",
        "scipy.linalg.cython_blas",
        "scipy.linalg.cython_lapack",

        # ── reportlab (PDF reporter) ────────────────────────────────────
        "reportlab",
        "reportlab.lib.styles",
        "reportlab.platypus",
    ],

    # Runtime hooks run BEFORE gui.py starts — sets CWD to exe directory
    # so that relative paths in config.py (e.g. "models/rf_nids_brain.pkl")
    # always resolve correctly regardless of where the user launched the exe.
    runtime_hooks=["rthook_set_cwd.py"],

    hookspath=[],
    hooksconfig={},

    # Packages we actively do NOT want bloating the bundle.
    # NOTE: do NOT exclude 'unittest' — pyparsing.testing (used by matplotlib)
    #       imports it at module level, causing a crash on startup if absent.
    excludes=[
        "IPython", "jupyter", "notebook",
        "pytest", "doctest", "_pytest",
        "tkinter.test",
    ],

    noarchive=False,
)

# ═══════════════════════════════════════════════════════════════════════
# PYZ — compressed Python bytecode archive
# ═══════════════════════════════════════════════════════════════════════

pyz = PYZ(a.pure, a.zipped_data)

# ═══════════════════════════════════════════════════════════════════════
# EXE — the actual NIDS-Tool.exe binary
# ═══════════════════════════════════════════════════════════════════════

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,           # --onedir: DLLs go in the folder, not inside exe
    name="NIDS-Tool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                       # IMPORTANT: UPX corrupts numpy/scipy DLLs — leave OFF
    console=False,                   # hides the black terminal window behind the GUI
    icon="cyber-security_ico.ico",   # taskbar + title-bar icon
)

# ═══════════════════════════════════════════════════════════════════════
# COLLECT — assemble everything into dist\NIDS-Tool\
# ═══════════════════════════════════════════════════════════════════════

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="NIDS-Tool",

    # PyInstaller 6 introduced a new "_internal" subdirectory by default.
    # Setting contents_directory="." disables that and puts every file
    # directly in dist\NIDS-Tool\ next to the exe.
    #
    # WHY THIS MATTERS FOR THIS APP:
    #   resource_path() → _BUNDLE_DIR / ...   (_MEIPASS = dist\NIDS-Tool\)
    #   writable_path() → exe.parent / ...    (also dist\NIDS-Tool\)
    #   config.ML_MODEL_PATH = Path("models/rf_nids_brain.pkl")  (relative = dist\NIDS-Tool\models\)
    # All three resolve to the same directory — no path confusion.
    contents_directory=".",
)
