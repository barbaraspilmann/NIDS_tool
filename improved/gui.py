# -*- coding: utf-8 -*-
"""
gui.py - CustomTkinter desktop GUI for the Hybrid NIDS (2026 synoptic project).

Tab layout:
  Detection Dashboard — metric cards, pie chart, run/stop controls
  Model Training      — train RF model, shows accuracy + 5-fold CV cards
  Reports             — browse and inspect saved JSON incident reports
  Analytics           — matplotlib charts (flow distribution, MITRE tactics)

PyInstaller notes:
  _FROZEN         True when running as a compiled .exe
  resource_path() read-only asset paths inside the bundle
  writable_path() writable output paths next to the .exe
  The .exe dispatcher checks for --nids-train / --nids-detect flags to
  route subprocess calls without needing a separate script per mode.
"""

import os
import sys
import json
import re
import threading
import subprocess
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Cap thread pool allocation before any numpy/sklearn import to prevent
# OpenBLAS from pre-allocating per-CPU buffers across two processes at once.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
from tkinter import filedialog, messagebox


# ═══════════════════════════════════════════════════════════════════════
# PYINSTALLER PATH HELPERS
# ═══════════════════════════════════════════════════════════════════════

_FROZEN     = getattr(sys, "frozen", False)
_BUNDLE_DIR = Path(sys._MEIPASS) if _FROZEN else Path(__file__).parent


def resource_path(relative: str) -> Path:
    """Read-only asset path — inside the .exe bundle when frozen."""
    return _BUNDLE_DIR / relative


def writable_path(relative: str) -> Path:
    """Writable output path — always next to the .exe / script file."""
    base = Path(sys.executable).parent if _FROZEN else Path(__file__).parent
    return base / relative


# Must be called before any multiprocessing/joblib code in a frozen exe.
# Intercepts loky worker re-entry so workers don't re-open the GUI.
if _FROZEN:
    multiprocessing.freeze_support()


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent

_VENV_PY  = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON    = str(_VENV_PY) if _VENV_PY.exists() else sys.executable

TRAIN_SCRIPT = str(resource_path("train_model.py"))
MAIN_SCRIPT  = str(resource_path("main.py"))
REPORTS_DIR  = writable_path("reports")
MODEL_PATH   = writable_path("models") / "rf_nids_brain.pkl"
ICON_PATH    = PROJECT_ROOT / "cyber-security_ico.ico"

# STARTUPINFO with SW_HIDE stops a console window flashing on subprocess
# launch, and also prevents the subprocess inheriting the bootloader's
# console handles — which can saturate the 64KB pipe buffer and deadlock.
_POPEN_SI: "subprocess.STARTUPINFO | None" = None
if sys.platform == "win32":
    _POPEN_SI = subprocess.STARTUPINFO()
    _POPEN_SI.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _POPEN_SI.wShowWindow = subprocess.SW_HIDE

# ── Colour palette (dark-mode) ───────────────────────────────────────
C_WIN    = "#0f172a"   # window background
C_PANEL  = "#1e293b"   # card / panel fill
C_BORDER = "#334155"   # card border
C_TEXT   = "#e2e8f0"   # primary text
C_MUTED  = "#64748b"   # de-emphasised labels
C_BLUE   = "#38bdf8"   # accent / neutral value
C_GREEN  = "#4ade80"   # success / benign
C_YELLOW = "#facc15"   # warning
C_ORANGE = "#f97316"   # elevated threat
C_RED    = "#ef4444"   # danger / attack
C_PURPLE = "#a855f7"   # ML / Stage 2

# Pie chart colours
PIE_S1    = "#3b82f6"  # Stage 1 volumetric (blue)
PIE_S2    = C_RED      # Stage 2 ML stealthy (red)
PIE_CLEAN = C_GREEN    # benign (green)

# AnalyticsTab chart colours (kept from original)
_DARK_BG   = C_PANEL
_DARK_TEXT = C_TEXT
_CLR_S1    = PIE_S1
_CLR_S2    = C_RED
_CLR_OTHER = C_PURPLE

_MITRE_COLOURS = {
    "TA0040": C_RED,    "TA0043": C_ORANGE, "TA0006": C_YELLOW,
    "TA0009": C_PURPLE, "TA0001": "#ec4899", "TA0002": "#14b8a6",
    "TA0003": "#6366f1",
}

# Global theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ═══════════════════════════════════════════════════════════════════════
# REUSABLE WIDGETS
# ═══════════════════════════════════════════════════════════════════════

class LogBox(ctk.CTkTextbox):
    """Thread-safe scrollable console log."""

    def __init__(self, master, **kw):
        kw.setdefault("font",     ctk.CTkFont(family="Courier New", size=10))
        kw.setdefault("state",    "disabled")
        kw.setdefault("wrap",     "word")
        kw.setdefault("fg_color", C_PANEL)
        super().__init__(master, **kw)

    def append(self, text: str):
        self.after(0, self._write, text)

    def _write(self, text: str):
        self.configure(state="normal")
        self.insert("end", text)
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class MetricCard(ctk.CTkFrame):
    """
    A single statistic display card.

        ┌────────────────────────┐
        │  Title (small muted)   │
        │  VALUE (large bold)    │
        │  subtitle (tiny)       │
        └────────────────────────┘
    """

    def __init__(self, master, title: str, value: str = "—",
                 subtitle: str = "", accent: str = C_BLUE, **kw):
        kw.setdefault("fg_color",      C_PANEL)
        kw.setdefault("corner_radius", 12)
        kw.setdefault("border_width",  1)
        kw.setdefault("border_color",  C_BORDER)
        super().__init__(master, **kw)

        ctk.CTkLabel(self, text=title,
                     font=ctk.CTkFont(size=11),
                     text_color=C_MUTED, anchor="w",
                     ).pack(padx=16, pady=(14, 1), anchor="w")

        self._val_lbl = ctk.CTkLabel(
            self, text=value,
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=accent, anchor="w",
        )
        self._val_lbl.pack(padx=16, pady=(0, 2), anchor="w")

        self._sub_lbl = ctk.CTkLabel(
            self, text=subtitle,
            font=ctk.CTkFont(size=10),
            text_color=C_MUTED, anchor="w",
        )
        self._sub_lbl.pack(padx=16, pady=(0, 14), anchor="w")

    def update(self, value: str, subtitle: str = "", accent: Optional[str] = None):
        """Call from any thread — schedules update on the Tk main loop."""
        self.after(0, self._apply, value, subtitle, accent)

    def _apply(self, value, subtitle, accent):
        self._val_lbl.configure(text=str(value))
        if subtitle:
            self._sub_lbl.configure(text=subtitle)
        if accent:
            self._val_lbl.configure(text_color=accent)


class CVFoldCard(ctk.CTkFrame):
    """Mini card for one K-Fold CV score."""

    def __init__(self, master, fold_num: int, **kw):
        kw.setdefault("fg_color",      C_PANEL)
        kw.setdefault("corner_radius", 8)
        kw.setdefault("border_width",  1)
        kw.setdefault("border_color",  C_BORDER)
        super().__init__(master, **kw)

        ctk.CTkLabel(self, text=f"Fold {fold_num}",
                     font=ctk.CTkFont(size=10), text_color=C_MUTED,
                     ).pack(pady=(8, 2))

        self._lbl = ctk.CTkLabel(self, text="—",
                                  font=ctk.CTkFont(size=14, weight="bold"),
                                  text_color=C_BLUE)
        self._lbl.pack(pady=(0, 8))

    def set_score(self, score: float):
        colour = (C_GREEN  if score >= 0.95 else
                  C_YELLOW if score >= 0.85 else C_RED)
        self.after(0, lambda: self._lbl.configure(
            text=f"{score * 100:.2f}%", text_color=colour
        ))


# ═══════════════════════════════════════════════════════════════════════
# TAB 1  -  DETECTION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class DetectTab(ctk.CTkFrame):
    """
    Replaces the raw text-box output with:
      - A progress bar that advances through pipeline stages in real time
      - Four Metric Cards  (Total Flows / Attacks / Benign / Threat Level)
      - An embedded Matplotlib pie chart of the attack distribution
      - A collapsible compact console log for debugging
    """

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self._process: Optional[subprocess.Popen] = None
        self._analytics_tab = None
        self._total_flows   = 0
        self._build_ui()

    def set_analytics_tab(self, tab):
        self._analytics_tab = tab

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header row ──────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(20, 4))
        ctk.CTkLabel(hdr, text="Detection Dashboard",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=C_BLUE).pack(side="left")
        self._model_badge = ctk.CTkLabel(hdr, text="",
                                          font=ctk.CTkFont(size=11))
        self._model_badge.pack(side="right", padx=4)
        self._refresh_model_badge()

        # ── Control panel ───────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=10)
        ctrl.pack(fill="x", padx=20, pady=(0, 8))

        mode_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        mode_row.pack(fill="x", padx=16, pady=(12, 4))
        ctk.CTkLabel(mode_row, text="Mode:", width=70, anchor="w").pack(side="left")
        self._mode_var = ctk.StringVar(value="hybrid")
        ctk.CTkRadioButton(mode_row, text="Legacy  (Rule-Based only)",
                           variable=self._mode_var, value="legacy",
                           command=self._on_mode_change).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(mode_row, text="Hybrid  (Rules + ML)",
                           variable=self._mode_var, value="hybrid",
                           command=self._on_mode_change).pack(side="left")

        file_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        file_row.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(file_row, text="Input CSV:", width=90, anchor="w").pack(side="left")
        self._logfile_var = ctk.StringVar(
            value=str(PROJECT_ROOT / "data" / "ciciot2023_train01.csv"))
        ctk.CTkEntry(file_row, textvariable=self._logfile_var,
                     width=440).pack(side="left", padx=6)
        ctk.CTkButton(file_row, text="Browse", width=80,
                      command=self._browse_logfile).pack(side="left")

        out_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        out_row.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(out_row, text="Report path:", width=90, anchor="w").pack(side="left")
        self._outpath_var = ctk.StringVar(
            value=str(writable_path("reports") / "incident_report"))
        ctk.CTkEntry(out_row, textvariable=self._outpath_var,
                     width=440).pack(side="left", padx=6)
        ctk.CTkButton(out_row, text="Browse", width=80,
                      command=self._browse_output).pack(side="left")

        btn_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(4, 12))
        self._run_btn = ctk.CTkButton(
            btn_row, text="  Run Detection",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=165, command=self._start)
        self._run_btn.pack(side="left")
        self._stop_btn = ctk.CTkButton(
            btn_row, text="  Stop", width=90, state="disabled",
            fg_color="#dc2626", hover_color="#b91c1c",
            command=self._stop)
        self._stop_btn.pack(side="left", padx=8)
        self._status_lbl = ctk.CTkLabel(btn_row, text="Ready",
                                         text_color=C_MUTED)
        self._status_lbl.pack(side="left", padx=12)

        # ── Progress bar ─────────────────────────────────────────────────
        pb_wrap = ctk.CTkFrame(self, fg_color="transparent")
        pb_wrap.pack(fill="x", padx=20, pady=(0, 10))
        self._pb = ctk.CTkProgressBar(pb_wrap, height=8,
                                       progress_color=C_BLUE)
        self._pb.pack(fill="x")
        self._pb.set(0)

        # ── Metric cards row ─────────────────────────────────────────────
        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(fill="x", padx=20, pady=(0, 10))
        cards.columnconfigure((0, 1, 2, 3), weight=1, uniform="c")

        self._c_total   = MetricCard(cards, "Total Flows",        accent=C_BLUE)
        self._c_attacks = MetricCard(cards, "Attacks Detected",   accent=C_RED)
        self._c_benign  = MetricCard(cards, "Benign Flows",        accent=C_GREEN)
        self._c_threat  = MetricCard(cards, "Threat Level",        accent=C_MUTED)

        self._c_total  .grid(row=0, column=0, padx=(0, 5), sticky="nsew")
        self._c_attacks.grid(row=0, column=1, padx=5,      sticky="nsew")
        self._c_benign .grid(row=0, column=2, padx=5,      sticky="nsew")
        self._c_threat .grid(row=0, column=3, padx=(5, 0), sticky="nsew")

        # ── Embedded pie chart ───────────────────────────────────────────
        chart_panel = ctk.CTkFrame(self, fg_color=C_PANEL,
                                    corner_radius=10, border_width=1,
                                    border_color=C_BORDER)
        chart_panel.pack(fill="both", expand=True, padx=20, pady=(0, 6))

        chart_hdr = ctk.CTkFrame(chart_panel, fg_color="transparent")
        chart_hdr.pack(fill="x", padx=16, pady=(10, 0))
        ctk.CTkLabel(chart_hdr, text="Attack Distribution",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C_TEXT).pack(side="left")

        self._fig    = Figure(figsize=(7, 2.8), facecolor=C_PANEL)
        self._ax     = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_panel)
        self._canvas.get_tk_widget().pack(fill="both", expand=True,
                                          padx=8, pady=(4, 8))
        self._draw_empty_chart()

        # ── Collapsible console log ──────────────────────────────────────
        self._log_open = False
        self._log_toggle = ctk.CTkButton(
            self, text="  Show Console Log", anchor="w",
            fg_color="transparent", hover_color=C_PANEL,
            font=ctk.CTkFont(size=11), text_color=C_MUTED,
            command=self._toggle_log)
        self._log_toggle.pack(fill="x", padx=20, pady=(0, 2))

        self._log_container = ctk.CTkFrame(self, fg_color="transparent")
        self._log = LogBox(self._log_container, height=110)
        self._log.pack(fill="both", expand=True)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _toggle_log(self):
        if self._log_open:
            self._log_container.pack_forget()
            self._log_toggle.configure(text="  Show Console Log")
        else:
            self._log_container.pack(fill="x", padx=20, pady=(0, 8))
            self._log_toggle.configure(text="  Hide Console Log")
        self._log_open = not self._log_open

    def _refresh_model_badge(self):
        if MODEL_PATH.exists():
            kb = MODEL_PATH.stat().st_size / 1024
            self._model_badge.configure(
                text=f"  Model ready  ({kb:,.0f} KB)",
                text_color=C_GREEN)
        else:
            self._model_badge.configure(
                text="  Model not found - run Training first",
                text_color=C_RED)

    def _on_mode_change(self):
        if self._mode_var.get() == "hybrid":
            self._logfile_var.set(
                str(PROJECT_ROOT / "data" / "ciciot2023_train01.csv"))
        else:
            self._logfile_var.set(
                str(PROJECT_ROOT / "logs" / "sample_log.csv"))

    def _browse_logfile(self):
        p = filedialog.askopenfilename(
            title="Select input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self._logfile_var.set(p)

    def _browse_output(self):
        p = filedialog.asksaveasfilename(
            title="Report base path (no extension)",
            initialdir=str(REPORTS_DIR), initialfile="incident_report")
        if p:
            self._outpath_var.set(str(Path(p).with_suffix("")))

    def _draw_empty_chart(self):
        self._ax.clear()
        self._ax.set_facecolor(C_PANEL)
        self._fig.set_facecolor(C_PANEL)
        self._ax.text(0.5, 0.5, "Run detection to populate chart",
                      ha="center", va="center",
                      color=C_MUTED, fontsize=11,
                      transform=self._ax.transAxes)
        self._ax.axis("off")
        self._fig.tight_layout()
        self._canvas.draw()

    # ── Detection run ────────────────────────────────────────────────────

    def _start(self):
        logfile = Path(self._logfile_var.get())
        if not logfile.exists():
            messagebox.showerror("File not found",
                                 f"Input CSV not found:\n{logfile}")
            return
        if self._mode_var.get() == "hybrid" and not MODEL_PATH.exists():
            messagebox.showerror(
                "Model not found",
                f"No trained model found at:\n{MODEL_PATH}\n\n"
                "Please open the Model Training tab and train first.")
            return

        self._log.clear()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._status_lbl.configure(text="Starting...", text_color=C_YELLOW)
        self._pb.configure(mode="indeterminate", progress_color=C_BLUE)
        self._pb.start()
        self._total_flows = 0

        for c in (self._c_total, self._c_attacks, self._c_benign, self._c_threat):
            c.update("—", "")

        out_base = self._outpath_var.get()
        if _FROZEN:
            cmd = [sys.executable, "--nids-detect",
                   "--logfile", str(logfile), "--output", out_base]
        else:
            cmd = [PYTHON, "-u", MAIN_SCRIPT,
                   "--logfile", str(logfile), "--output", out_base]
        if self._mode_var.get() == "hybrid":
            cmd.append("--hybrid")

        env = os.environ.copy()
        env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                    "NIDS_GUI_MODE": "1"})

        threading.Thread(target=self._run_subprocess,
                         args=(cmd, env), daemon=True).start()

    def _stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log.append("\n[GUI] Stopped by user.\n")
        self._finish(cancelled=True)

    def _run_subprocess(self, cmd, env):
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), env=env,
            startupinfo=_POPEN_SI)   # Windows: suppress console window flicker
        for line in self._process.stdout:
            self._log.append(line)
            self._on_line(line)
        self._process.wait()
        rc = self._process.returncode
        self._log.append(f"\n[GUI] Process exited (code {rc}).\n")
        self.after(0, self._finish, rc != 0)

    def _on_line(self, line: str):
        """Parse each stdout line to drive the progress bar."""
        # Determinate progress milestones
        if "[TIMER] Data loading"  in line:
            self.after(0, self._set_pb, 0.20,
                       "Stage 1: Applying rules...", C_YELLOW)
        elif "[TIMER] Stage 1"     in line:
            self.after(0, self._set_pb, 0.40,
                       "Loading ML model...",         C_YELLOW)
        elif "[TIMER] Model load"  in line:
            self.after(0, self._set_pb, 0.55,
                       "Stage 2: ML inference...",   C_YELLOW)
        elif "[TIMER] Stage 2"     in line:
            self.after(0, self._set_pb, 0.85,
                       "Saving report...",            C_YELLOW)
        elif "[TIMER] Reporting"   in line:
            self.after(0, self._set_pb, 0.95, "Finalising...", C_YELLOW)
        # Parse total flows count
        m = re.search(r"Loaded ([\d,]+) flows", line)
        if m:
            self._total_flows = int(m.group(1).replace(",", ""))

    def _set_pb(self, val: float, status: str = "", colour: str = C_BLUE):
        self._pb.stop()
        self._pb.configure(mode="determinate", progress_color=colour)
        self._pb.set(val)
        if status:
            self._status_lbl.configure(text=status, text_color=colour)

    def _finish(self, cancelled: bool = False):
        self._pb.stop()
        self._pb.configure(mode="determinate")
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._refresh_model_badge()

        if cancelled:
            self._pb.set(0)
            self._status_lbl.configure(text="Stopped.", text_color=C_MUTED)
            return

        self._pb.set(1.0)
        self._pb.configure(progress_color=C_GREEN)
        self._status_lbl.configure(text="Complete", text_color=C_GREEN)
        self.after(600, self._load_latest_report)
        if self._analytics_tab:
            self.after(800, self._analytics_tab.refresh)

    def _load_latest_report(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        reports = sorted(REPORTS_DIR.glob("*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if reports:
            self._update_dashboard(reports[0])

    def _update_dashboard(self, json_path: Path):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return

        s1_flows, s2_flows = 0, 0
        for inc in data.get("incidents", []):
            count = int(inc.get("flagged_flow_count", 1))
            rule  = inc.get("rule", "")
            if any(k in rule for k in ("Stage 1", "Rule 1", "Rule 2", "Canary")):
                s1_flows += count
            elif any(k in rule for k in ("Stage 2", "ML")):
                s2_flows += count

        attack  = s1_flows + s2_flows
        total   = self._total_flows if self._total_flows > 0 else attack
        benign  = max(0, total - attack)
        threat, clr = self._threat_level(attack, total)

        self._c_total  .update(f"{total:,}",
                               subtitle="flows analysed", accent=C_BLUE)
        self._c_attacks.update(f"{attack:,}",
                               subtitle="flagged as attacks", accent=clr)
        self._c_benign .update(f"{benign:,}",
                               subtitle="cleared as benign", accent=C_GREEN)
        pct = f"{attack/max(total,1)*100:.1f}% of traffic"
        self._c_threat .update(threat, subtitle=pct, accent=clr)
        self._draw_pie(s1_flows, s2_flows, benign)

    @staticmethod
    def _threat_level(attacks: int, total: int):
        if total == 0 or attacks == 0:
            return "CLEAN",    C_GREEN
        r = attacks / total
        if r < 0.01:   return "LOW",      C_YELLOW
        if r < 0.10:   return "MEDIUM",   C_ORANGE
        if r < 0.50:   return "HIGH",     C_RED
        return             "CRITICAL",    C_RED

    def _draw_pie(self, s1: int, s2: int, benign: int):
        self._ax.clear()
        self._ax.set_facecolor(C_PANEL)
        self._fig.set_facecolor(C_PANEL)

        total = max(s1 + s2 + benign, 1)   # guard against zero-division
        slices, legend_labels, colours = [], [], []
        if s1 > 0:
            slices.append(s1)
            legend_labels.append(f"Stage 1: {s1:,}  ({s1/total*100:.1f}%)")
            colours.append(PIE_S1)
        if s2 > 0:
            slices.append(s2)
            legend_labels.append(f"Stage 2 ML: {s2:,}  ({s2/total*100:.1f}%)")
            colours.append(PIE_S2)
        if benign > 0:
            slices.append(benign)
            legend_labels.append(f"Benign: {benign:,}  ({benign/total*100:.1f}%)")
            colours.append(PIE_CLEAN)

        if not slices:
            self._ax.text(0.5, 0.5, "No incidents detected",
                          ha="center", va="center",
                          color=C_GREEN, fontsize=12,
                          transform=self._ax.transAxes)
            self._ax.axis("off")
        else:
            # No labels/autopct on the wedges — all text goes in the legend.
            # This prevents the tiny Benign / Stage-2 slices from producing
            # overlapping annotations that stack on top of each other.
            wedges, _ = self._ax.pie(
                slices,
                colors=colours,
                startangle=90,
                wedgeprops={"linewidth": 1.5, "edgecolor": C_PANEL},
            )

            # Legend anchored to the right of the pie, vertically centred.
            leg = self._ax.legend(
                wedges,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=9,
                framealpha=0,           # transparent fill — blends with dark panel
                borderpad=0.6,
            )
            # Force legend text colour to match the dark theme.
            # (labelcolor= kwarg needs matplotlib ≥ 3.5; set_color() works on all versions)
            for txt in leg.get_texts():
                txt.set_color(C_TEXT)
            leg.get_frame().set_linewidth(0)    # no visible border box

        # Leave room on the right for the legend — do NOT call tight_layout
        # here because it would clip the externally-anchored legend.
        self._fig.subplots_adjust(left=0.02, right=0.62, top=0.95, bottom=0.05)
        self._canvas.draw()


# ═══════════════════════════════════════════════════════════════════════
# TAB 2  -  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════

class TrainTab(ctk.CTkFrame):
    """
    Runs train_model.py as a subprocess and parses stdout to:
      - Advance the progress bar through training stages
      - Display final accuracy as a large, prominent label
      - Fill five CV Fold cards with per-fold accuracy scores
      - Show CV mean ± std dev cleanly
    """

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self._process: Optional[subprocess.Popen] = None
        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(20, 4))
        ctk.CTkLabel(hdr, text="Model Training",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=C_BLUE).pack(side="left")
        self._model_status = ctk.CTkLabel(hdr, text="",
                                           font=ctk.CTkFont(size=11))
        self._model_status.pack(side="right")
        self._refresh_model_status()

        # ── Controls ────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=10)
        ctrl.pack(fill="x", padx=20, pady=(0, 8))

        path_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        path_row.pack(fill="x", padx=16, pady=(12, 4))
        ctk.CTkLabel(path_row, text="Dataset CSV:", width=100,
                     anchor="w").pack(side="left")
        self._dataset_var = ctk.StringVar(
            value=str(PROJECT_ROOT / "data" / "ciciot2023_train01.csv"))
        ctk.CTkEntry(path_row, textvariable=self._dataset_var,
                     width=420).pack(side="left", padx=6)
        ctk.CTkButton(path_row, text="Browse", width=80,
                      command=self._browse_dataset).pack(side="left")

        frac_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        frac_row.pack(fill="x", padx=16, pady=(4, 2))
        ctk.CTkLabel(frac_row, text="Sample fraction:", width=120,
                     anchor="w").pack(side="left")
        self._frac_var   = ctk.DoubleVar(value=1.0)
        self._frac_label = ctk.CTkLabel(frac_row, text="100%", width=50)
        ctk.CTkSlider(frac_row, from_=0.05, to=1.0, number_of_steps=19,
                      variable=self._frac_var,
                      command=lambda v: self._frac_label.configure(
                          text=f"{v*100:.0f}%"),
                      width=300).pack(side="left", padx=6)
        self._frac_label.pack(side="left")

        chunk_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        chunk_row.pack(fill="x", padx=16, pady=(2, 4))
        ctk.CTkLabel(chunk_row, text="Max chunks:", width=120,
                     anchor="w").pack(side="left")
        self._chunks_var   = ctk.IntVar(value=2)   # demo default: 100k rows, ~30s on 1 core
        self._chunks_label = ctk.CTkLabel(chunk_row,
                                           text="2  (100,000 rows)", width=140)

        def _upd_chunk(v):
            n = int(float(v))
            self._chunks_label.configure(text=f"{n}  ({n*50_000:,} rows)")

        ctk.CTkSlider(chunk_row, from_=1, to=20, number_of_steps=19,
                      variable=self._chunks_var,
                      command=_upd_chunk,
                      width=300).pack(side="left", padx=6)
        self._chunks_label.pack(side="left")

        btn_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(4, 12))
        self._train_btn = ctk.CTkButton(
            btn_row, text="  Train Model",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=160, command=self._start)
        self._train_btn.pack(side="left")
        self._stop_btn = ctk.CTkButton(
            btn_row, text="  Stop", width=90, state="disabled",
            fg_color="#dc2626", hover_color="#b91c1c",
            command=self._stop)
        self._stop_btn.pack(side="left", padx=8)
        self._status_lbl = ctk.CTkLabel(btn_row, text="Ready",
                                         text_color=C_MUTED)
        self._status_lbl.pack(side="left", padx=12)

        # ── Progress bar ─────────────────────────────────────────────────
        pb_wrap = ctk.CTkFrame(self, fg_color="transparent")
        pb_wrap.pack(fill="x", padx=20, pady=(0, 10))
        self._pb = ctk.CTkProgressBar(pb_wrap, height=8,
                                       progress_color=C_BLUE)
        self._pb.pack(fill="x")
        self._pb.set(0)

        # ── Accuracy card (large central display) ────────────────────────
        acc_panel = ctk.CTkFrame(self, fg_color=C_PANEL,
                                  corner_radius=12, border_width=1,
                                  border_color=C_BORDER)
        acc_panel.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(acc_panel, text="MODEL ACCURACY",
                     font=ctk.CTkFont(size=12),
                     text_color=C_MUTED).pack(pady=(18, 2))

        self._acc_label = ctk.CTkLabel(
            acc_panel, text="—",
            font=ctk.CTkFont(size=52, weight="bold"),
            text_color=C_MUTED)
        self._acc_label.pack(pady=(0, 4))

        self._acc_sub = ctk.CTkLabel(
            acc_panel, text="Train the model to see results",
            font=ctk.CTkFont(size=11), text_color=C_MUTED)
        self._acc_sub.pack(pady=(0, 18))

        # ── 5-Fold CV cards ──────────────────────────────────────────────
        cv_section = ctk.CTkFrame(self, fg_color="transparent")
        cv_section.pack(fill="x", padx=20, pady=(0, 8))

        cv_header = ctk.CTkFrame(cv_section, fg_color="transparent")
        cv_header.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(cv_header,
                     text="5-Fold Cross-Validation",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C_TEXT).pack(side="left")
        self._cv_summary = ctk.CTkLabel(cv_header, text="",
                                         font=ctk.CTkFont(size=11),
                                         text_color=C_MUTED)
        self._cv_summary.pack(side="right")

        folds_row = ctk.CTkFrame(cv_section, fg_color="transparent")
        folds_row.pack(fill="x")
        folds_row.columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="f")

        self._cv_cards = []
        for i in range(5):
            card = CVFoldCard(folds_row, i + 1)
            card.grid(row=0, column=i,
                      padx=(0 if i == 0 else 4, 4 if i < 4 else 0),
                      sticky="nsew")
            self._cv_cards.append(card)

        # ── Collapsible training log ──────────────────────────────────────
        self._log_open = False
        self._log_toggle = ctk.CTkButton(
            self, text="  Show Training Log", anchor="w",
            fg_color="transparent", hover_color=C_PANEL,
            font=ctk.CTkFont(size=11), text_color=C_MUTED,
            command=self._toggle_log)
        self._log_toggle.pack(fill="x", padx=20, pady=(4, 2))

        self._log_container = ctk.CTkFrame(self, fg_color="transparent")
        self._log = LogBox(self._log_container, height=130)
        self._log.pack(fill="both", expand=True)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _toggle_log(self):
        if self._log_open:
            self._log_container.pack_forget()
            self._log_toggle.configure(text="  Show Training Log")
        else:
            self._log_container.pack(fill="x", padx=20, pady=(0, 8))
            self._log_toggle.configure(text="  Hide Training Log")
        self._log_open = not self._log_open

    def _refresh_model_status(self):
        if MODEL_PATH.exists():
            kb = MODEL_PATH.stat().st_size / 1024
            self._model_status.configure(
                text=f"  Existing model: {kb:,.0f} KB", text_color=C_GREEN)
        else:
            self._model_status.configure(
                text="  No model file yet", text_color=C_MUTED)

    def _browse_dataset(self):
        p = filedialog.askopenfilename(
            title="Select CICIoT2023 CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT / "data"))
        if p:
            self._dataset_var.set(p)

    # ── Training run ─────────────────────────────────────────────────────

    def _start(self):
        dataset = Path(self._dataset_var.get())
        if not dataset.exists():
            messagebox.showerror("File not found",
                                 f"Dataset not found:\n{dataset}")
            return

        self._log.clear()
        self._train_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._status_lbl.configure(text="Starting...", text_color=C_YELLOW)
        self._pb.configure(mode="indeterminate", progress_color=C_BLUE)
        self._pb.start()
        # Reset displays
        self._acc_label.configure(text="—", text_color=C_MUTED)
        self._acc_sub.configure(text="Training in progress...")
        self._cv_summary.configure(text="")
        for card in self._cv_cards:
            card._lbl.configure(text="—", text_color=C_BLUE)

        env_extra = {
            "NIDS_DATASET_PATH":    str(dataset),
            "NIDS_SAMPLE_FRACTION": str(self._frac_var.get()),
            "NIDS_MAX_CHUNKS":      str(int(self._chunks_var.get())),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS":      "1",
            # Tells train_model.py to save results to disk instead of calling
            # serve_forever() — that would block the subprocess from exiting.
            "NIDS_GUI_MODE":        "1",
        }

        if _FROZEN:
            cmd = [sys.executable, "--nids-train",
                   "--dataset", str(dataset),
                   "--fraction", str(self._frac_var.get()),
                   "--chunks",   str(int(self._chunks_var.get()))]
        else:
            cmd = [PYTHON, "-u", TRAIN_SCRIPT]

        threading.Thread(target=self._run_subprocess,
                         args=(cmd, env_extra), daemon=True).start()

    def _stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log.append("\n[GUI] Training stopped by user.\n")
        self._finish(cancelled=True)

    def _run_subprocess(self, cmd, env_extra):
        env = os.environ.copy()
        env.update(env_extra)
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), env=env,
            startupinfo=_POPEN_SI)   # Windows: suppress console window flicker
        for line in self._process.stdout:
            self._log.append(line)
            self._on_line(line)
        self._process.wait()
        rc = self._process.returncode
        self._log.append(f"\n[GUI] Process exited (code {rc}).\n")
        self.after(0, self._finish, rc != 0)

    def _on_line(self, line: str):
        """Parse training stdout to update progress bar and metric cards."""
        # ── Progress milestones ─────────────────────────────────────────
        if "Chunk" in line and "read" in line:
            # Extract chunk number for fine-grained progress
            m = re.search(r"Chunk\s+(\d+)", line)
            if m:
                frac = min(0.05 + int(m.group(1)) * 0.08, 0.40)
                self.after(0, self._set_pb, frac,
                           f"Loading data (chunk {m.group(1)})...", C_YELLOW)

        elif "Training RandomForest" in line or "Training complete" in line:
            self.after(0, self._set_pb, 0.55,
                       "Training Random Forest...", C_YELLOW)

        elif "Evaluating on held-out" in line:
            self.after(0, self._set_pb, 0.70, "Evaluating...", C_YELLOW)

        elif "5-Fold Cross-Validation" in line or "Running 5-Fold" in line:
            self.after(0, self._set_pb, 0.80,
                       "Running 5-Fold CV...", C_YELLOW)

        elif "SHAP" in line and "analysis" in line.lower():
            self.after(0, self._set_pb, 0.88,
                       "Generating SHAP plot...", C_YELLOW)

        elif "Saving model" in line:
            self.after(0, self._set_pb, 0.95, "Saving model...", C_YELLOW)

        # ── Accuracy ────────────────────────────────────────────────────
        m = re.search(
            r"ACCURACY SCORE\s*:\s*[\d.]+\s+\(([\d.]+)\s*%\)", line)
        if m:
            pct = float(m.group(1))
            colour = C_GREEN if pct >= 90 else (C_YELLOW if pct >= 75 else C_RED)
            self.after(0, self._acc_label.configure,
                       {"text": f"{pct:.2f}%", "text_color": colour})
            self.after(0, self._acc_sub.configure,
                       {"text": "Test set accuracy  (100-tree Random Forest)"})

        # ── CV fold scores ───────────────────────────────────────────────
        m = re.search(r"Fold scores\s*:\s*(.+)", line)
        if m:
            parts = m.group(1).replace("|", " ").split()
            scores = []
            for p in parts:
                try:
                    scores.append(float(p))
                except ValueError:
                    pass
            for i, score in enumerate(scores[:5]):
                self._cv_cards[i].set_score(score)

        # ── CV mean / std ────────────────────────────────────────────────
        m_mean = re.search(r"Mean acc\.\s*:\s*[\d.]+\s+\(([\d.]+)%\)", line)
        m_std  = re.search(r"Std dev\s*:\s*[\d.]+\s+\(\+/- ([\d.]+)%\)", line)
        if m_mean:
            self._cv_mean = float(m_mean.group(1))
        if m_std:
            self._cv_std = float(m_std.group(1))
        if hasattr(self, "_cv_mean") and hasattr(self, "_cv_std"):
            self.after(0, self._cv_summary.configure,
                       {"text": f"Mean: {self._cv_mean:.2f}%   "
                                f"+/-  {self._cv_std:.2f}%",
                        "text_color": C_BLUE})

    def _set_pb(self, val: float, status: str = "", colour: str = C_BLUE):
        self._pb.stop()
        self._pb.configure(mode="determinate", progress_color=colour)
        self._pb.set(val)
        if status:
            self._status_lbl.configure(text=status, text_color=colour)

    def _finish(self, cancelled: bool = False):
        self._pb.stop()
        self._pb.configure(mode="determinate")
        self._train_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._refresh_model_status()

        if cancelled:
            self._pb.set(0)
            self._status_lbl.configure(text="Stopped.", text_color=C_MUTED)
            return

        self._pb.set(1.0)
        self._pb.configure(progress_color=C_GREEN)
        if MODEL_PATH.exists():
            kb = MODEL_PATH.stat().st_size / 1024
            self._status_lbl.configure(
                text=f"Done.  Model saved ({kb:,.0f} KB)", text_color=C_GREEN)
            self._acc_sub.configure(
                text=f"Test set accuracy  (saved: {kb:,.0f} KB)")
        else:
            self._status_lbl.configure(
                text="Finished — check log above.", text_color=C_MUTED)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3  -  REPORTS BROWSER
# ═══════════════════════════════════════════════════════════════════════

class ResultsTab(ctk.CTkFrame):
    """Browse and inspect saved JSON incident reports."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Incident Reports",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=C_BLUE,
                     ).pack(anchor="w", padx=20, pady=(20, 4))

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=4)
        ctk.CTkButton(top, text="Refresh list", width=120,
                      command=self._refresh_list).pack(side="left")
        ctk.CTkButton(top, text="Open reports folder", width=160,
                      command=self._open_folder).pack(side="left", padx=8)

        pane = ctk.CTkFrame(self, fg_color="transparent")
        pane.pack(fill="both", expand=True, padx=20, pady=8)

        left = ctk.CTkFrame(pane)
        left.pack(side="left", fill="y", padx=(0, 8))
        ctk.CTkLabel(left, text="Saved reports:").pack(anchor="w", padx=8, pady=4)
        self._listbox    = ctk.CTkScrollableFrame(left, width=240, height=400)
        self._listbox.pack(fill="y", expand=True, padx=4, pady=4)
        self._list_btns: List = []

        right = ctk.CTkFrame(pane)
        right.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(right, text="Incident details:").pack(
            anchor="w", padx=8, pady=4)
        self._detail = LogBox(right, height=400)
        self._detail.pack(fill="both", expand=True, padx=4, pady=4)

        self._refresh_list()

    def _refresh_list(self):
        for btn in self._list_btns:
            btn.destroy()
        self._list_btns.clear()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        reports = sorted(REPORTS_DIR.glob("*.json"), reverse=True)
        if not reports:
            ctk.CTkLabel(self._listbox, text="No reports yet.",
                         text_color=C_MUTED).pack()
            return
        for rp in reports:
            btn = ctk.CTkButton(
                self._listbox, text=rp.name, anchor="w", width=220,
                fg_color="transparent", hover_color=("gray70", "gray30"),
                command=lambda p=rp: self._load_report(p))
            btn.pack(fill="x", pady=2)
            self._list_btns.append(btn)

    def _load_report(self, path: Path):
        self._detail.clear()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self._detail.append(f"Error: {e}")
            return

        incidents = data.get("incidents", [])
        self._detail.append(f"Report  : {path.name}\n")
        self._detail.append(f"Total   : {data.get('total_incidents', len(incidents))} incident(s)\n")
        self._detail.append("=" * 60 + "\n\n")

        if not incidents:
            self._detail.append("No incidents — all traffic was clean.\n")
            return

        for i, inc in enumerate(incidents, 1):
            self._detail.append(f"[{i}] {inc.get('rule','')}\n")
            self._detail.append(f"    Timestamp  : {inc.get('timestamp','')}\n")
            self._detail.append(f"    Source IP  : {inc.get('source_ip','N/A')}\n")
            if "destination_ip" in inc:
                self._detail.append(f"    Dest IP    : {inc['destination_ip']}\n")
            if "ml_confidence" in inc:
                self._detail.append(
                    f"    ML Conf.   : {float(inc['ml_confidence'])*100:.1f}%\n")
            if "mitre_tactic" in inc:
                self._detail.append(f"    MITRE Tactic    : {inc['mitre_tactic']}\n")
                self._detail.append(f"    MITRE Technique : {inc['mitre_technique']}\n")
            if "flagged_flow_count" in inc:
                self._detail.append(
                    f"    Flow count : {inc['flagged_flow_count']:,}\n")
            self._detail.append(f"    Details    : {inc.get('details','')}\n\n")

    def _open_folder(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(REPORTS_DIR))


# ═══════════════════════════════════════════════════════════════════════
# TAB 4  -  ANALYTICS
# ═══════════════════════════════════════════════════════════════════════

class AnalyticsTab(ctk.CTkFrame):
    """
    Embedded Matplotlib charts (pie + MITRE bar).
    Auto-refreshes after every successful detection run.
    """

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self._build_ui()
        self.after(300, self.refresh)

    def _build_ui(self):
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(20, 4))
        ctk.CTkLabel(hdr, text="Detection Analytics",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=C_BLUE).pack(side="left")
        ctk.CTkButton(hdr, text="Refresh Charts", width=140,
                      command=self.refresh).pack(side="right")
        self._status_lbl = ctk.CTkLabel(hdr, text="", text_color=C_MUTED,
                                         font=ctk.CTkFont(size=11))
        self._status_lbl.pack(side="right", padx=12)

        charts = ctk.CTkFrame(self, fg_color="transparent")
        charts.pack(fill="both", expand=True, padx=20, pady=8)

        left  = ctk.CTkFrame(charts, fg_color=_DARK_BG, corner_radius=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._fig_pie = Figure(figsize=(5, 4), facecolor=_DARK_BG)
        self._ax_pie  = self._fig_pie.add_subplot(111)
        self._cpie    = FigureCanvasTkAgg(self._fig_pie, master=left)
        self._cpie.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        right = ctk.CTkFrame(charts, fg_color=_DARK_BG, corner_radius=8)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._fig_bar = Figure(figsize=(5, 4), facecolor=_DARK_BG)
        self._ax_bar  = self._fig_bar.add_subplot(111)
        self._cbar    = FigureCanvasTkAgg(self._fig_bar, master=right)
        self._cbar.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        self._placeholder()

    def refresh(self):
        rp = self._latest_report()
        if rp is None:
            self._status_lbl.configure(
                text="No reports found - run detection first.", text_color=C_MUTED)
            return
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except Exception as e:
            self._status_lbl.configure(text=f"Error: {e}", text_color=C_RED)
            return
        incidents = data.get("incidents", [])
        self._status_lbl.configure(
            text=f"Loaded: {rp.name}  ({len(incidents)} incident(s))",
            text_color=C_GREEN)
        self._draw_pie(incidents)
        self._draw_bar(incidents)

    def _placeholder(self):
        for ax, fig, canvas, title in [
            (self._ax_pie, self._fig_pie, self._cpie, "Flow Distribution"),
            (self._ax_bar, self._fig_bar, self._cbar, "MITRE ATT&CK Tactics"),
        ]:
            ax.clear(); ax.set_facecolor(_DARK_BG)
            ax.text(0.5, 0.5, "No data yet\nRun detection to populate",
                    ha="center", va="center", color="gray",
                    fontsize=10, transform=ax.transAxes)
            ax.set_title(title, color=_DARK_TEXT, fontsize=12, pad=10)
            ax.axis("off"); fig.tight_layout(); canvas.draw()

    def _draw_pie(self, incidents):
        s1, s2, other = 0, 0, 0
        for inc in incidents:
            n = int(inc.get("flagged_flow_count", 1))
            r = inc.get("rule", "")
            if "Stage 1" in r: s1 += n
            elif "Stage 2" in r: s2 += n
            else: other += n

        ax = self._ax_pie
        ax.clear(); ax.set_facecolor(_DARK_BG)
        if s1 == 0 and s2 == 0 and other == 0:
            ax.text(0.5, 0.5, "No flagged flows",
                    ha="center", va="center", color="gray", fontsize=10,
                    transform=ax.transAxes)
            ax.axis("off")
        else:
            slices, labels, colours = [], [], []
            if s1:
                slices.append(s1); labels.append(f"Stage 1\n({s1:,})"); colours.append(_CLR_S1)
            if s2:
                slices.append(s2); labels.append(f"Stage 2 ML\n({s2:,})"); colours.append(_CLR_S2)
            if other:
                slices.append(other); labels.append(f"Other\n({other:,})"); colours.append(_CLR_OTHER)
            _, _, auts = ax.pie(slices, labels=labels, colors=colours,
                                autopct="%1.1f%%", startangle=90,
                                textprops={"color": _DARK_TEXT, "fontsize": 9})
            for at in auts:
                at.set_color(_DARK_BG); at.set_fontsize(8)

        ax.set_title("Flagged Flows: Stage 1 vs Stage 2",
                     color=_DARK_TEXT, fontsize=12, pad=10)
        self._fig_pie.tight_layout(); self._cpie.draw()

    def _draw_bar(self, incidents):
        tc: Dict[str, int] = {}
        for inc in incidents:
            tactic = inc.get("mitre_tactic")
            if not tactic or tactic == "Unknown":
                rule = inc.get("rule", "Unknown")
                tactic = ("Volumetric" if "Stage 1" in rule else
                          "ML Anomaly" if "Stage 2" in rule else "Rule-Based")
            short = tactic.split("—")[-1].strip() if "—" in tactic else tactic
            tc[short] = tc.get(short, 0) + 1

        ax = self._ax_bar
        ax.clear(); ax.set_facecolor(_DARK_BG)
        self._fig_bar.set_facecolor(_DARK_BG)

        if not tc:
            ax.text(0.5, 0.5, "No incident data", ha="center", va="center",
                    color="gray", fontsize=10, transform=ax.transAxes)
            ax.axis("off")
        else:
            items = sorted(tc.items(), key=lambda x: x[1], reverse=True)
            lbls  = [i[0] for i in items]
            vals  = [i[1] for i in items]
            clrs  = [(_CLR_S1 if "Volum" in l else
                      _CLR_S2 if "ML" in l else _CLR_OTHER)
                     for l in lbls]
            bars  = ax.barh(range(len(lbls)), vals, color=clrs, height=0.6)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() + 0.05,
                        bar.get_y() + bar.get_height() / 2,
                        str(v), va="center", color=_DARK_TEXT, fontsize=9)
            ax.set_yticks(list(range(len(lbls))))
            ax.set_yticklabels(lbls, color=_DARK_TEXT, fontsize=8)
            ax.tick_params(axis="x", colors=_DARK_TEXT, labelsize=8)
            ax.set_xlabel("Incident Count", color=_DARK_TEXT, fontsize=9)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            for sp in ["bottom", "left"]:
                ax.spines[sp].set_color(C_BORDER)
            ax.set_facecolor(_DARK_BG)

        ax.set_title("MITRE ATT&CK Tactic Distribution",
                     color=_DARK_TEXT, fontsize=12, pad=10)
        self._fig_bar.tight_layout(); self._cbar.draw()

    @staticmethod
    def _latest_report() -> Optional[Path]:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        reports = sorted(REPORTS_DIR.glob("*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        return reports[0] if reports else None


# ═══════════════════════════════════════════════════════════════════════
# ROOT APPLICATION WINDOW
# ═══════════════════════════════════════════════════════════════════════

class NIDSApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Hybrid NIDS  |  Network Intrusion Detection System")
        self.geometry("1020x740")
        self.minsize(880, 640)
        self.configure(fg_color=C_WIN)

        # Window icon
        if ICON_PATH.exists():
            try:
                self.iconbitmap(str(ICON_PATH))
            except Exception:
                pass

        # ── Header bar ──────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, height=52, corner_radius=0,
                           fg_color=C_PANEL)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr,
                     text="  Hybrid NIDS  |  Network Intrusion Detection System",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=C_BLUE).pack(side="left", padx=16)
        ctk.CTkLabel(hdr, text="CICIoT2023  |  Random Forest  |  MITRE ATT&CK",
                     text_color=C_MUTED).pack(side="right", padx=16)

        # ── Tab view ─────────────────────────────────────────────────────
        tabs = ctk.CTkTabview(self, fg_color=C_WIN,
                               segmented_button_fg_color=C_PANEL,
                               segmented_button_selected_color=C_BLUE,
                               segmented_button_selected_hover_color="#0ea5e9",
                               segmented_button_unselected_color=C_PANEL,
                               segmented_button_unselected_hover_color="#334155",
                               text_color=C_TEXT)
        tabs.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        for name in ("Detection Dashboard", "Model Training",
                     "Reports", "Analytics"):
            tabs.add(name)

        # Mount tabs
        detect_tab    = DetectTab(tabs.tab("Detection Dashboard"))
        detect_tab.pack(fill="both", expand=True)

        TrainTab(tabs.tab("Model Training")).pack(fill="both", expand=True)
        ResultsTab(tabs.tab("Reports")).pack(fill="both", expand=True)

        analytics_tab = AnalyticsTab(tabs.tab("Analytics"))
        analytics_tab.pack(fill="both", expand=True)

        # Wire analytics auto-refresh to detection completion
        detect_tab.set_analytics_tab(analytics_tab)


# ═══════════════════════════════════════════════════════════════════════
# PYINSTALLER SUBPROCESS DISPATCHER + ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def _dispatch_train():
    """
    Called when the .exe is launched with --nids-train.
    Reads CLI args, sets env vars, then runs train_model.main().
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--nids-train",  action="store_true")
    ap.add_argument("--dataset",  default="data/ciciot2023_train01.csv")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--chunks",   type=int,   default=4)
    args, _ = ap.parse_known_args()

    os.environ["NIDS_DATASET_PATH"]    = str(writable_path(args.dataset)
                                             if not Path(args.dataset).is_absolute()
                                             else args.dataset)
    os.environ["NIDS_SAMPLE_FRACTION"] = str(args.fraction)
    os.environ["NIDS_MAX_CHUNKS"]      = str(args.chunks)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["NIDS_GUI_MODE"]        = "1"   # ensure flag is set in frozen path

    import train_model
    train_model.main()


def _dispatch_detect():
    """
    Called when the .exe is launched with --nids-detect.
    Rebuilds sys.argv to look like main.py was called directly.
    """
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--nids-detect",  action="store_true")
    ap.add_argument("--logfile",   required=True)
    ap.add_argument("--output",    default="reports/incident_report")
    ap.add_argument("--hybrid",    action="store_true")
    args, _ = ap.parse_known_args()

    # Reconstruct sys.argv for main.py's own argparse
    sys.argv = [sys.argv[0],
                "--logfile", args.logfile,
                "--output",  args.output]
    if args.hybrid:
        sys.argv.append("--hybrid")

    import main as nids_main
    nids_main.main()


def main():
    # Route to training or detection if launched as a subprocess from the exe.
    if "--nids-train"  in sys.argv:
        _dispatch_train();  return
    if "--nids-detect" in sys.argv:
        _dispatch_detect(); return

    # Normal GUI launch
    app = NIDSApp()
    app.mainloop()


if __name__ == "__main__":
    # Needed for PyInstaller/loky worker interception when running on Windows.
    multiprocessing.freeze_support()
    main()
