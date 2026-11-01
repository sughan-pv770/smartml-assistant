"""
smartml.mial.file_dialog
========================
Subprocess-isolated native file-picker dialog.

Why a subprocess?
-----------------
Tkinter and Win32 common file dialogs must run on the main thread of a process;
invoking ``filedialog.askopenfilename`` inside a ``pynput`` callback thread or
worker thread deadlocks or raises ``RuntimeError: main thread is not in main loop``.
Running this module as ``python -m smartml.mial.file_dialog`` guarantees the
dialog runs on the child process's main thread and prints the chosen path to stdout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..agent.nlp import ALLOWED_DATA_EXTENSIONS


def open_file_dialog_subprocess(
    initial_dir: Optional[str | Path] = None,
    timeout_s: float = 120.0,
) -> Optional[str]:
    """Launch a short child process to let the user pick a dataset file.

    Returns:
        The absolute file path selected by the user, or None if dismissed/cancelled.
    """
    start_dir = str(initial_dir or Path.cwd())
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "smartml.mial.file_dialog", start_dir],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    chosen = proc.stdout.strip()
    if not chosen:
        return None
    p = Path(chosen)
    if p.is_file() and p.suffix.lower() in ALLOWED_DATA_EXTENSIONS:
        return str(p.resolve())
    return None


def _run_dialog_main() -> int:
    initial_dir = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else os.getcwd()
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Mial — Select a Dataset File",
            initialdir=initial_dir,
            filetypes=[
                ("Dataset files", "*.csv *.tsv *.xlsx *.xls *.json *.parquet *.feather"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if chosen:
            sys.stdout.write(chosen.strip())
            sys.stdout.flush()
            return 0
    except Exception:
        pass
    return 1


if __name__ == "__main__":
    sys.exit(_run_dialog_main())

