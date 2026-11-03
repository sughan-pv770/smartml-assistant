"""
run_mial.py
===========
Direct launcher for the Mial Desktop Voice Assistant for SmartML.
Double-click or run: python run_mial.py

Features:
- Global wake hotkey (Ctrl+Alt+M) floats Mial on top of any active window
- Speaks with natural voice and listens via microphone
- Executes AutoML workflows (clean, train 9 models, tune, explain, report)
- Answers any AI, ML, or general hackathon questions powered by NVIDIA Nemotron
"""

from __future__ import annotations

import multiprocessing
import sys

from smartml.mial.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())

