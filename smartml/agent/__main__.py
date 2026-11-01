"""Entry point for ``python -m smartml.agent``."""

from __future__ import annotations

import multiprocessing
import sys

from .voice_agent import main_cli

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main_cli())

