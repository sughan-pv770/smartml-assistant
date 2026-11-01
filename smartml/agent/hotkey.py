"""
smartml.agent.hotkey
====================
Global keyboard shortcut listener built lazily on ``pynput``.
Supports human-readable combos like ``"ctrl+alt+m"``, ``"<ctrl>+<alt>+m"``,
or ``"esc"`` / ``"<esc>"``.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

_log = logging.getLogger("smartml.agent")

_SPECIAL_KEYS = {
    "ctrl", "control", "alt", "shift", "cmd", "win", "super",
    "esc", "escape", "space", "tab", "enter", "return",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
}


def parse_hotkey(combo: str) -> str:
    """Normalise a human-friendly shortcut (e.g. ``'Ctrl+Alt+M'``) into pynput format (``'<ctrl>+<alt>+m'``)."""
    parts = [p.strip().lower().strip("<>") for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid empty hotkey string: {combo!r}")

    out: list[str] = []
    for p in parts:
        if p == "control":
            p = "ctrl"
        elif p == "escape":
            p = "esc"
        elif p in ("win", "super"):
            p = "cmd"

        if p in _SPECIAL_KEYS:
            out.append(f"<{p}>")
        elif len(p) == 1:
            out.append(p)
        else:
            raise ValueError(f"Unrecognised key token {p!r} in hotkey {combo!r}")
    return "+".join(out)


class HotkeyListener:
    """Global keyboard shortcut listener wrapping ``pynput.keyboard.GlobalHotKeys``."""

    def __init__(
        self,
        hotkey: str,
        callback: Callable[[], None],
    ) -> None:
        self.hotkey = parse_hotkey(hotkey)
        self.callback = callback
        self._listener = None

    def start(self) -> None:
        """Start the background global hotkey listener thread."""
        if self._listener is not None:
            return
        try:
            from pynput import keyboard  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Global hotkey listening requires 'pynput'. "
                "Install with: pip install pynput (or run in --text mode)."
            ) from exc

        self._listener = keyboard.GlobalHotKeys({self.hotkey: self.callback})
        self._listener.daemon = True
        self._listener.start()
        _log.info("Registered global hotkey: %s", self.hotkey)

    def stop(self) -> None:
        """Stop and unregister the global hotkey listener."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

