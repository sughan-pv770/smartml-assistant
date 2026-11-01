"""
smartml.mial
============
Mial — Hotkey-Triggered Offline Voice AI Agent built on top of ``smartml.agent``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BrainResponse",
    "CancellableRunner",
    "MIAL_SYSTEM_PROMPT",
    "MialConfig",
    "MialController",
    "MialLLMBrain",
    "MialPopupApp",
    "MialSpeaker",
    "MialState",
    "SmartMLToolExecutor",
    "is_heavy_command",
    "main",
    "open_file_dialog_subprocess",
]

_LAZY_MAP = {
    "main": (".cli", "main"),
    "MialConfig": (".config", "MialConfig"),
    "MialController": (".controller", "MialController"),
    "MialState": (".controller", "MialState"),
    "open_file_dialog_subprocess": (".file_dialog", "open_file_dialog_subprocess"),
    "BrainResponse": (".llm_brain", "BrainResponse"),
    "MIAL_SYSTEM_PROMPT": (".llm_brain", "MIAL_SYSTEM_PROMPT"),
    "MialLLMBrain": (".llm_brain", "MialLLMBrain"),
    "MialPopupApp": (".popup_ui", "MialPopupApp"),
    "CancellableRunner": (".runner", "CancellableRunner"),
    "is_heavy_command": (".runner", "is_heavy_command"),
    "MialSpeaker": (".speaker", "MialSpeaker"),
    "SmartMLToolExecutor": (".tools", "SmartMLToolExecutor"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        import importlib
        mod_name, attr_name = _LAZY_MAP[name]
        mod = importlib.import_module(mod_name, __name__)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


