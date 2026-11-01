"""
smartml.mial.config
===================
Configuration dataclass and TOML loader for the Mial voice AI agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _default_allowed_dirs() -> list[str]:
    home = Path.home()
    candidates = [
        Path.cwd(),
        home / "Downloads",
        home / "Desktop",
        home / "Documents",
    ]
    return [str(p) for p in candidates]


@dataclass
class MialConfig:
    """Runtime settings for Mial.

    Attributes:
        hotkey: Global keyboard shortcut to wake Mial (default ``"<ctrl>+<alt>+m"``).
        cancel_key: Shortcut to abort a running heavy training job (default ``"<esc>"``).
        stt_engine: Speech-to-text backend. Pinned to ``"whisper"`` by default so
            zero audio is ever sent to the cloud unless ``online_stt=True``.
        whisper_model: Whisper model size (``"base"``, ``"small"``, ``"tiny"``).
            Note: downloads ~150 MB on first run and caches locally for offline use.
        online_stt: Explicit opt-in required to allow cloud STT (Google).
        offline_only: When True, disables all cloud LLM calls (such as NVIDIA NIM) even if
            ``NVIDIA_API_KEY`` is set in the environment. When False and ``NVIDIA_API_KEY``
            is set, spoken transcripts and aggregate SmartML tool summaries (never raw audio
            or raw DataFrame cell values) are sent to NVIDIA NIM.
        speak_replies: Whether Mial speaks responses aloud.
        follow_up_window_s: Seconds to listen for a follow-up question after replying.
        max_retries: Number of consecutive silent/unclear attempts before returning to IDLE.
        output_dir: Directory where reports, plots, models, and logs are saved.
        allowed_dirs: Folders where Mial is allowed to search and load datasets.
        greeting: Spoken greeting when triggered via hotkey.
    """

    hotkey: str = "<ctrl>+<alt>+m"
    cancel_key: str = "<esc>"
    stt_engine: str = "whisper"
    whisper_model: str = "base"
    online_stt: bool = False
    online_tts: bool = False
    offline_only: bool = False
    audit_log: bool = False
    speak_replies: bool = True
    voice_name: str = "en-IN-NeerjaNeural"
    follow_up_window_s: float = 5.0
    max_retries: int = 2
    output_dir: str = "smartml_output"
    allowed_dirs: list[str] = field(default_factory=_default_allowed_dirs)
    greeting: str = "Hi, I'm Mial. What would you like to do?"

    @property
    def effective_stt_engine(self) -> str:
        """Return the effective STT engine respecting the strict offline default."""
        if self.stt_engine.lower() in ("google", "online"):
            if not self.online_stt:
                return "whisper"
            return "google"
        if self.stt_engine.lower() == "auto":
            return "auto" if self.online_stt else "whisper"
        return "whisper"

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "MialConfig":
        """Load configuration from ``mial.toml`` or ``~/.smartml/mial.toml`` if present."""
        search_paths: list[Path] = []
        if path is not None:
            search_paths.append(Path(path))
        search_paths.extend([
            Path.cwd() / "mial.toml",
            Path.home() / ".smartml" / "mial.toml",
        ])

        toml_path: Optional[Path] = None
        for p in search_paths:
            if p.is_file():
                toml_path = p
                break

        if toml_path is None:
            return cls()

        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                return cls()

        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        section = raw.get("mial", raw)
        cfg = cls()
        for key in (
            "hotkey",
            "cancel_key",
            "stt_engine",
            "whisper_model",
            "online_stt",
            "online_tts",
            "offline_only",
            "audit_log",
            "speak_replies",
            "follow_up_window_s",
            "max_retries",
            "output_dir",
            "allowed_dirs",
            "greeting",
        ):
            if key in section:
                setattr(cfg, key, section[key])
        return cfg

