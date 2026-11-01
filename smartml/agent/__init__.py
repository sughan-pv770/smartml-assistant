"""
smartml.agent
=============
Offline-first voice and text agent infrastructure for SmartML.
"""

from __future__ import annotations

from .audio import (
    AudioRecorder,
    EnergyVAD,
    MicrophoneUnavailableError,
    SpeechToText,
    STTFailureError,
    TextToSpeech,
    beep,
)
from .hotkey import HotkeyListener, parse_hotkey
from .nlp import (
    ALLOWED_DATA_EXTENSIONS,
    find_dataset_candidates,
    find_dataset_file,
    is_within_allowed_dirs,
    normalize_transcript,
    speakable,
    split_compound_commands,
)
from .voice_agent import SmartMLAgent, VoiceAgent, _parse_voice_command, main_cli

__all__ = [
    "ALLOWED_DATA_EXTENSIONS",
    "AudioRecorder",
    "EnergyVAD",
    "HotkeyListener",
    "MicrophoneUnavailableError",
    "SmartMLAgent",
    "SpeechToText",
    "STTFailureError",
    "TextToSpeech",
    "VoiceAgent",
    "_parse_voice_command",
    "beep",
    "find_dataset_candidates",
    "find_dataset_file",
    "is_within_allowed_dirs",
    "main_cli",
    "normalize_transcript",
    "parse_hotkey",
    "speakable",
    "split_compound_commands",
]

