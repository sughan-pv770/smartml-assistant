"""
smartml.agent.voice_agent
=========================
VoiceAgent (and SmartMLAgent alias): connects spoken/typed natural-language
commands to SmartML's Assistant, with safe dataset discovery, partial-match
disambiguation, compound command execution, and JSONL audit logging.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from ..assistant import Assistant
from ..core import loaders, textutil
from .audio import AudioRecorder, SpeechToText, TextToSpeech, beep
from .hotkey import HotkeyListener
from .nlp import (
    ALLOWED_DATA_EXTENSIONS,
    find_dataset_candidates,
    find_dataset_file,
    is_within_allowed_dirs,
    normalize_transcript,
    speakable,
    split_compound_commands,
)

_log = logging.getLogger("smartml.agent")


def _parse_voice_command(text: str) -> list[str]:
    """Normalise and split a spoken command string into ordered atomic commands."""
    return split_compound_commands(text)


class VoiceAgent:
    """Voice and text agent layered over :class:`~smartml.assistant.Assistant`.

    Args:
        df: Optional pre-loaded DataFrame.
        target: Optional target column name.
        output_dir: Output directory for reports, plots, models, and logs.
        search_dirs: Allowed directories for dataset discovery.
        stt: Speech-to-text backend (``"whisper"``, ``"google"``, ``"auto"``).
        whisper_model: Whisper model size (``"base"``, ``"small"``, ``"tiny"``).
        speak_replies: Whether to speak replies aloud.
        hotkey: Global hotkey string (default ``"<ctrl>+<alt>+m"``).
        allow_online_stt: Whether ``stt="auto"`` may fall back to online Google STT.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        target: Optional[str] = None,
        output_dir: str = "smartml_output",
        search_dirs: Optional[Sequence[str | Path]] = None,
        stt: str = "whisper",
        whisper_model: str = "base",
        speak_replies: bool = True,
        hotkey: str = "<ctrl>+<alt>+m",
        allow_online_stt: bool = False,
        speaker: Optional[Any] = None,
        audit_log: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.audit_log = audit_log
        os.makedirs(self.output_dir, exist_ok=True)

        if search_dirs is None:
            home = Path.home()
            self.search_dirs: list[Path] = [
                Path.cwd(),
                home / "Downloads",
                home / "Desktop",
                home / "Documents",
            ]
        else:
            self.search_dirs = [Path(d) for d in search_dirs]

        self.assistant: Optional[Assistant] = (
            Assistant(df, target=target, output_dir=output_dir)
            if df is not None
            else None
        )
        self.current_file: Optional[str] = None
        self.pending_candidates: list[str] = []
        self.last_reply: str = ""
        self.should_stop: bool = False

        self.hotkey = hotkey
        self.speak_replies = speak_replies
        self.recorder = AudioRecorder()
        self.stt = SpeechToText(
            engine=stt,
            model_size=whisper_model,
            allow_online_fallback=allow_online_stt,
        )
        self.speaker = speaker if speaker is not None else TextToSpeech(enabled=speak_replies)

    @property
    def has_dataset(self) -> bool:
        """True if a dataset is currently loaded into the Assistant."""
        return self.assistant is not None

    def load_dataframe(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> str:
        """Load a DataFrame into a fresh Assistant session."""
        self.assistant = Assistant(df, target=target, output_dir=self.output_dir)
        self.current_file = source_path
        self.pending_candidates.clear()
        n_rows, n_cols = df.shape
        fname = os.path.basename(source_path) if source_path else "dataset"
        cols_preview = ", ".join(str(c) for c in list(df.columns[:6]))
        more = "..." if len(df.columns) > 6 else ""
        msg = (
            f"Loaded '{fname}' with {n_rows} rows and {n_cols} columns "
            f"({cols_preview}{more})."
        )
        if target:
            msg += f" Target set to '{target}'."
        self.last_reply = msg
        return msg

    def load_file(self, path: str, allow_outside: bool = False) -> str:
        """Load a dataset file after verifying extension and allowed folder boundaries."""
        p = Path(path)
        if p.suffix.lower() not in ALLOWED_DATA_EXTENSIONS:
            return (
                f"Unsupported file extension '{p.suffix}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_DATA_EXTENSIONS))}."
            )
        if not allow_outside and not is_within_allowed_dirs(p, self.search_dirs):
            return (
                f"Access denied: '{path}' is outside the configured allowed dataset folders."
            )
        try:
            df = loaders.load(str(p.resolve()))
        except Exception as exc:
            return f"Failed to load '{p.name}': {exc}"
        return self.load_dataframe(df, source_path=str(p.resolve()))

    def handle_command(self, text: str, **context: Any) -> str:
        """Process a spoken or typed natural-language utterance (single or compound)."""
        norm_raw = normalize_transcript(text)
        if not norm_raw:
            return "I didn't catch that. Please say a command."

        # If we are waiting for the user to disambiguate between multiple files
        if self.pending_candidates:
            choice_reply = self._try_resolve_disambiguation(norm_raw)
            if choice_reply is not None:
                self._log_turn(text, choice_reply)
                self.last_reply = choice_reply
                return choice_reply

        commands = _parse_voice_command(text)
        if not commands:
            return "I didn't catch that. Please say a command."

        replies: list[str] = []
        for cmd in commands:
            single_reply = self._execute_single(cmd, **context)
            replies.append(single_reply)
            if self.should_stop or self.pending_candidates:
                break
            # Stop compound chain if a step failed or couldn't run
            if any(
                err_marker in single_reply.lower()
                for err_marker in (
                    "please load a dataset first",
                    "couldn't find",
                    "could not find",
                    "access denied",
                    "failed to load",
                    "i need a target column first",
                )
            ):
                break

        full_reply = "\n\n".join(replies)
        self.last_reply = full_reply
        self._log_turn(text, full_reply)
        return full_reply

    def _try_resolve_disambiguation(self, norm: str) -> Optional[str]:
        """Resolve a user response when multiple dataset files matched."""
        low = norm.lower().strip()
        if low in ("cancel", "never mind", "none", "stop"):
            self.pending_candidates.clear()
            return "Cancelled file selection."

        # Check numeric index ("1", "first", "number 2")
        ordinal_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        idx: Optional[int] = None
        for word, num in ordinal_map.items():
            if re.search(rf"\b{word}\b", low):
                idx = num
                break
        if idx is None:
            m_num = re.search(r"\b(\d+)\b", low)
            if m_num:
                idx = int(m_num.group(1))

        if idx is not None and 1 <= idx <= len(self.pending_candidates):
            chosen = self.pending_candidates[idx - 1]
            self.pending_candidates.clear()
            return self.load_file(chosen)

        # Check filename substring match against pending candidates
        for cand in self.pending_candidates:
            stem = Path(cand).stem.lower()
            if low in stem or any(tok in stem for tok in re.findall(r"[a-z0-9]+", low) if len(tok) >= 3):
                self.pending_candidates.clear()
                return self.load_file(cand)

        return None

    def _execute_single(self, cmd: str, **context: Any) -> str:
        low = cmd.lower().strip()

        # 1. Stop / exit
        if re.search(r"^(stop|quit|exit|goodbye|bye|go to sleep)$", low):
            self.should_stop = True
            return "Going back to sleep. Press the hotkey whenever you need me."

        # 2. Repeat last reply
        if re.search(r"\b(say that again|repeat|what did you say)\b", low):
            return self.last_reply or "I haven't said anything yet in this session."

        # 3. Status check
        if re.search(r"^(status|what is loaded|session status|current status)$", low):
            if self.assistant is None:
                return "No dataset is loaded yet. Say 'load <filename>' to begin."
            s = self.assistant.session
            rows, cols = s.raw_df.shape
            target_str = f"target='{s.target}'" if s.target else "no target set"
            trained_str = (
                f"best model='{s.training_result.best_name}'"
                if s.training_result
                else "no model trained yet"
            )
            return f"Session status: {rows} rows × {cols} columns, {target_str}, {trained_str}."

        # 4. Load dataset file (avoiding "load model")
        if re.search(r"\b(load|open|read|use)\b", low) and not re.search(r"\bmodel\b", low):
            # Check if an explicit path outside allowed dirs was attempted
            path_m = re.search(
                r"([A-Za-z]:[\\/][\w\-./\\ ]+\.(?:csv|tsv|xlsx?|json|parquet|feather)|"
                r"(?:\.\.?[\\/]|/)[\w\-./\\]+\.(?:csv|tsv|xlsx?|json|parquet|feather))",
                cmd,
                re.IGNORECASE,
            )
            if path_m:
                raw_p = Path(path_m.group(1).strip())
                if not is_within_allowed_dirs(raw_p, self.search_dirs):
                    return (
                        f"Access denied: '{raw_p}' is outside the allowed dataset folders."
                    )

            candidates = find_dataset_candidates(cmd, search_dirs=self.search_dirs)
            if not candidates:
                return (
                    f"I couldn't find a dataset matching '{cmd}' in your allowed folders."
                )
            if len(candidates) == 1:
                return self.load_file(candidates[0])

            # Multiple files matched -> ask user to disambiguate
            self.pending_candidates = candidates[:5]
            options = ", ".join(
                f"{i + 1}) {Path(p).name}" for i, p in enumerate(self.pending_candidates)
            )
            return f"I found {len(self.pending_candidates)} matching files: {options}. Which one would you like to load?"

        # 5. All other commands require a loaded dataset
        if self.assistant is None:
            return (
                "Please load a dataset first — for example, say "
                "'load student survey' or pick a file."
            )

        # Enforce strict model security when called from voice agent
        context.setdefault("strict_security", True)
        return self.assistant.ask(cmd, **context)

    def _log_turn(self, user_text: str, reply_text: str) -> None:
        """Append command and reply to ``<output_dir>/agent_log.jsonl`` when audit_log is enabled."""
        if not self.audit_log:
            return
        try:
            log_path = Path(self.output_dir) / "agent_log.jsonl"
            entry = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "command": user_text,
                "reply": reply_text,
                "spoken": speakable(reply_text),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            _log.debug("Could not write agent_log.jsonl: %s", exc)


# Alias for backward compatibility
SmartMLAgent = VoiceAgent


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for ``smartml-voice`` / ``python -m smartml.agent``."""
    parser = argparse.ArgumentParser(
        prog="smartml-voice",
        description="SmartML Voice & Text Agent",
    )
    parser.add_argument("--data", type=str, default=None, help="Initial dataset file path to load.")
    parser.add_argument("--target", type=str, default=None, help="Initial target column name.")
    parser.add_argument("--hotkey", type=str, default="<ctrl>+<alt>+m", help="Global hotkey combo.")
    parser.add_argument("--stt", type=str, default="whisper", choices=["whisper", "google", "auto"])
    parser.add_argument("--text", action="store_true", help="Run in interactive typed terminal mode.")
    parser.add_argument("--no-voice-output", action="store_true", help="Disable spoken TTS output.")
    args = parser.parse_args(argv)

    agent = VoiceAgent(
        target=args.target,
        stt=args.stt,
        speak_replies=not args.no_voice_output,
        hotkey=args.hotkey,
    )
    if args.data:
        print(agent.load_file(args.data, allow_outside=True))

    if args.text:
        print("SmartML Voice Agent (typed mode). Type 'stop' to quit.")
        while not agent.should_stop:
            try:
                line = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            reply = agent.handle_command(line)
            print(f"\nAgent > {reply}\n")
            if agent.speak_replies:
                agent.speaker.speak(speakable(reply), wait=False)
        return 0

    # Delegate full hotkey + voice state machine to MialController
    from ..mial.controller import MialController
    from ..mial.config import MialConfig

    cfg = MialConfig(
        hotkey=args.hotkey,
        stt_engine=args.stt,
        speak_replies=not args.no_voice_output,
    )
    controller = MialController(config=cfg, agent=agent)
    controller.run_forever()
    return 0

