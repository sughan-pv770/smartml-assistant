"""
smartml.mial.controller
=======================
MialController — the conversational state-machine brain for Mial.

Handles:
1. Global hotkey activation (``Ctrl+Alt+M``) and mid-speech barge-in.
2. Greeting the user in natural English.
3. Listening for English commands via offline Whisper STT (or injected callable for tests).
4. 2-strike retry logic with explicit ``retry_count = 0`` reset on every state exit.
5. Dataset readiness check: if the user asks to analyze/train without a dataset loaded,
   prompts for a file name or launches the subprocess native file picker, loads the file,
   and automatically runs the pending task.
6. Multi-file disambiguation when a spoken partial name matches several files.
7. Child-process execution for heavy jobs with ``Esc`` cancellation.
8. 5-second follow-up window for multi-turn conversation.
"""

from __future__ import annotations

import enum
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ..agent.audio import MicrophoneUnavailableError, STTFailureError, beep
from ..agent.hotkey import HotkeyListener
from ..agent.nlp import speakable
from ..agent.voice_agent import VoiceAgent
from .config import MialConfig
from .file_dialog import open_file_dialog_subprocess
from .runner import CancellableRunner
from .speaker import MialSpeaker

_log = logging.getLogger("smartml.mial")


class MialState(str, enum.Enum):
    IDLE = "IDLE"
    GREETING = "GREETING"
    LISTENING_TASK = "LISTENING_TASK"
    AWAITING_DATASET = "AWAITING_DATASET"
    DISAMBIGUATING_FILE = "DISAMBIGUATING_FILE"
    EXECUTING = "EXECUTING"
    SPEAKING_RESULT = "SPEAKING_RESULT"
    FOLLOW_UP = "FOLLOW_UP"
    # Explicit error states
    MICROPHONE_UNAVAILABLE = "MicrophoneUnavailable"
    STT_FAILURE = "STTFailure"
    DIALOG_DISMISSED = "DialogDismissed"


class MialController:
    """Conversational state machine orchestrating the Mial voice AI agent."""

    def __init__(
        self,
        config: Optional[MialConfig] = None,
        agent: Optional[VoiceAgent] = None,
        speaker: Optional[MialSpeaker] = None,
        listen_fn: Optional[Callable[[Optional[float]], str]] = None,
        dialog_fn: Optional[Callable[[], Optional[str]]] = None,
        register_global_esc: bool = True,
    ) -> None:
        self.config = config or MialConfig()
        self.speaker = speaker or MialSpeaker(
            enabled=self.config.speak_replies,
            voice_name=self.config.voice_name,
            prefer_neural_en_in=self.config.online_tts,
        )
        self.agent = agent or VoiceAgent(
            output_dir=self.config.output_dir,
            search_dirs=self.config.allowed_dirs,
            stt=self.config.effective_stt_engine,
            whisper_model=self.config.whisper_model,
            speak_replies=self.config.speak_replies,
            hotkey=self.config.hotkey,
            allow_online_stt=self.config.online_stt,
            speaker=self.speaker,
            audit_log=self.config.audit_log,
        )
        self.runner = CancellableRunner(
            self.agent,
            cancel_key=self.config.cancel_key,
            register_global_esc=register_global_esc,
        )

        self._listen_fn = listen_fn or self._default_listen
        self._dialog_fn = dialog_fn or (
            lambda: open_file_dialog_subprocess(
                initial_dir=self.config.allowed_dirs[0] if self.config.allowed_dirs else Path.cwd()
            )
        )

        self.state: MialState = MialState.IDLE
        self.retry_count: int = 0
        self.pending_task: Optional[str] = None
        self.state_history: list[MialState] = [MialState.IDLE]
        self._greeted_once: bool = False

        self._wake_event = threading.Event()
        self._barge_in_event = threading.Event()
        self._stop_daemon = threading.Event()

    def _transition(self, new_state: MialState) -> None:
        """Transition to *new_state*; reset retry_count only when entering IDLE."""
        if new_state == MialState.IDLE:
            self.retry_count = 0
        self.state = new_state
        self.state_history.append(new_state)
        _log.debug("Mial state -> %s (retry_count=%d)", new_state.value, self.retry_count)

    def _default_listen(self, timeout_s: Optional[float] = None) -> str:
        """Record utterance from microphone and transcribe with configured STT."""
        beep(880, 100)
        print("\n[Mial listening...]")
        wav_bytes = self.agent.recorder.record_utterance(timeout_s=timeout_s)
        if not wav_bytes:
            return ""
        text = self.agent.stt.transcribe(wav_bytes)
        if text:
            print(f"You > {text}")
        return text

    def _say(self, text: str, wait: bool = True) -> None:
        """Print full reply to console (UTF-8 safe on Windows cp1252) and speak concise summary."""
        if not text:
            return
        try:
            print(f"Mial > {text}")
        except UnicodeEncodeError:
            safe_text = text.encode("ascii", errors="replace").decode("ascii")
            print(f"Mial > {safe_text}")
        if self.config.speak_replies:
            spoken = speakable(text)
            self.speaker.speak(spoken, wait=wait)

    # ------------------------------------------------------------------ #
    # Hotkey & Barge-In Trigger
    # ------------------------------------------------------------------ #

    def on_hotkey_pressed(self) -> None:
        """Callback fired when the user presses ``Ctrl+Alt+M``."""
        if self.state in (MialState.GREETING, MialState.SPEAKING_RESULT):
            # Barge-in: immediately interrupt ongoing speech and listen
            self.speaker.stop()
            self._barge_in_event.set()
            self._wake_event.set()
            return

        if self.state == MialState.IDLE:
            self._wake_event.set()

    # ------------------------------------------------------------------ #
    # Interactive Session Execution (One Wake Cycle)
    # ------------------------------------------------------------------ #

    def trigger_session(self) -> MialState:
        """Run one complete wake-up session starting from GREETING until returning to IDLE."""
        self._barge_in_event.clear()
        self.pending_task = None
        self._transition(MialState.GREETING)
        self._say(self.config.greeting, wait=True)

        self._transition(MialState.LISTENING_TASK)

        while self.state != MialState.IDLE:
            if self.state == MialState.LISTENING_TASK:
                self._step_listening_task()
            elif self.state == MialState.DISAMBIGUATING_FILE:
                self._step_disambiguating_file()
            elif self.state == MialState.AWAITING_DATASET:
                self._step_awaiting_dataset()
            elif self.state == MialState.FOLLOW_UP:
                self._step_follow_up()
            elif self.state in (
                MialState.MICROPHONE_UNAVAILABLE,
                MialState.STT_FAILURE,
                MialState.DIALOG_DISMISSED,
            ):
                self._transition(MialState.IDLE)
            else:
                break

        return self.state

    def _listen_safely(self, timeout_s: Optional[float] = None) -> Optional[str]:
        """Invoke listener and route hardware/STT errors to dedicated error states."""
        try:
            return self._listen_fn(timeout_s)
        except MicrophoneUnavailableError as exc:
            self._transition(MialState.MICROPHONE_UNAVAILABLE)
            self._say(
                f"Microphone unavailable: {exc} You can run 'mial --text' for keyboard mode."
            )
            return None
        except STTFailureError as exc:
            self._transition(MialState.STT_FAILURE)
            self._say(
                f"Offline speech recognition unavailable: {exc} "
                "Install faster-whisper or set online_stt=true in mial.toml."
            )
            return None

    def _step_listening_task(self) -> None:
        utterance = self._listen_safely()
        if utterance is None:
            return  # Transitioned to error state

        if not utterance.strip():
            self.retry_count += 1
            if self.retry_count <= self.config.max_retries:
                self._say("Sorry, I didn't catch that. What would you like to do?")
                return
            self._say("Going back to sleep. Press Control Alt M whenever you need me.")
            self._transition(MialState.IDLE)
            return

        self._dispatch_utterance(utterance.strip())

    def _dispatch_utterance(self, utterance: str) -> None:
        low = utterance.lower().strip()
        if re.search(r"^(stop|quit|exit|goodbye|bye|go to sleep|nothing|cancel)$", low):
            self._say("Stopped. Going back to sleep.")
            self._transition(MialState.IDLE)
            return

        # If user asks to pick/browse a file directly
        if re.search(r"\b(pick a file|browse|open file dialog|choose a file)\b", low):
            self._transition(MialState.AWAITING_DATASET)
            self._open_picker_and_continue()
            return

        self._transition(MialState.EXECUTING)
        reply, was_cancelled = self.runner.execute(utterance)

        if was_cancelled:
            self._say("Training cancelled.")
            self._transition(MialState.IDLE)
            return

        # Did the command trigger multi-file disambiguation?
        if self.agent.pending_candidates:
            self._transition(MialState.DISAMBIGUATING_FILE)
            self._say(reply)
            return

        # Did the user ask for an ML/analysis task before loading a dataset?
        if "please load a dataset first" in reply.lower():
            self.pending_task = utterance
            self._transition(MialState.AWAITING_DATASET)
            self._say(
                "No dataset is loaded yet. Please say the dataset file name, "
                "or say 'pick a file' to open the file browser."
            )
            return

        if self.agent.should_stop:
            self.agent.should_stop = False
            self._say(reply)
            self._transition(MialState.IDLE)
            return

        self.retry_count = 0
        self._speak_result_and_enter_followup(reply)

    def _step_disambiguating_file(self) -> None:
        utterance = self._listen_safely()
        if utterance is None:
            return
        if not utterance.strip():
            self.retry_count += 1
            if self.retry_count <= self.config.max_retries:
                self._say("Please say the number or name of the file you want to load.")
                return
            self.agent.pending_candidates.clear()
            self._say("No file selected. Going back to sleep.")
            self._transition(MialState.IDLE)
            return

        self._transition(MialState.EXECUTING)
        reply = self.agent.handle_command(utterance.strip())
        if self.agent.has_dataset and self.pending_task:
            task = self.pending_task
            self.pending_task = None
            task_reply, cancelled = self.runner.execute(task)
            if cancelled:
                self._say("Training cancelled.")
                self._transition(MialState.IDLE)
                return
            reply = f"{reply}\n\n{task_reply}"

        self.retry_count = 0
        self._speak_result_and_enter_followup(reply)

    def _step_awaiting_dataset(self) -> None:
        utterance = self._listen_safely()
        if utterance is None:
            return

        if not utterance.strip():
            # If user stayed silent when asked for a dataset name, open native file picker
            self._open_picker_and_continue()
            return

        low = utterance.lower().strip()
        if low in ("cancel", "stop", "never mind", "quit"):
            self.pending_task = None
            self._say("Cancelled. Going back to sleep.")
            self._transition(MialState.IDLE)
            return

        if re.search(r"\b(pick|browse|dialog|choose|window)\b", low):
            self._open_picker_and_continue()
            return

        # User spoke a file name (e.g. "student survey" or "load student survey")
        load_cmd = utterance if re.search(r"\b(load|open)\b", low) else f"load {utterance}"
        self._transition(MialState.EXECUTING)
        load_reply = self.agent.handle_command(load_cmd)

        if self.agent.pending_candidates:
            self._transition(MialState.DISAMBIGUATING_FILE)
            self._say(load_reply)
            return

        if not self.agent.has_dataset:
            self.retry_count += 1
            if self.retry_count <= self.config.max_retries:
                self.state = MialState.AWAITING_DATASET
                self._say(f"{load_reply} Try saying the file name again, or say 'pick a file'.")
                return
            self._transition(MialState.DIALOG_DISMISSED)
            self._say("No dataset selected. Going back to sleep.")
            return

        # Dataset is now loaded! If there was a pending task, run it immediately
        if self.pending_task:
            task = self.pending_task
            self.pending_task = None
            task_reply, cancelled = self.runner.execute(task)
            if cancelled:
                self._say("Training cancelled.")
                self._transition(MialState.IDLE)
                return
            load_reply = f"{load_reply}\n\n{task_reply}"

        self.retry_count = 0
        self._speak_result_and_enter_followup(load_reply)

    def _open_picker_and_continue(self) -> None:
        chosen = self._dialog_fn()
        if not chosen:
            self._transition(MialState.DIALOG_DISMISSED)
            self._say("No dataset file was selected. Going back to sleep.")
            return

        self._transition(MialState.EXECUTING)
        load_reply = self.agent.load_file(chosen, allow_outside=True)
        if not self.agent.has_dataset:
            self._say(load_reply)
            self._transition(MialState.IDLE)
            return

        if self.pending_task:
            task = self.pending_task
            self.pending_task = None
            task_reply, cancelled = self.runner.execute(task)
            if cancelled:
                self._say("Training cancelled.")
                self._transition(MialState.IDLE)
                return
            load_reply = f"{load_reply}\n\n{task_reply}"

        self.retry_count = 0
        self._speak_result_and_enter_followup(load_reply)

    def _speak_result_and_enter_followup(self, reply: str) -> None:
        self._barge_in_event.clear()
        self._transition(MialState.SPEAKING_RESULT)
        self._say(reply, wait=True)

        if self._barge_in_event.is_set():
            self._barge_in_event.clear()
            self._transition(MialState.LISTENING_TASK)
            return

        if self.config.follow_up_window_s <= 0:
            self._transition(MialState.IDLE)
            return

        self._transition(MialState.FOLLOW_UP)

    def _step_follow_up(self) -> None:
        utterance = self._listen_safely(timeout_s=self.config.follow_up_window_s)
        if utterance is None:
            return
        if not utterance.strip():
            # 5s silence during follow-up returns quietly to IDLE
            self._transition(MialState.IDLE)
            return

        low = utterance.lower().strip()
        if re.search(r"^(thank you|thanks|that's all|nothing else|stop|done|bye)$", low):
            self._say("You're welcome! Press Control Alt M whenever you need me.")
            self._transition(MialState.IDLE)
            return

        self._dispatch_utterance(utterance.strip())

    # ------------------------------------------------------------------ #
    # Background Hotkey Daemon Loop
    # ------------------------------------------------------------------ #

    def run_forever(self) -> None:
        """Start the global ``Ctrl+Alt+M`` hotkey listener and wait for wake events."""
        listener = HotkeyListener(self.config.hotkey, self.on_hotkey_pressed)
        listener.start()
        print(
            f"Mial v2.1.0 active! Press {self.config.hotkey} from anywhere to speak "
            f"(or Ctrl+C in terminal to quit)."
        )
        try:
            while not self._stop_daemon.is_set():
                if self._wake_event.wait(timeout=0.2):
                    self._wake_event.clear()
                    self.trigger_session()
        except KeyboardInterrupt:
            print("\nShutting down Mial...")
        finally:
            listener.stop()
            self.speaker.shutdown()

