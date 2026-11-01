"""
smartml.mial.cli
================
Command-line interface for Mial (``mial`` console script).
Pointing ``[project.scripts]`` at ``smartml.mial.cli:main`` ensures importing
``smartml.mial.__main__`` never executes the CLI at import time.
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from typing import Optional, Sequence

from ..agent.nlp import speakable
from .config import MialConfig
from .controller import MialController
from .speaker import MialSpeaker


def run_latency_benchmark() -> int:
    """Measure real SAPI5 time-to-first-audio and mid-utterance barge-in stop latency."""
    print("=== Mial Latency & Cancellation Benchmark ===")
    speaker = MialSpeaker(enabled=True)
    print(f"Active TTS backend: {'Windows SAPI5 COM thread' if speaker.using_sapi else 'Subprocess TTS fallback'}")

    stream_ms, phoneme_ms = speaker.measure_first_audio_ms("Hi, I'm Mial. What would you like to do?")
    time.sleep(0.05)
    stop_ms = speaker.stop()
    speaker.shutdown()

    print(f"  Hotkey -> SAPI stream active (RunningState==2): {stream_ms:.2f} ms (target < 300 ms)")
    print(f"  Hotkey -> first audible phoneme (PhonemeId>0) : {phoneme_ms:.2f} ms (target < 300 ms)")
    print(f"  Barge-in SVSFPurgeBeforeSpeak stop latency    : {stop_ms:.2f} ms (target < 50 ms)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the ``mial`` command."""
    multiprocessing.freeze_support()
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="mial",
        description=(
            "Mial - Hotkey-triggered voice assistant for SmartML. "
            "Audio stays local in RAM via offline faster-whisper (downloads ~150 MB once on first voice use). "
            "When NVIDIA_API_KEY is set, spoken transcripts and aggregate tool summaries (never raw cell values "
            "or audio) are sent to NVIDIA NIM; pass --offline to run 100% locally."
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Path to mial.toml config file.")
    parser.add_argument("--data", type=str, default=None, help="Dataset file path to preload.")
    parser.add_argument("--target", type=str, default=None, help="Target column name.")
    parser.add_argument("--cmd", type=str, default=None, help="Auto-execute an initial natural-language command on launch.")
    parser.add_argument("--hotkey", type=str, default=None, help="Override global wake hotkey (default: <ctrl>+<alt>+m).")
    parser.add_argument("--gemini-key", type=str, default=None, help="Optional Gemini API key.")
    parser.add_argument("--offline", action="store_true", help="Disable all cloud LLM calls even when NVIDIA_API_KEY is set.")
    parser.add_argument("--headless", action="store_true", help="Run without the desktop popup UI.")
    parser.add_argument("--text", action="store_true", help="Run in interactive typed English terminal mode.")
    parser.add_argument("--wake", action="store_true", help="Trigger a voice greeting immediately on launch.")
    parser.add_argument("--online-stt", action="store_true", help="Explicitly opt in to online Google STT fallback.")
    parser.add_argument("--online-tts", action="store_true", help="Explicitly opt in to online neural TTS.")
    parser.add_argument("--audit-log", action="store_true", help="Explicitly opt in to writing transcript audit logs.")
    parser.add_argument("--no-voice-output", action="store_true", help="Mute spoken voice replies.")
    parser.add_argument("--benchmark", action="store_true", help="Run TTS greeting & barge-in latency benchmark.")
    args = parser.parse_args(argv)

    if args.benchmark:
        return run_latency_benchmark()

    if args.gemini_key:
        import os
        os.environ["GEMINI_API_KEY"] = args.gemini_key.strip()

    cfg = MialConfig.load(args.config)
    if args.hotkey:
        cfg.hotkey = args.hotkey
    if args.online_stt:
        cfg.online_stt = True
    if args.online_tts:
        cfg.online_tts = True
    if args.offline:
        cfg.offline_only = True
    if args.audit_log:
        cfg.audit_log = True
    if args.no_voice_output:
        cfg.speak_replies = False

    controller = MialController(config=cfg)

    # Default mode: launch the desktop popup window
    if not args.text and not args.headless:
        from .popup_ui import MialPopupApp

        app = MialPopupApp(
            controller=controller,
            initial_dataset=args.data,
            initial_command=args.cmd,
        )
        return app.run()

    if args.data:
        msg = controller.agent.load_file(args.data, allow_outside=True)
        print(f"Mial > {msg}")
        if args.target and controller.agent.assistant is not None:
            t_msg = controller.agent.assistant.set_target(args.target)
            print(f"Mial > {t_msg}")

    if args.text:
        print(f"Mial > {cfg.greeting}")
        if cfg.speak_replies:
            controller.speaker.speak(cfg.greeting, wait=False)
        print("(Typed English mode - type a command like 'load student file' or 'stop' to quit.)")
        while not controller.agent.should_stop:
            try:
                user_in = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_in:
                continue
            reply, cancelled = controller.runner.execute(user_in)
            print(f"\nMial > {reply}")
            if cfg.speak_replies and not cancelled:
                controller.speaker.speak(speakable(reply), wait=False)
        controller.speaker.shutdown()
        return 0

    if args.wake:
        controller.trigger_session()

    controller.run_forever()
    return 0

