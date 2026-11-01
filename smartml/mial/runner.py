"""
smartml.mial.runner
===================
Cancellable execution runner for Mial.

Fast commands (analyze, clean, feature importance, explain, cluster, report, wrangling)
run directly in-process in milliseconds.
Heavy commands (train, tune, full pipeline, forecast, semi/self-supervised, RL)
run in a spawned child process (``subprocess.Popen``) that starts in ~5 ms on Windows,
can be terminated at any instant with ``Esc`` (< 20 ms reaction time), and returns
the updated ``Session`` via fast local ``joblib`` serialisation (~1.6 MB in < 10 ms).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import joblib

from ..agent.hotkey import HotkeyListener
from ..agent.voice_agent import VoiceAgent
from ..assistant import Assistant

_log = logging.getLogger("smartml.mial")

_HEAVY_PATTERNS = re.compile(
    r"\b("
    r"train\b|build\b.*\bmodel\b|fit\b.*\bmodel\b|best\s+model\b|compare\b.*\bmodels?\b|"
    r"do\s+everything\b|full\s+pipeline\b|run\s+all\b|end\s+to\s+end\b|"
    r"tune\b|forecast\b|semi[- ]supervised\b|self[- ]supervised\b|"
    r"reinforcement\s+learning\b|bandit\b"
    r")",
    re.IGNORECASE,
)


def is_heavy_command(text: str) -> bool:
    """Return True if *text* triggers model training, tuning, full pipeline, or forecasting."""
    return bool(_HEAVY_PATTERNS.search(text or ""))


def _kill_popen_tree(proc: subprocess.Popen) -> None:
    """Immediately kill *proc* and any child processes using psutil."""
    if proc.poll() is not None:
        return
    pid = proc.pid
    if pid is not None:
        try:
            import psutil  # type: ignore

            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
            psutil.wait_procs(children + [parent], timeout=0.2)
            return
        except Exception:
            pass
    try:
        proc.kill()
        proc.wait(timeout=0.2)
    except Exception:
        pass


class CancellableRunner:
    """Executes VoiceAgent commands, isolating heavy jobs in a cancellable child process."""

    def __init__(
        self,
        agent: VoiceAgent,
        cancel_key: str = "<esc>",
        register_global_esc: bool = True,
    ) -> None:
        self.agent = agent
        self.cancel_key = cancel_key
        self.register_global_esc = register_global_esc
        self._cancel_event = threading.Event()
        self._active_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def cancel(self) -> bool:
        """Signal cancellation and kill any active child training process immediately."""
        self._cancel_event.set()
        with self._lock:
            proc = self._active_proc
            if proc is not None and proc.poll() is None:
                _kill_popen_tree(proc)
                return True
        return False

    def execute(self, text: str, **context: Any) -> tuple[str, bool]:
        """Execute *text* and return ``(reply, was_cancelled)``."""
        self._cancel_event.clear()

        # If no dataset is loaded yet, or the command is fast, run in-process
        if not self.agent.has_dataset or not is_heavy_command(text):
            reply = self.agent.handle_command(text, **context)
            return reply, False

        from ..agent.voice_agent import _parse_voice_command

        commands = _parse_voice_command(text)
        if not commands:
            return "I didn't catch that. Please say a command.", False

        replies: list[str] = []
        for cmd in commands:
            if self._cancel_event.is_set():
                replies.append("Training cancelled.")
                return "\n\n".join(replies), True

            if self.agent.has_dataset and is_heavy_command(cmd):
                single_reply, cancelled = self._run_heavy_in_subprocess(cmd, **context)
                replies.append(single_reply)
                if cancelled:
                    full = "\n\n".join(replies)
                    self.agent.last_reply = full
                    self.agent._log_turn(text, full)
                    return full, True
            else:
                single_reply = self.agent._execute_single(cmd, **context)
                replies.append(single_reply)

            if self.agent.should_stop or self.agent.pending_candidates:
                break
            if any(
                marker in single_reply.lower()
                for marker in (
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
        self.agent.last_reply = full_reply
        self.agent._log_turn(text, full_reply)
        return full_reply, False

    def _run_heavy_in_subprocess(self, cmd: str, **context: Any) -> tuple[str, bool]:
        assert self.agent.assistant is not None
        context.setdefault("strict_security", True)

        tmp_dir = Path(tempfile.mkdtemp(prefix="mial_job_"))
        in_path = tmp_dir / "in_job.joblib"
        out_path = tmp_dir / "out_job.joblib"

        joblib.dump((self.agent.assistant.session, cmd, context), in_path)

        esc_listener: Optional[HotkeyListener] = None
        if self.register_global_esc:
            try:
                esc_listener = HotkeyListener(self.cancel_key, self.cancel)
                esc_listener.start()
            except Exception as exc:
                _log.debug("Could not register Esc hotkey listener: %s", exc)
                esc_listener = None

        proc: Optional[subprocess.Popen] = None
        try:
            if self._cancel_event.is_set():
                return "Training cancelled.", True

            with self._lock:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "smartml.mial.runner", str(in_path), str(out_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._active_proc = proc

            while proc.poll() is None:
                if self._cancel_event.is_set():
                    _kill_popen_tree(proc)
                    return "Training cancelled.", True
                time.sleep(0.02)

            if self._cancel_event.is_set():
                _kill_popen_tree(proc)
                return "Training cancelled.", True

            if out_path.is_file():
                status, reply, new_session = joblib.load(out_path)
                if status == "OK" and new_session is not None:
                    self.agent.assistant.session = new_session
                return str(reply), False

            if proc.returncode != 0:
                return "Training cancelled.", True
            return "Command finished with no output.", False
        finally:
            if esc_listener is not None:
                esc_listener.stop()
            with self._lock:
                if self._active_proc is proc:
                    self._active_proc = None
            for p in (in_path, out_path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                tmp_dir.rmdir()
            except OSError:
                pass


def _worker_entry() -> int:
    if len(sys.argv) < 3:
        return 1
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
    try:
        session, cmd, context = joblib.load(in_path)
        assistant = Assistant.__new__(Assistant)
        assistant.session = session
        with joblib.parallel_backend("threading"):
            reply = assistant.ask(cmd, **context)
        joblib.dump(("OK", reply, assistant.session), out_path)
        return 0
    except Exception as exc:
        joblib.dump(("ERR", f"Command failed in worker: {exc}", None), out_path)
        return 0


if __name__ == "__main__":
    sys.exit(_worker_entry())
