"""
smartml.agent.audio
===================
Offline-first audio recording (Energy-based VAD), speech-to-text (faster-whisper
by default, optional Google fallback), and stdin-safe text-to-speech.
All heavy audio dependencies (sounddevice, faster_whisper, speech_recognition)
are imported lazily so ``import smartml`` works without them.
"""

from __future__ import annotations

import io
import logging
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger("smartml.agent")


class MicrophoneUnavailableError(RuntimeError):
    """Raised when no audio input device or sounddevice library is available."""


class STTFailureError(RuntimeError):
    """Raised when speech-to-text transcription fails or engine is unavailable."""


class EnergyVAD:
    """Pure-NumPy RMS energy voice activity detector (no C++ compiler required).

    Args:
        energy_threshold: Minimum normalized RMS energy (0.0–1.0) treated as speech.
        silence_duration_s: Seconds of trailing silence after speech before stopping.
        max_duration_s: Hard cap on total recording length in seconds.
        no_speech_timeout_s: Seconds to wait for speech to begin before giving up.
    """

    def __init__(
        self,
        energy_threshold: float = 0.015,
        silence_duration_s: float = 0.6,
        max_duration_s: float = 15.0,
        no_speech_timeout_s: float = 4.0,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.silence_duration_s = silence_duration_s
        self.max_duration_s = max_duration_s
        self.no_speech_timeout_s = no_speech_timeout_s

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        """Compute normalized RMS energy of a float32/int16 audio frame."""
        if frame.size == 0:
            return 0.0
        arr = frame.astype(np.float32)
        if np.max(np.abs(arr)) > 1.5:
            arr = arr / 32768.0
        return float(math.sqrt(float(np.mean(arr * arr))))


class AudioRecorder:
    """Records speech from the default microphone using EnergyVAD."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad: Optional[EnergyVAD] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.vad = vad or EnergyVAD()

    def record_utterance(self, timeout_s: Optional[float] = None) -> bytes:
        """Record a single spoken utterance and return 16-bit mono PCM WAV bytes.

        Raises:
            MicrophoneUnavailableError: If sounddevice or a microphone is missing.
        """
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise MicrophoneUnavailableError(
                "Microphone recording requires 'sounddevice'. "
                "Install with: pip install sounddevice (or run with --text)."
            ) from exc

        chunk_duration = 0.1  # 100 ms chunks
        chunk_samples = int(self.sample_rate * chunk_duration)
        max_chunks = int(self.vad.max_duration_s / chunk_duration)
        wait_timeout = timeout_s if timeout_s is not None else self.vad.no_speech_timeout_s
        no_speech_chunks = int(wait_timeout / chunk_duration)
        silence_limit_chunks = int(self.vad.silence_duration_s / chunk_duration)

        frames: list[np.ndarray] = []
        speech_started = False
        silent_chunks = 0

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_samples,
            ) as stream:
                for i in range(max_chunks):
                    data, _ = stream.read(chunk_samples)
                    chunk = np.asarray(data, dtype=np.int16).reshape(-1)
                    energy = self.vad.rms(chunk)
                    frames.append(chunk)

                    if energy >= self.vad.energy_threshold:
                        speech_started = True
                        silent_chunks = 0
                    else:
                        if speech_started:
                            silent_chunks += 1
                            if silent_chunks >= silence_limit_chunks:
                                break
                        elif i + 1 >= no_speech_chunks:
                            # No speech detected within wait_timeout
                            return b""
        except Exception as exc:
            raise MicrophoneUnavailableError(
                f"Could not access microphone: {exc}"
            ) from exc

        if not speech_started or not frames:
            return b""

        pcm = np.concatenate(frames)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()


class SpeechToText:
    """Speech-to-text transcriber.

    Args:
        engine: ``"whisper"`` (strictly offline faster-whisper, default for Mial),
            ``"google"`` (online SpeechRecognition), or ``"auto"``.
        model_size: Whisper model size (``"base"``, ``"small"``, ``"tiny"``).
        language: ISO language code (default ``"en"``).
        allow_online_fallback: Only used when ``engine="auto"``. When False,
            never sends audio to cloud services.
    """

    def __init__(
        self,
        engine: str = "whisper",
        model_size: str = "base",
        language: str = "en",
        allow_online_fallback: bool = False,
    ) -> None:
        self.engine = engine.lower()
        self.model_size = model_size
        self.language = language
        self.allow_online_fallback = allow_online_fallback
        self._whisper_model = None

    def _load_whisper(self):
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as exc:
                raise STTFailureError(
                    "Offline speech recognition requires 'faster-whisper'. "
                    "Install with: pip install faster-whisper"
                ) from exc
            self._whisper_model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
        return self._whisper_model

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes into English text."""
        if not wav_bytes:
            return ""

        if self.engine == "whisper":
            return self._transcribe_whisper(wav_bytes)
        if self.engine == "google":
            return self._transcribe_google(wav_bytes)
        if self.engine == "auto":
            try:
                return self._transcribe_whisper(wav_bytes)
            except Exception as exc:
                if not self.allow_online_fallback:
                    raise STTFailureError(
                        f"Offline Whisper STT failed and online fallback is disabled: {exc}"
                    ) from exc
                _log.warning("Whisper unavailable (%s); falling back to Google STT.", exc)
                return self._transcribe_google(wav_bytes)

        raise STTFailureError(f"Unsupported STT engine '{self.engine}'.")

    def _transcribe_whisper(self, wav_bytes: bytes) -> str:
        model = self._load_whisper()
        try:
            segments, _ = model.transcribe(
                io.BytesIO(wav_bytes),
                language=self.language,
                beam_size=1,
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            raise STTFailureError(f"Whisper transcription failed: {exc}") from exc

    def _transcribe_google(self, wav_bytes: bytes) -> str:
        try:
            import speech_recognition as sr  # type: ignore
        except ImportError as exc:
            raise STTFailureError(
                "Online STT requires 'SpeechRecognition'. Install with: pip install SpeechRecognition"
            ) from exc

        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)
        try:
            return str(recognizer.recognize_google(audio_data, language=self.language)).strip()
        except sr.UnknownValueError:
            return ""
        except Exception as exc:
            raise STTFailureError(f"Google STT failed: {exc}") from exc


class TextToSpeech:
    """Cross-platform subprocess speaker that passes text strictly via stdin."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._proc: Optional[subprocess.Popen] = None

    def speak(self, text: str, wait: bool = True) -> None:
        """Speak *text* aloud, passing text over stdin (never interpolated in shell)."""
        if not self.enabled or not text or not text.strip():
            return
        self.stop()

        try:
            if sys.platform == "win32":
                ps_script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$t = [Console]::In.ReadToEnd(); "
                    "if ($t) { $s.Speak($t) }"
                )
                self._proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            elif sys.platform == "darwin":
                self._proc = subprocess.Popen(
                    ["say", "-f", "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            else:
                self._proc = subprocess.Popen(
                    ["espeak-ng", "--stdin"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )

            assert self._proc.stdin is not None
            self._proc.stdin.write(text)
            self._proc.stdin.close()
            if wait:
                self._proc.wait(timeout=30)
        except Exception as exc:
            _log.debug("TTS subprocess unavailable: %s", exc)

    def stop(self) -> None:
        """Interrupt any currently running speech subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None


def beep(frequency: int = 880, duration_ms: int = 120) -> None:
    """Play a short non-blocking audible wake tone on Windows, or terminal bell elsewhere."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(frequency, duration_ms)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass

