"""
smartml.mial.speaker
====================
Indian-English Neural Voice Speaker (`en-IN-NeerjaNeural` / `en-IN-PrabhatNeural`)
with instant `sounddevice.stop()` barge-in, backed by a dedicated apartment-threaded
Windows SAPI5 fallback (`pythoncom.CoInitialize()` + `SAPI.SpVoice`) when offline.
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

from ..agent.audio import TextToSpeech

_log = logging.getLogger("smartml.mial")

SVSFDefault = 0
SVSFlagsAsync = 1
SVSFPurgeBeforeSpeak = 2


class MialSpeaker:
    """Thread-safe, interruptible Indian-English (`en-IN`) neural + offline SAPI5 speaker."""

    def __init__(
        self,
        enabled: bool = True,
        rate: int = -1,
        voice_name: str = "en-IN-NeerjaNeural",
        prefer_neural_en_in: bool = False,
    ) -> None:
        self.enabled = enabled
        self.rate = rate
        self.voice_name = voice_name
        self.prefer_neural_en_in = prefer_neural_en_in
        self._stop_flag = threading.Event()
        self._neural_active = False

        self._cmd_queue: "queue.Queue[tuple[str, Optional[str], Optional[threading.Event]]]" = queue.Queue()
        self._sapi_ready = threading.Event()
        self._using_sapi = False
        self._bench_t0: float = 0.0
        self._bench_result: tuple[float, float] = (0.0, 0.0)
        self._fallback = TextToSpeech(enabled=enabled)
        self._thread: Optional[threading.Thread] = None

        if self.enabled and sys.platform == "win32":
            self._thread = threading.Thread(
                target=self._sapi_worker,
                name="MialSAPIThread",
                daemon=True,
            )
            self._thread.start()
            self._sapi_ready.wait(timeout=2.0)

    def set_rate(self, rate: int) -> None:
        """Adjust speech rate (-5 slowest to +5 fastest; -1 is natural Indian-English rhythm)."""
        self.rate = max(-5, min(5, int(rate)))
        if self._using_sapi:
            self._cmd_queue.put(("RATE", str(self.rate), None))

    def _rate_to_edge_pct(self) -> str:
        # Map slider (-5..+5) to edge-tts percentage string (e.g., -1 -> "-6%")
        pct = self.rate * 6
        return f"{pct:+d}%"

    def _try_speak_neural_en_in(self, text: str, wait: bool = True) -> bool:
        """Synthesise and play natural Indian-English neural voice via edge-tts + PyAV + sounddevice."""
        if not self.prefer_neural_en_in:
            return False
        try:
            import av  # type: ignore
            import edge_tts  # type: ignore
            import sounddevice as sd  # type: ignore
        except ImportError:
            return False

        self._stop_flag.clear()

        def _synth_and_play() -> bool:
            try:
                async def _fetch_mp3() -> bytes:
                    comm = edge_tts.Communicate(
                        text=text,
                        voice=self.voice_name,
                        rate=self._rate_to_edge_pct(),
                    )
                    mp3_buf = bytearray()
                    async for chunk in comm.stream():
                        if self._stop_flag.is_set():
                            return b""
                        if chunk.get("type") == "audio":
                            mp3_buf.extend(chunk["data"])
                    return bytes(mp3_buf)

                mp3_bytes = asyncio.run(_fetch_mp3())
                if not mp3_bytes or self._stop_flag.is_set():
                    return True

                container = av.open(io.BytesIO(mp3_bytes), format="mp3")
                resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
                pcm_chunks: list[np.ndarray] = []
                for frame in container.decode(audio=0):
                    for rframe in resampler.resample(frame):
                        pcm_chunks.append(rframe.to_ndarray().reshape(-1))
                if not pcm_chunks or self._stop_flag.is_set():
                    return True

                pcm = np.concatenate(pcm_chunks).astype(np.float32) / 32768.0
                self._neural_active = True
                sd.play(pcm, samplerate=24000)
                while sd.get_stream().active:
                    if self._stop_flag.is_set():
                        sd.stop()
                        break
                    time.sleep(0.03)
                self._neural_active = False
                return True
            except Exception as exc:
                self._neural_active = False
                _log.debug("Neural en-IN TTS fallback to SAPI (%s)", exc)
                return False

        if wait:
            return _synth_and_play()
        else:
            def _bg():
                if not _synth_and_play():
                    self._speak_sapi_or_fallback(text, wait=False)

            threading.Thread(target=_bg, daemon=True).start()
            return True

    def _sapi_worker(self) -> None:
        voice = None
        pythoncom_mod = None
        try:
            import pythoncom  # type: ignore
            import win32com.client  # type: ignore

            pythoncom.CoInitialize()
            pythoncom_mod = pythoncom
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            try:
                voices = voice.GetVoices()
                # Prefer Indian English (Heera / Ravi / India) first, then warm female voices
                preferred = ("heera", "ravi", "india", "en-in", "natural", "zira", "hazel")
                chosen_voice = None
                for kw in preferred:
                    for i in range(voices.Count):
                        v = voices.Item(i)
                        desc = str(v.GetDescription()).lower()
                        if kw in desc:
                            chosen_voice = v
                            break
                    if chosen_voice is not None:
                        break
                if chosen_voice is not None:
                    voice.Voice = chosen_voice
            except Exception:
                pass
            voice.Rate = self.rate
            try:
                orig_vol = int(voice.Volume)
                voice.Volume = 0
                voice.Speak("ready", SVSFDefault)
                voice.Volume = orig_vol
            except Exception:
                voice.Speak("", SVSFlagsAsync)
            self._using_sapi = True
        except Exception as exc:
            _log.debug("SAPI.SpVoice unavailable (%s); using subprocess TTS fallback.", exc)
            self._using_sapi = False
        finally:
            self._sapi_ready.set()

        if not self._using_sapi or voice is None:
            if pythoncom_mod is not None:
                try:
                    pythoncom_mod.CoUninitialize()
                except Exception:
                    pass
            return

        try:
            while True:
                try:
                    action, text, done_evt = self._cmd_queue.get(timeout=0.05)
                except queue.Empty:
                    try:
                        pythoncom_mod.PumpWaitingMessages()
                    except Exception:
                        pass
                    continue
                try:
                    if action == "QUIT":
                        voice.Speak("", SVSFlagsAsync | SVSFPurgeBeforeSpeak)
                        break
                    elif action == "RATE" and text is not None:
                        voice.Rate = int(text)
                    elif action == "STOP":
                        voice.Speak("", SVSFlagsAsync | SVSFPurgeBeforeSpeak)
                    elif action == "BENCHMARK" and text:
                        voice.Speak(text, SVSFlagsAsync | SVSFPurgeBeforeSpeak)
                        stream_ms: Optional[float] = None
                        phoneme_ms: Optional[float] = None
                        deadline = time.perf_counter() + 3.0
                        while time.perf_counter() < deadline:
                            try:
                                pythoncom_mod.PumpWaitingMessages()
                            except Exception:
                                pass
                            st = voice.Status
                            if stream_ms is None and int(st.RunningState) == 2:
                                stream_ms = (time.perf_counter() - self._bench_t0) * 1000.0
                            if phoneme_ms is None and int(getattr(st, "PhonemeId", 0) or 0) > 0:
                                phoneme_ms = (time.perf_counter() - self._bench_t0) * 1000.0
                                break
                            if stream_ms is not None and int(st.RunningState) == 1:
                                break
                            time.sleep(0.001)
                        now_ms = (time.perf_counter() - self._bench_t0) * 1000.0
                        self._bench_result = (
                            stream_ms if stream_ms is not None else now_ms,
                            phoneme_ms if phoneme_ms is not None else (stream_ms if stream_ms is not None else now_ms),
                        )
                    elif action == "SPEAK" and text:
                        if done_evt is not None:
                            voice.Speak(text, SVSFlagsAsync | SVSFPurgeBeforeSpeak)
                            while not voice.WaitUntilDone(50):
                                try:
                                    pythoncom_mod.PumpWaitingMessages()
                                except Exception:
                                    pass
                                if not self._cmd_queue.empty():
                                    break
                        else:
                            voice.Speak(text, SVSFlagsAsync | SVSFPurgeBeforeSpeak)
                except Exception as exc:
                    _log.debug("SAPI speak error: %s", exc)
                finally:
                    if done_evt is not None:
                        done_evt.set()
        finally:
            if pythoncom_mod is not None:
                try:
                    pythoncom_mod.CoUninitialize()
                except Exception:
                    pass

    @property
    def using_sapi(self) -> bool:
        return self._using_sapi

    def _speak_sapi_or_fallback(self, text: str, wait: bool = True) -> None:
        if self._using_sapi:
            done_evt = threading.Event() if wait else None
            self._cmd_queue.put(("SPEAK", text.strip(), done_evt))
            if done_evt is not None:
                done_evt.wait(timeout=30.0)
            return
        self._fallback.speak(text, wait=wait)

    def measure_first_audio_ms(
        self,
        text: str = "Hi, I'm Mial. What would you like to do?",
    ) -> tuple[float, float]:
        """Measure real elapsed ms from trigger to (SAPI stream active, first audible phoneme).

        Leaves SAPI speaking so the caller can immediately benchmark mid-speech ``stop()``.
        """
        if not self.enabled or not text or not text.strip():
            return (0.0, 0.0)
        if self._using_sapi:
            done_evt = threading.Event()
            self._bench_result = (0.0, 0.0)
            self._bench_t0 = time.perf_counter()
            self._cmd_queue.put(("BENCHMARK", text.strip(), done_evt))
            done_evt.wait(timeout=5.0)
            return self._bench_result
        ms = self.speak(text.strip(), wait=True)
        return (ms, ms)

    def speak(self, text: str, wait: bool = True) -> float:
        """Speak *text* using Indian-English neural TTS (or offline SAPI5 fallback)."""
        t0 = time.perf_counter()
        if not self.enabled or not text or not text.strip():
            return 0.0

        if self._try_speak_neural_en_in(text.strip(), wait=wait):
            return (time.perf_counter() - t0) * 1000.0

        self._speak_sapi_or_fallback(text.strip(), wait=wait)
        return (time.perf_counter() - t0) * 1000.0

    def stop(self) -> float:
        """Immediately stop neural or SAPI5 speech (0 ms barge-in)."""
        t0 = time.perf_counter()
        self._stop_flag.set()
        if self._neural_active:
            try:
                import sounddevice as sd  # type: ignore
                sd.stop()
            except Exception:
                pass
            self._neural_active = False

        if not self.enabled:
            return 0.0

        if self._using_sapi:
            done_evt = threading.Event()
            self._cmd_queue.put(("STOP", None, done_evt))
            done_evt.wait(timeout=1.0)
            return (time.perf_counter() - t0) * 1000.0

        self._fallback.stop()
        return (time.perf_counter() - t0) * 1000.0

    def shutdown(self) -> None:
        self._stop_flag.set()
        if self._using_sapi:
            self._cmd_queue.put(("QUIT", None, None))
        else:
            self._fallback.stop()
