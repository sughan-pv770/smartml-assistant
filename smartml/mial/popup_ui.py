"""
smartml.mial.popup_ui
=====================
Desktop popup window for Mial, the voice assistant for SmartML.

Features:
- Global hotkey (default Ctrl+Alt+M) wakes Mial and greets on first wake
  ("Hi, I'm Mial. What would you like to do?")
- Clean status indicator (IDLE, LISTENING, RUNNING, SPEAKING, STOPPED, ERROR)
- Works with any CSV, TSV, Excel, Parquet, Feather, or JSON dataset
- Displays full metrics and tables on screen while keeping spoken answers short and symbol-free
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from ..agent.audio import MicrophoneUnavailableError, STTFailureError, beep
from ..agent.hotkey import HotkeyListener
from .controller import MialController
from .llm_brain import MialLLMBrain


def _interpolate_color(c1: str, c2: str, factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def _format_hotkey_label(hotkey: str) -> str:
    parts = [p.strip("<> ").capitalize() for p in hotkey.split("+") if p.strip()]
    return "+".join(parts) if parts else "Ctrl+Alt+M"


class MialPopupApp:
    """Modern gradient desktop popup window for Mial."""

    def __init__(
        self,
        controller: MialController,
        initial_dataset: Optional[str] = None,
        initial_command: Optional[str] = None,
    ) -> None:
        self.controller = controller
        self.controller.speaker.set_rate(-1)
        self.brain = MialLLMBrain(
            agent=self.controller.agent,
            offline_only=self.controller.config.offline_only,
        )
        self.initial_dataset = initial_dataset
        self.initial_command = initial_command
        self._busy = False
        self._greeted_once = False
        self._hotkey_display = _format_hotkey_label(self.controller.config.hotkey)

    def run(self) -> int:
        import tkinter as tk
        from tkinter import filedialog, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.root = tk.Tk()
        self.root.title("Mial - SmartML Voice Assistant")
        self.root.geometry("600x740")
        self.root.configure(bg="#090d16")
        self.root.attributes("-topmost", True)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x_pos = max(40, screen_w - 640)
        y_pos = max(40, screen_h - 810)
        self.root.geometry(f"600x740+{x_pos}+{y_pos}")

        self._build_ui(tk, ttk)

        self._hotkey_listener: Optional[HotkeyListener] = None
        try:
            self._hotkey_listener = HotkeyListener(
                self.controller.config.hotkey,
                lambda: self.root.after(0, self.on_hotkey_triggered),
            )
            self._hotkey_listener.start()
        except Exception:
            pass

        if self.initial_dataset:
            self.root.after(150, lambda: self._load_dataset_path(self.initial_dataset, auto_cmd=self.initial_command))
        else:
            self.root.after(150, self._show_ready_banner)
            if self.initial_command:
                self.root.after(500, lambda: self._submit_text_command(self.initial_command))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()
        return 0

    def _draw_gradient_header(self, canvas, width: int, height: int) -> None:
        canvas.delete("grad_stripe")
        c_top = "#2e1065"      # Deep obsidian violet
        c_mid = "#312e81"      # Indigo
        c_bot = "#090d16"      # Dark midnight background
        steps = max(1, height)
        for y in range(steps):
            frac = y / steps
            if frac < 0.5:
                color = _interpolate_color(c_top, c_mid, frac * 2.0)
            else:
                color = _interpolate_color(c_mid, c_bot, (frac - 0.5) * 2.0)
            canvas.create_line(0, y, width, y, fill=color, tags="grad_stripe")
        canvas.create_line(0, height - 1, width, height - 1, fill="#1e293b", width=1, tags="grad_stripe")

    def _build_ui(self, tk, ttk) -> None:
        # 1. Gradient Header Canvas with overlaid widgets
        self.header_canvas = tk.Canvas(
            self.root,
            height=78,
            bg="#090d16",
            highlightthickness=0,
            bd=0,
        )
        self.header_canvas.pack(fill="x")

        def _on_header_resize(event):
            self._draw_gradient_header(self.header_canvas, event.width, event.height)

        self.header_canvas.bind("<Configure>", _on_header_resize)

        header_content = tk.Frame(self.header_canvas, bg="", bd=0)
        self.header_canvas.create_window((14, 12), window=header_content, anchor="nw")

        self.state_pill = tk.Label(
            header_content,
            text="IDLE",
            font=("Consolas", 10, "bold"),
            bg="#0b1329",
            fg="#38bdf8",
            padx=12,
            pady=5,
            relief="solid",
            bd=1,
            highlightthickness=0,
        )
        self.state_pill.pack(side="left", padx=(0, 14))

        title_box = tk.Frame(header_content, bg="")
        title_box.pack(side="left", fill="x", expand=True)

        tk.Label(
            title_box,
            text="MIAL  |  Voice-First ML Assistant",
            font=("Segoe UI", 12, "bold"),
            bg="#181938",
            fg="#f8fafc",
            anchor="w",
            padx=4,
        ).pack(fill="x")

        self.status_lbl = tk.Label(
            title_box,
            text=f"{self.brain.active_backend_label}  |  Hotkey: {self._hotkey_display}",
            font=("Segoe UI", 9),
            bg="#181938",
            fg="#93c5fd",
            anchor="w",
            padx=4,
        )
        self.status_lbl.pack(fill="x")

        self.topmost_var = tk.BooleanVar(value=True)
        pin_btn = tk.Checkbutton(
            self.header_canvas,
            text="Pin Window",
            variable=self.topmost_var,
            command=lambda: self.root.attributes("-topmost", self.topmost_var.get()),
            bg="#1e1b4b",
            fg="#e0e7ff",
            selectcolor="#0f172a",
            activebackground="#312e81",
            activeforeground="#ffffff",
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=3,
            relief="solid",
            bd=1,
        )
        self.header_canvas.create_window((585, 24), window=pin_btn, anchor="ne")

        # 2. Dataset & Target Card (Gradient glass styling)
        ds_frame = tk.Frame(self.root, bg="#111827", padx=12, pady=7, highlightbackground="#1e293b", highlightthickness=1)
        ds_frame.pack(fill="x", padx=10, pady=(8, 4))

        tk.Button(
            ds_frame,
            text="Load Dataset",
            command=self._on_browse_dataset,
            bg="#4f46e5",
            fg="#ffffff",
            activebackground="#4338ca",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
            cursor="hand2",
        ).pack(side="left")

        self.dataset_badge = tk.Label(
            ds_frame,
            text="No dataset loaded",
            font=("Segoe UI", 9),
            bg="#111827",
            fg="#cbd5e1",
            padx=10,
            anchor="w",
        )
        self.dataset_badge.pack(side="left", fill="x", expand=True)

        tk.Label(
            ds_frame,
            text="Target:",
            font=("Segoe UI", 9, "bold"),
            bg="#111827",
            fg="#38bdf8",
        ).pack(side="left", padx=(4, 4))

        self.target_var = tk.StringVar(value="[Select]")
        self.target_combo = ttk.Combobox(
            ds_frame,
            textvariable=self.target_var,
            values=["[Select]"],
            width=16,
            state="readonly",
        )
        self.target_combo.pack(side="left")
        self.target_combo.bind("<<ComboboxSelected>>", self._on_target_selected)

        # 3. Voice Rate & API Key Card
        voice_frame = tk.Frame(self.root, bg="#0f172a", padx=12, pady=5, highlightbackground="#1e293b", highlightthickness=1)
        voice_frame.pack(fill="x", padx=10, pady=2)

        tk.Label(
            voice_frame,
            text="Speed:",
            font=("Segoe UI", 8, "bold"),
            bg="#0f172a",
            fg="#94a3b8",
        ).pack(side="left")

        self.speed_scale = tk.Scale(
            voice_frame,
            from_=-4,
            to=2,
            orient="horizontal",
            length=90,
            bg="#0f172a",
            fg="#38bdf8",
            highlightthickness=0,
            troughcolor="#090d16",
            command=lambda val: self.controller.speaker.set_rate(int(float(val))),
        )
        self.speed_scale.set(-1)
        self.speed_scale.pack(side="left", padx=(4, 14))

        tk.Label(
            voice_frame,
            text="NIM Key:",
            font=("Segoe UI", 8),
            bg="#0f172a",
            fg="#94a3b8",
        ).pack(side="left", padx=(4, 4))

        self.api_key_var = tk.StringVar(value=self.brain.nvidia_api_key or self.brain.gemini_api_key)
        tk.Entry(
            voice_frame,
            textvariable=self.api_key_var,
            show="*",
            width=20,
            bg="#090d16",
            fg="#f8fafc",
            insertbackground="#38bdf8",
            relief="solid",
            bd=1,
            highlightthickness=0,
        ).pack(side="left", padx=2)

        tk.Button(
            voice_frame,
            text="Apply",
            command=self._on_apply_api_key,
            bg="#1e293b",
            fg="#38bdf8",
            activebackground="#334155",
            relief="flat",
            font=("Segoe UI", 8, "bold"),
            padx=9,
            cursor="hand2",
        ).pack(side="left", padx=(6, 0))

        # 4. Action Skill Bar (Indigo Gradient Accent Buttons)
        skills_frame = tk.Frame(self.root, bg="#090d16", padx=8, pady=4)
        skills_frame.pack(fill="x")

        skill_buttons = [
            ("Summary", "Summary of the data"),
            ("Clean & Train", "Clean my dataset and train the best model"),
            ("Tune", "Tune the winning model"),
            ("Why Winner", "Why did that model win?"),
            ("Importance", "What matters most?"),
            ("Report", "Show me the report"),
            ("Undo", "Undo"),
        ]
        for label, cmd in skill_buttons:
            tk.Button(
                skills_frame,
                text=label,
                command=lambda c=cmd: self._submit_text_command(c),
                bg="#1e1b4b",
                fg="#c7d2fe",
                activebackground="#3730a3",
                activeforeground="#ffffff",
                relief="solid",
                bd=1,
                font=("Segoe UI", 8, "bold"),
                padx=8,
                pady=3,
                cursor="hand2",
            ).pack(side="left", padx=2, pady=2)

        # 5. Screen Output & Transcript Feed
        chat_container = tk.Frame(self.root, bg="#090d16", padx=10, pady=4)
        chat_container.pack(fill="both", expand=True)

        self.chat_box = tk.Text(
            chat_container,
            bg="#070b14",
            fg="#e2e8f0",
            font=("Segoe UI", 10),
            wrap="word",
            relief="solid",
            bd=1,
            padx=12,
            pady=12,
            state="disabled",
            highlightbackground="#1e293b",
            highlightthickness=1,
        )
        scrollbar = ttk.Scrollbar(chat_container, command=self.chat_box.yview)
        self.chat_box.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.chat_box.pack(side="left", fill="both", expand=True)

        self.chat_box.tag_configure("user", foreground="#38bdf8", font=("Segoe UI", 10, "bold"))
        self.chat_box.tag_configure("skill", foreground="#a5b4fc", font=("Consolas", 9))
        self.chat_box.tag_configure("speech", foreground="#34d399", font=("Segoe UI", 10, "italic"))
        self.chat_box.tag_configure("mial", foreground="#f8fafc", font=("Segoe UI", 10))
        self.chat_box.tag_configure("system", foreground="#94a3b8", font=("Segoe UI", 9))

        # 6. Bottom Voice & Text Bar with glowing buttons
        bottom = tk.Frame(self.root, bg="#0b1329", padx=10, pady=10, highlightbackground="#1e293b", highlightthickness=1)
        bottom.pack(fill="x", side="bottom")

        self.mic_btn = tk.Button(
            bottom,
            text=f"Speak ({self._hotkey_display})",
            command=self.on_hotkey_triggered,
            bg="#059669",
            fg="#ffffff",
            activebackground="#047857",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            cursor="hand2",
        )
        self.mic_btn.pack(side="left", padx=(0, 6))

        self.stop_btn = tk.Button(
            bottom,
            text="Stop",
            command=self._on_stop_clicked,
            bg="#e11d48",
            fg="#ffffff",
            activebackground="#be123c",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            cursor="hand2",
        )
        self.stop_btn.pack(side="left", padx=(0, 8))

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            bottom,
            textvariable=self.input_var,
            bg="#070b14",
            fg="#f8fafc",
            insertbackground="#38bdf8",
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            highlightbackground="#1e293b",
            highlightthickness=1,
        )
        self.input_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.input_entry.bind("<Return>", lambda _e: self._on_send_clicked())

        tk.Button(
            bottom,
            text="Send",
            command=self._on_send_clicked,
            bg="#4f46e5",
            fg="#ffffff",
            activebackground="#4338ca",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=6,
            cursor="hand2",
        ).pack(side="right")

    # ------------------------------------------------------------------ #
    # UI Helpers
    # ------------------------------------------------------------------ #

    def _append_chat(self, sender: str, text: str, tag: str = "mial", body_tag: Optional[str] = None) -> None:
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"{sender}\n", (tag,))
        self.chat_box.insert("end", f"{text}\n\n", (body_tag or ("mial" if tag != "system" else "system"),))
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _set_state_ui(self, badge: str, status_text: str) -> None:
        color_map = {
            "IDLE": "#38bdf8",
            "READY": "#38bdf8",
            "LISTENING": "#f59e0b",
            "RUNNING": "#a78bfa",
            "SPEAKING": "#4ade80",
            "STOPPED": "#f87171",
            "ERROR": "#f87171",
        }
        self.state_pill.configure(text=badge, fg=color_map.get(badge, "#38bdf8"))
        self.status_lbl.configure(text=status_text)
        self.root.update_idletasks()

    def _show_ready_banner(self) -> None:
        privacy_line = (
            "Privacy: 100% local offline mode (--offline active; nothing leaves your machine)."
            if self.brain.offline_only or not self.brain.nvidia_api_key
            else "Privacy: Audio stays local in RAM; transcript + aggregate tool summaries (never raw cell values) go to NVIDIA NIM."
        )
        banner = (
            f"Brain: {self.brain.active_backend_label}\n"
            f"Speech-to-text: Offline Whisper (first voice use downloads ~150 MB local Whisper model once)\n"
            f"{privacy_line}\n"
            f"Wake hotkey: Press {self._hotkey_display} or click Speak, or type a command below."
        )
        self._append_chat("Mial Ready", banner, "system")

    def _speak_bg(self, speech_text: str) -> None:
        self.controller.speaker.speak(speech_text, wait=True)
        self.root.after(
            0,
            lambda: self._set_state_ui(
                "IDLE",
                f"{self.brain.active_backend_label} | Hotkey: {self._hotkey_display}",
            ),
        )

    # ------------------------------------------------------------------ #
    # Dataset & Target Handlers
    # ------------------------------------------------------------------ #

    def _on_browse_dataset(self) -> None:
        chosen = self.filedialog.askopenfilename(
            title="Select Dataset (CSV, Excel, Parquet, JSON)",
            filetypes=[
                ("Dataset files", "*.csv *.tsv *.xlsx *.xls *.json *.parquet *.feather"),
                ("All files", "*.*"),
            ],
        )
        if chosen:
            self._load_dataset_path(chosen)

    def _load_dataset_path(self, path: str, auto_cmd: Optional[str] = None) -> None:
        out = self.brain.executor.execute_one("load_dataset", {"path": path, "allow_outside": True})
        if out.get("status") == "ok" and self.controller.agent.assistant is not None:
            df = self.controller.agent.assistant.session.raw_df
            fname = os.path.basename(path)
            self.dataset_badge.configure(
                text=f"{fname} ({df.shape[0]} x {df.shape[1]})",
                fg="#4ade80",
            )
            cols = ["[Select]"] + [str(c) for c in df.columns]
            self.target_combo.configure(values=cols)
            self.target_var.set("[Select]")

            msg = f"Loaded {fname} ({df.shape[0]} rows x {df.shape[1]} columns)."
            for w in out.get("warnings") or []:
                msg += f"\nNote: {w}"
            self._append_chat("Mial (Screen)", msg, "mial")

            if auto_cmd:
                self.root.after(250, lambda: self._submit_text_command(auto_cmd))
            else:
                spoken = f"Loaded {df.shape[0]} rows and {df.shape[1]} columns. What would you like to do?"
                self._append_chat("Mial (Spoken)", spoken, "speech", "speech")
                if self.controller.config.speak_replies:
                    threading.Thread(target=self._speak_bg, args=(spoken,), daemon=True).start()
        else:
            self._append_chat("Mial (Notice)", str(out.get("error", "Could not load file.")), "system")

    def _on_target_selected(self, _event=None) -> None:
        val = self.target_var.get()
        if val and val != "[Select]" and self.controller.agent.assistant is not None:
            msg = self.controller.agent.assistant.set_target(val)
            self._append_chat("Target Set", msg, "skill", "skill")

    def _on_apply_api_key(self) -> None:
        key = self.api_key_var.get().strip()
        self.brain.set_api_key(key)
        self._set_state_ui("IDLE", f"{self.brain.active_backend_label} | Hotkey: {self._hotkey_display}")
        self._append_chat("Configuration", f"Active brain: {self.brain.active_backend_label}", "skill", "skill")

    # ------------------------------------------------------------------ #
    # Hotkey Wake & Speech Capture
    # ------------------------------------------------------------------ #

    def on_hotkey_triggered(self) -> None:
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass

        self.controller.speaker.stop()
        if self._busy:
            return

        self._busy = True
        self._set_state_ui("LISTENING", "Listening... speak your request")
        self.mic_btn.configure(text="Listening...", bg="#d97706")
        threading.Thread(target=self._wake_and_capture_worker, daemon=True).start()

    def _wake_and_capture_worker(self) -> None:
        try:
            if not self._greeted_once:
                self._greeted_once = True
                if self.controller.config.speak_replies:
                    self.controller.speaker.speak(self.controller.config.greeting, wait=True)
            else:
                beep(880, 90)

            wav_bytes = self.controller.agent.recorder.record_utterance(timeout_s=7.0)
            if not wav_bytes:
                self.root.after(0, self._on_voice_empty)
                return
            self.root.after(0, lambda: self._set_state_ui("RUNNING", "Transcribing speech offline..."))
            transcript = self.controller.agent.stt.transcribe(wav_bytes)
            if not transcript.strip():
                self.root.after(0, self._on_voice_empty)
                return
            self.root.after(0, lambda: self._execute_user_request(transcript.strip()))
        except (MicrophoneUnavailableError, STTFailureError) as exc:
            self.root.after(0, lambda: self._on_voice_error(str(exc)))
        except Exception as exc:
            self.root.after(0, lambda: self._on_voice_error(f"Voice error: {exc}"))

    def _on_voice_empty(self) -> None:
        self._busy = False
        self.mic_btn.configure(text=f"Speak ({self._hotkey_display})", bg="#059669")
        msg = "I didn't catch that. Please say it again or type below."
        self._set_state_ui("IDLE", msg)
        if self.controller.config.speak_replies:
            threading.Thread(target=self._speak_bg, args=(msg,), daemon=True).start()

    def _on_voice_error(self, err_msg: str) -> None:
        self._busy = False
        self.mic_btn.configure(text=f"Speak ({self._hotkey_display})", bg="#059669")
        self._set_state_ui("ERROR", "Microphone or speech-to-text unavailable; type below.")
        self._append_chat("Voice Notice", err_msg, "system")

    # ------------------------------------------------------------------ #
    # Request Execution
    # ------------------------------------------------------------------ #

    def _on_send_clicked(self) -> None:
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self._submit_text_command(text)

    def _submit_text_command(self, text: str) -> None:
        self.controller.speaker.stop()
        if self._busy:
            return
        self._busy = True
        self._execute_user_request(text)

    def _execute_user_request(self, text: str) -> None:
        self.mic_btn.configure(text=f"Speak ({self._hotkey_display})", bg="#059669")
        self._append_chat("You", text, "user")
        self._set_state_ui("RUNNING", "Running SmartML tools...")
        threading.Thread(target=self._brain_worker, args=(text,), daemon=True).start()

    def _on_interim_plan(self, interim_info: dict[str, Any]) -> None:
        calls = [c.get("tool", "") for c in (interim_info.get("tool_calls") or [])]
        interim_speech = str(interim_info.get("speech") or "").strip()
        plan_line = " -> ".join(calls) if calls else "processing"
        self.root.after(0, lambda: self._set_state_ui("RUNNING", f"Executing: {plan_line}"))
        self.root.after(0, lambda: self._append_chat("Tools", plan_line, "skill", "skill"))
        if interim_speech and self.controller.config.speak_replies:
            self.controller.speaker.speak(interim_speech, wait=False)

    def _brain_worker(self, text: str) -> None:
        response = self.brain.process(
            text,
            runner=self.controller.runner,
            on_interim=self._on_interim_plan,
        )
        self.root.after(0, lambda: self._on_brain_finished(response))

    def _on_brain_finished(self, response) -> None:
        self._busy = False
        if self.controller.agent.has_dataset and self.controller.agent.assistant is not None:
            s = self.controller.agent.assistant.session
            fname = os.path.basename(self.controller.agent.current_file or "dataset")
            self.dataset_badge.configure(
                text=f"{fname} ({s.raw_df.shape[0]} x {s.raw_df.shape[1]})",
                fg="#4ade80",
            )
            cols = ["[Select]"] + [str(c) for c in s.raw_df.columns]
            self.target_combo.configure(values=cols)
            if s.target:
                self.target_var.set(s.target)

        self._append_chat("Mial (Screen)", response.display_text, "mial")
        if response.speech_text:
            self._append_chat("Mial (Spoken)", response.speech_text, "speech", "speech")
            if self.controller.config.speak_replies:
                self._set_state_ui("SPEAKING", "Speaking reply...")
                threading.Thread(target=self._speak_bg, args=(response.speech_text,), daemon=True).start()
            else:
                self._set_state_ui("IDLE", f"{self.brain.active_backend_label} | Hotkey: {self._hotkey_display}")
        else:
            self._set_state_ui("IDLE", f"{self.brain.active_backend_label} | Hotkey: {self._hotkey_display}")

    def _on_stop_clicked(self) -> None:
        self.controller.speaker.stop()
        cancelled = self.controller.runner.cancel()
        self._busy = False
        self.mic_btn.configure(text=f"Speak ({self._hotkey_display})", bg="#059669")
        if cancelled:
            self._append_chat("Stopped", "Cancelled running job.", "system")
        self._set_state_ui("STOPPED", "Stopped")

    def _on_close(self) -> None:
        try:
            self.controller.speaker.stop()
            self.controller.runner.cancel()
            if self._hotkey_listener is not None:
                self._hotkey_listener.stop()
            self.controller.speaker.shutdown()
        except Exception:
            pass
        self.root.destroy()

