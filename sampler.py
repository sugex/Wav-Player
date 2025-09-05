import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import resample
import os


class WavSampler:
    def __init__(self, root):
        self.root = root
        self.root.title("WAV Player with Reverb & Device Settings")
        self.root.geometry("700x500")
        self.root.configure(bg="#222222")

        # ---------------- MENU BAR ----------------
        menubar = tk.Menu(root)
        root.config(menu=menubar)

        # File
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load WAV Files", command=self.load_wav_files)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)

        # Settings
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Select Audio Output Device", command=self.open_audio_output_window)

        # Reverb
        reverb_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reverb", menu=reverb_menu)
        self.reverb_var = tk.IntVar(value=1)  # default ON
        reverb_menu.add_checkbutton(label="Enable Reverb", variable=self.reverb_var)
        reverb_menu.add_command(label="Parameters...", command=self.open_reverb_window)

        # ---------------- PLAYLIST ----------------
        playlist_frame = tk.Frame(root, bg="#333333")
        playlist_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(playlist_frame, text="Playlist (WAV files):", bg="#333333", fg="white",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.playlistbox = tk.Listbox(
            playlist_frame, bg="#111111", fg="white",
            selectbackground="#5555ff", activestyle="none",
            font=("Segoe UI", 10), height=10
        )
        self.playlistbox.pack(fill="both", expand=True, pady=5)
        self.playlistbox.bind("<<ListboxSelect>>", self.on_playlist_select)
        self.playlistbox.bind("<Double-Button-1>", self.on_double_click_play)

        # ---------------- CONTROLS ----------------
        controls_frame = tk.Frame(root, bg="#222222")
        controls_frame.pack(pady=5)

        self.play_button = tk.Button(
            controls_frame, text="Play", command=self.play_audio,
            state="disabled", bg="#444444", fg="white", width=10
        )
        self.play_button.pack(side="left", padx=10)

        self.stop_button = tk.Button(
            controls_frame, text="Stop", command=self.stop_audio,
            state="disabled", bg="#444444", fg="white", width=10
        )
        self.stop_button.pack(side="left", padx=10)

        # ---------------- STATUS ----------------
        self.status_label = tk.Label(
            root, text="No files loaded",
            bg="#222222", fg="#dddddd", font=("Segoe UI", 10)
        )
        self.status_label.pack(pady=5)

        # ---------------- INTERNAL STATE ----------------
        self.audio_files = []   # list of (filepath, audio_data, samplerate)
        self.current_index = None
        self.stream = None
        self.is_playing = False
        self.position = 0
        self.audio = None
        self.samplerate = None

        # Default values for reverb sliders
        self.delay_val = 111
        self.decay_val = 0.50
        self.mix_val = 16

        # thread-safe reverb flag
        self.reverb_enabled_flag = bool(self.reverb_var.get())
        self._reverb_updater_job = None

        # Default device (prefer DirectSound kalau ada)
        self.selected_device_index = self._find_default_output_device()
        self.status_label.config(text=f"Default output device index: {self.selected_device_index}")

    # ---------------- REVERB PARAMETER WINDOW ----------------
    def open_reverb_window(self):
        win = tk.Toplevel(self.root)
        win.title("Reverb Parameters")
        win.geometry("350x250")
        win.configure(bg="#333333")

        # Delay
        self.delay_label = tk.Label(win, text=f"Delay (ms): {self.delay_val}", bg="#333333", fg="white")
        self.delay_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.delay_slider = tk.Scale(
            win, from_=10, to=500, resolution=1, orient="horizontal",
            command=self.update_delay_label, bg="#333333", fg="white", highlightthickness=0
        )
        self.delay_slider.set(self.delay_val)
        self.delay_slider.pack(fill="x", padx=10)

        # Decay
        self.decay_label = tk.Label(win, text=f"Decay (seconds): {self.decay_val:.2f}", bg="#333333", fg="white")
        self.decay_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.decay_slider = tk.Scale(
            win, from_=0.1, to=5.0, resolution=0.01, orient="horizontal",
            command=self.update_decay_label, bg="#333333", fg="white", highlightthickness=0
        )
        self.decay_slider.set(self.decay_val)
        self.decay_slider.pack(fill="x", padx=10)

        # Mix
        self.mix_label = tk.Label(win, text=f"Mix (% wet): {self.mix_val}", bg="#333333", fg="white")
        self.mix_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.mix_slider = tk.Scale(
            win, from_=0, to=100, resolution=1, orient="horizontal",
            command=self.update_mix_label, bg="#333333", fg="white", highlightthickness=0
        )
        self.mix_slider.set(self.mix_val)
        self.mix_slider.pack(fill="x", padx=10)

    def update_delay_label(self, val):
        self.delay_val = int(val)
        self.delay_label.config(text=f"Delay (ms): {val}")

    def update_decay_label(self, val):
        self.decay_val = float(val)
        self.decay_label.config(text=f"Decay (seconds): {float(val):.2f}")

    def update_mix_label(self, val):
        self.mix_val = int(val)
        self.mix_label.config(text=f"Mix (% wet): {val}")

    # ---------------- FILE HANDLING ----------------
    def load_wav_files(self):
        files = filedialog.askopenfilenames(filetypes=[("WAV files", "*.wav")])
        if not files:
            return

        for file_path in files:
            try:
                data, samplerate = sf.read(file_path, dtype='float32')
                if data.ndim > 1:
                    data = np.mean(data, axis=1)  # convert stereo → mono
                self.audio_files.append((file_path, data, samplerate))
                self.playlistbox.insert(tk.END, os.path.basename(file_path))
            except Exception as e:
                messagebox.showerror("Error Loading WAV", f"Failed to load {file_path}:\n{e}")

        if self.audio_files and self.current_index is None:
            self.current_index = 0
            self.playlistbox.selection_clear(0, tk.END)
            self.playlistbox.selection_set(0)
            self.play_button.config(state="normal")
            self.update_status()

    def on_playlist_select(self, event):
        sel = self.playlistbox.curselection()
        if sel:
            self.current_index = sel[0]
            self.play_button.config(state="normal")
            self.stop_audio()
            self.update_status()

    def on_double_click_play(self, event):
        sel = self.playlistbox.curselection()
        if sel:
            self.current_index = sel[0]
            self.play_button.config(state="normal")
            self.stop_audio()
            self.update_status()
            self.play_audio()

    def update_status(self):
        if self.current_index is None:
            self.status_label.config(text="No file selected")
        else:
            filepath = self.audio_files[self.current_index][0]
            self.status_label.config(text=f"Selected: {filepath}")

    # ---------------- DEVICE SELECTION ----------------
    def _find_default_output_device(self):
        try:
            default_output_index = sd.default.device[1]
            default_device_info = sd.query_devices(default_output_index)
            hostapi_name = sd.query_hostapis(default_device_info['hostapi'])['name']

            if "DirectSound" in hostapi_name:
                return default_output_index
            else:
                for i, dev in enumerate(sd.query_devices()):
                    if dev['max_output_channels'] > 0:
                        host_name = sd.query_hostapis(dev['hostapi'])['name']
                        if "DirectSound" in host_name:
                            return i
            return default_output_index
        except Exception:
            return None

    def open_audio_output_window(self):
        window = tk.Toplevel(self.root)
        window.title("Select Audio Output Device")
        window.geometry("700x400")
        window.configure(bg="#222222")

        label = tk.Label(window, text="Select Audio Output Device:",
                         bg="#222222", fg="white", font=("Segoe UI", 12, "bold"))
        label.pack(pady=5)

        listbox = tk.Listbox(
            window, bg="#111111", fg="white",
            selectbackground="#5555ff", activestyle="none",
            font=("Segoe UI", 10)
        )
        listbox.pack(fill="both", expand=True, padx=10, pady=10)

        devices = []
        all_devices = sd.query_devices()
        groups = {}
        for idx, dev in enumerate(all_devices):
            hostapi = sd.query_hostapis(dev['hostapi'])['name']
            if dev['max_output_channels'] > 0:
                groups.setdefault(hostapi, []).append((idx, dev))

        for hostapi, devs in groups.items():
            listbox.insert(tk.END, f"=== {hostapi} ===")
            for idx, dev in devs:
                sr = int(dev['default_samplerate'])
                low_lat = dev['default_low_output_latency'] * 1000
                high_lat = dev['default_high_output_latency'] * 1000
                line = (f"[{idx}] {dev['name']} | SR: {sr} Hz | float32 | "
                        f"Lat: {low_lat:.1f}-{high_lat:.1f} ms")
                listbox.insert(tk.END, line)
                devices.append(idx)

        def on_select(evt):
            sel = listbox.curselection()
            if not sel:
                return
            line = listbox.get(sel[0])
            if line.startswith("==="):
                return
            try:
                dev_idx = int(line.split(']')[0][1:])
            except:
                return

            self.selected_device_index = dev_idx
            dev = sd.query_devices(dev_idx)
            sr = int(dev['default_samplerate'])
            low_lat = dev['default_low_output_latency'] * 1000
            high_lat = dev['default_high_output_latency'] * 1000
            self.status_label.config(
                text=(f"Output Device: {dev['name']} "
                      f"({sr} Hz, {low_lat:.1f}-{high_lat:.1f} ms)")
            )
            window.destroy()

        listbox.bind("<<ListboxSelect>>", on_select)

    # ---------------- REVERB FLAG SYNC ----------------
    def _start_reverb_updater(self):
        if self._reverb_updater_job is not None:
            return

        def _update():
            self.reverb_enabled_flag = bool(self.reverb_var.get())
            if self.stream is not None and getattr(self.stream, "active", False):
                self._reverb_updater_job = self.root.after(50, _update)
            else:
                self._reverb_updater_job = None

        self._reverb_updater_job = self.root.after(0, _update)

    def _stop_reverb_updater(self):
        if self._reverb_updater_job is not None:
            try:
                self.root.after_cancel(self._reverb_updater_job)
            except Exception:
                pass
            self._reverb_updater_job = None

    # ---------------- PLAYBACK ----------------
    def play_audio(self):
        if self.is_playing or self.current_index is None:
            return
        if self.selected_device_index is None:
            messagebox.showwarning("No output device", "Please select an audio output device in Settings.")
            return

        def start_playback():
            filepath, data, sr = self.audio_files[self.current_index]
            device_info = sd.query_devices(self.selected_device_index)
            device_sr = int(device_info['default_samplerate'])

            audio = data
            if sr != device_sr:
                num_samples = round(len(audio) * device_sr / sr)
                audio = resample(audio, num_samples)
                sr = device_sr

            self.audio = audio
            self.samplerate = sr
            self.position = 0

            self._start_reverb_updater()

            decay_time = max(0.0001, self.decay_val)
            mix_percent = self.mix_val / 100.0
            delay_ms = self.delay_val
            delay_samples = max(1, int(sr * delay_ms / 1000))
            feedback = np.exp(-delay_ms / 1000 / decay_time) if decay_time > 0 else 0.0

            delay_buffer = np.zeros(delay_samples, dtype=np.float32)
            delay_pos = 0

            def callback(outdata, frames, time, status):
                nonlocal delay_pos
                if status:
                    print(f"Playback status: {status}")

                chunk = self.audio[self.position:self.position + frames]
                if len(chunk) < frames:
                    chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')

                output = np.zeros(frames, dtype=np.float32)
                enabled = self.reverb_enabled_flag

                if enabled and delay_samples > 0:
                    for i in range(frames):
                        dry = chunk[i]
                        delayed_sample = delay_buffer[delay_pos]
                        wet = delayed_sample
                        out_sample = dry * (1 - mix_percent) + wet * mix_percent
                        delay_buffer[delay_pos] = dry + delayed_sample * feedback
                        delay_pos = (delay_pos + 1) % delay_samples
                        output[i] = out_sample
                else:
                    output[:] = chunk

                outdata[:] = output.reshape(-1, 1)
                self.position += frames
                if self.position >= len(self.audio):
                    raise sd.CallbackStop()

            try:
                self.stream = sd.OutputStream(
                    device=self.selected_device_index, channels=1,
                    samplerate=sr, callback=callback
                )
                self.stream.start()
                self.is_playing = True
                self.play_button.config(state="disabled")
                self.stop_button.config(state="normal")

                def check_stream():
                    if not self.stream.active:
                        # file selesai → pindah ke berikutnya
                        self.current_index = (self.current_index + 1) % len(self.audio_files)
                        if self.is_playing:  # loop kalau masih mode play
                            start_playback()
                        else:
                            self.is_playing = False
                            self.play_button.config(state="normal")
                            self.stop_button.config(state="disabled")
                            self._stop_reverb_updater()
                    else:
                        self.root.after(100, check_stream)

                self.root.after(100, check_stream)

            except Exception as e:
                messagebox.showerror("Playback Error", str(e))
                self.is_playing = False
                self.play_button.config(state="normal")
                self.stop_button.config(state="disabled")
                self._stop_reverb_updater()

        start_playback()

    def stop_audio(self):
        if self.stream and self.is_playing:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            self.is_playing = False
            self.play_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="Playback stopped")
        self._stop_reverb_updater()


if __name__ == "__main__":
    root = tk.Tk()
    app = WavSampler(root)
    root.mainloop()
