"""
Low-latency audio playback using sounddevice.

AudioPlayer uses a callback-based OutputStream so that monitor sounds
(from the DrumMonitor) can be mixed into the same stream as the backing
track — no second device, no device-exclusivity conflicts.

DrumMonitor synthesises simple percussive sounds from numpy and feeds them
into the AudioPlayer mix queue in real time as MIDI hits arrive.

Audio interface / driver notes (Windows):
  - For lowest latency use WASAPI exclusive mode (set via device selection)
  - ASIO drivers (e.g. ASIO4ALL) give the best results with non-dedicated hardware
  - With a good audio interface you can expect 5–10 ms output latency
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


# ---------------------------------------------------------------------------
# DrumMonitor — synthesised percussive sounds mixed in real time
# ---------------------------------------------------------------------------

def _make_kick(sr: int) -> np.ndarray:
    """80–50 Hz pitch-sweep with exponential decay (~300 ms)."""
    dur = 0.30
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    freq = 150 * np.exp(-t * 18) + 45
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 14)
    click = np.exp(-t * 300) * 0.6     # sharp transient
    return ((np.sin(phase) + click) * env).astype(np.float32)


def _make_snare(sr: int) -> np.ndarray:
    """Tone body + noise crack (~150 ms)."""
    dur = 0.15
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    env = np.exp(-t * 28)
    body = np.sin(2 * np.pi * 220 * t) * 0.35
    crack = np.random.default_rng(0).standard_normal(len(t)) * 0.65
    return ((body + crack) * env).astype(np.float32)


def _make_hihat_closed(sr: int) -> np.ndarray:
    """Tight high-frequency noise (~35 ms)."""
    dur = 0.035
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    env = np.exp(-t * 160)
    noise = np.random.default_rng(1).standard_normal(len(t))
    hpf = np.diff(noise, prepend=noise[0])   # crude high-pass
    return (hpf * env * 0.55).astype(np.float32)


def _make_hihat_open(sr: int) -> np.ndarray:
    """Open hi-hat — same character, longer tail (~280 ms)."""
    dur = 0.28
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    env = np.exp(-t * 13)
    noise = np.random.default_rng(2).standard_normal(len(t))
    hpf = np.diff(noise, prepend=noise[0])
    return (hpf * env * 0.50).astype(np.float32)


def _make_tom(sr: int, freq: float = 110.0) -> np.ndarray:
    """Mid-range pitch-sweep tom (~200 ms)."""
    dur = 0.20
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    sweep = freq * np.exp(-t * 10) + freq * 0.25
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    env = np.exp(-t * 18)
    return (np.sin(phase) * env).astype(np.float32)


def _make_cymbal(sr: int, decay: float = 9.0) -> np.ndarray:
    """Bright noise cymbal — decay controls length (~500 ms for crash)."""
    dur = 0.50
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    env = np.exp(-t * decay)
    noise = np.random.default_rng(3).standard_normal(len(t))
    hpf = np.diff(noise, prepend=noise[0])
    return (hpf * env * 0.40).astype(np.float32)


class DrumMonitor:
    """
    Synthesises percussive sounds and feeds them to an AudioPlayer mix queue.

    Usage:
        monitor = DrumMonitor(volume=0.8)
        monitor.attach(player)          # give it the player's queue
        listener.set_monitor(monitor)   # MIDI events will call monitor.trigger()
    """

    # MIDI note → sample key
    _NOTE_MAP: dict[int, str] = {
        **{n: "kick"         for n in (35, 36)},
        **{n: "snare"        for n in (38, 39, 40)},
        **{n: "hihat_closed" for n in (42, 44)},
        **{n: "hihat_open"   for n in (46,)},
        **{n: "tom_low"      for n in (41, 43, 45)},
        **{n: "tom_mid"      for n in (47, 48)},
        **{n: "tom_high"     for n in (50,)},
        **{n: "crash"        for n in (49, 52, 55, 57)},
        **{n: "ride"         for n in (51, 53, 59)},
    }

    def __init__(self, volume: float = 0.75, samplerate: int = 44100) -> None:
        self.volume = max(0.0, min(1.0, volume))
        self._sr = samplerate
        self._samples = self._build_samples()
        self._queue: queue.SimpleQueue | None = None

    def _build_samples(self) -> dict[str, np.ndarray]:
        sr = self._sr
        return {
            "kick":         _make_kick(sr),
            "snare":        _make_snare(sr),
            "hihat_closed": _make_hihat_closed(sr),
            "hihat_open":   _make_hihat_open(sr),
            "tom_low":      _make_tom(sr, 80.0),
            "tom_mid":      _make_tom(sr, 110.0),
            "tom_high":     _make_tom(sr, 160.0),
            "crash":        _make_cymbal(sr, decay=8.0),
            "ride":         _make_cymbal(sr, decay=14.0),
        }

    def attach(self, player: "AudioPlayer") -> None:
        """Connect this monitor to an AudioPlayer's mix queue."""
        self._queue = player._monitor_queue

    def trigger(self, note: int, velocity: int) -> None:
        """Called (from the MIDI thread) when a drum is hit."""
        if self._queue is None or self.volume == 0:
            return
        key = self._NOTE_MAP.get(note, "hihat_closed")
        sample = self._samples[key]
        scale = self.volume * (velocity / 127.0)
        self._queue.put_nowait(sample * scale)


# ---------------------------------------------------------------------------
# AudioPlayer — callback-based OutputStream with monitor mix support
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Click-bar generator
# ---------------------------------------------------------------------------

def generate_bar_click(
    bpm: float,
    samplerate: int = 44100,
    n_channels: int = 2,
    n_beats: int = 4,
    beats_per_bar: int = 4,
) -> np.ndarray:
    """
    Generate a *n_beats*-beat click track at *bpm*.

    Beat 1 of every bar (every *beats_per_bar* beats) is accented with a
    higher-pitched, louder click — correct for both 1-bar count-ins and
    full-song metronome tracks.

    Returns:
        float32 array of shape [samples, n_channels].
    """
    beat_s   = 60.0 / bpm
    bar_s    = n_beats * beat_s
    n_samp   = int(bar_s * samplerate)
    audio    = np.zeros((n_samp, n_channels), dtype=np.float32)

    click_len = int(0.012 * samplerate)          # 12 ms click body
    t = np.linspace(0, 0.012, click_len, endpoint=False)

    for beat in range(n_beats):
        accent = (beat % beats_per_bar) == 0     # downbeat of each bar
        freq = 1500.0 if accent else 900.0
        vol  = 0.90  if accent else 0.60
        click = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 500) * vol).astype(np.float32)

        start = int(beat * beat_s * samplerate)
        end   = min(start + click_len, n_samp)
        n     = end - start
        for ch in range(n_channels):
            audio[start:end, ch] = click[:n]

    return audio


class AudioPlayer:
    def __init__(self, device: int | str | None = None) -> None:
        self.device = device
        self._stream_start_time: float = 0.0
        self._stream: sd.OutputStream | None = None
        self._done_event = threading.Event()
        # SimpleQueue is safe to write from the MIDI thread and read from the
        # audio callback thread without explicit locking.
        self._monitor_queue: queue.SimpleQueue = queue.SimpleQueue()

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> list[dict]:
        devs = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                devs.append(
                    {
                        "index": i,
                        "name": d["name"],
                        "channels": d["max_output_channels"],
                        "default_sr": int(d["default_samplerate"]),
                        "hostapi": sd.query_hostapis(d["hostapi"])["name"],
                    }
                )
        return devs

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play(
        self,
        audio: Path | tuple[np.ndarray, int],
        latency: str | float = "low",
        monitor: DrumMonitor | None = None,
    ) -> float:
        """
        Play audio via a callback OutputStream.

        Args:
            audio: Either a Path to an audio file, or a (data, samplerate)
                   tuple where data is float32 shape [samples, channels].
            latency: 'low', 'high', or seconds.
            monitor: Optional DrumMonitor to mix in real time.

        Returns:
            perf_counter() timestamp immediately before the stream starts.
        """
        if isinstance(audio, tuple):
            data, samplerate = audio
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 1:
                data = data[:, np.newaxis]
        else:
            data, samplerate = sf.read(str(audio), dtype="float32", always_2d=True)
        n_frames, n_ch = data.shape

        if monitor is not None:
            monitor.attach(self)

        # Flush any leftover monitor sounds from a previous session
        while not self._monitor_queue.empty():
            try:
                self._monitor_queue.get_nowait()
            except Exception:
                break

        self._done_event.clear()
        pos = [0]                           # mutable cell captured by closure
        active_monitor: list[list] = []    # [[sample_array, current_offset], ...]

        def _callback(outdata: np.ndarray, frames: int, _time, _status) -> None:
            # ── Backing track ────────────────────────────────────────
            start = pos[0]
            end   = start + frames
            chunk = data[start:end]
            got   = len(chunk)

            if got < frames:
                outdata[:got]  = chunk
                outdata[got:]  = 0.0
                pos[0] = n_frames
                raise sd.CallbackStop()
            else:
                outdata[:] = chunk
                pos[0] = end

            # ── Monitor hits ─────────────────────────────────────────
            while True:
                try:
                    sample = self._monitor_queue.get_nowait()
                    active_monitor.append([sample, 0])
                except queue.Empty:
                    break

            surviving: list[list] = []
            for item in active_monitor:
                s_data, s_pos = item
                s_end  = s_pos + frames
                chunk_s = s_data[s_pos:s_end]
                n_s = len(chunk_s)
                if n_s:
                    # Mono sample → broadcast to all channels
                    outdata[:n_s] += chunk_s[:, np.newaxis]
                if s_end < len(s_data):
                    item[1] = s_end
                    surviving.append(item)
            active_monitor[:] = surviving

            # Soft clip to avoid distortion when monitor + backing overlap
            np.clip(outdata, -1.0, 1.0, out=outdata)

        self._stream = sd.OutputStream(
            samplerate=samplerate,
            channels=n_ch,
            dtype="float32",
            callback=_callback,
            finished_callback=self._done_event.set,
            device=self.device,
            latency=latency,
        )

        self._stream_start_time = time.perf_counter()
        self._stream.start()
        return self._stream_start_time

    def wait(self) -> None:
        """Block until playback completes (polls so Ctrl+C stays responsive)."""
        while not self._done_event.wait(timeout=0.25):
            pass
        if self._stream:
            self._stream.close()
            self._stream = None

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._done_event.set()

    @property
    def stream_start_time(self) -> float:
        return self._stream_start_time


# ---------------------------------------------------------------------------
# LatencyCalibrator (unchanged)
# ---------------------------------------------------------------------------

class LatencyCalibrator:
    """
    Measures audio output latency so the scorer can compensate.

    Procedure:
      1. Play a sharp click (kick drum transient).
      2. User hits their kick pad in sync with each click.
      3. We measure the systematic offset between reference click times
         and the received MIDI timestamps.
      4. The mean offset is the latency to subtract during scoring.
    """

    CLICK_INTERVAL_S = 1.0
    NUM_CLICKS = 8

    def __init__(self, player: AudioPlayer) -> None:
        self.player = player

    def generate_click_track(self, samplerate: int = 44100) -> np.ndarray:
        duration = self.NUM_CLICKS * self.CLICK_INTERVAL_S + 1.0
        total_samples = int(duration * samplerate)
        track = np.zeros((total_samples, 2), dtype="float32")

        click_len = int(0.01 * samplerate)
        t = np.linspace(0, 0.01, click_len)
        click = (np.sin(2 * np.pi * 1000 * t) * np.hanning(click_len)).astype(
            "float32"
        )

        for i in range(self.NUM_CLICKS):
            start = int(i * self.CLICK_INTERVAL_S * samplerate)
            end = start + click_len
            track[start:end, 0] = click
            track[start:end, 1] = click

        return track

    def reference_times(self) -> list[float]:
        return [i * self.CLICK_INTERVAL_S for i in range(self.NUM_CLICKS)]

    def compute_latency(self, midi_events: list) -> float:
        if not midi_events:
            return 0.0

        refs = self.reference_times()
        hit_times = [e.timestamp for e in midi_events]

        offsets = []
        used: set[int] = set()
        for ref in refs:
            best_i = min(
                (i for i in range(len(hit_times)) if i not in used),
                key=lambda i: abs(hit_times[i] - ref),
                default=None,
            )
            if best_i is not None and abs(hit_times[best_i] - ref) < 0.3:
                offsets.append((hit_times[best_i] - ref) * 1000)
                used.add(best_i)

        return float(np.median(offsets)) if offsets else 0.0
