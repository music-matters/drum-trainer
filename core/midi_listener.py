"""
MIDI input capture using python-rtmidi.

Captures note-on events from an electronic drum kit (MIDI over USB) and
timestamps each hit relative to a reference start time provided by the caller.

The caller (player.py / session runner) sets the start time at the exact
moment audio playback begins, so all timestamps are in the same time space
as the reference onsets from the analyzer.

General MIDI drum note map is included for display / per-instrument scoring.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import rtmidi

# General MIDI percussion note map (channel 10, notes 35-81)
GM_DRUM_MAP: dict[int, str] = {
    35: "Kick (acoustic)",
    36: "Kick",
    37: "Snare (cross-stick)",
    38: "Snare",
    39: "Snare (clap)",
    40: "Snare (electric)",
    41: "Hi-hat (low floor)",
    42: "Hi-hat (closed)",
    43: "Hi-hat (high floor)",
    44: "Hi-hat (pedal)",
    45: "Tom (low)",
    46: "Hi-hat (open)",
    47: "Tom (low-mid)",
    48: "Tom (high-mid)",
    49: "Crash 1",
    50: "Tom (high)",
    51: "Ride (bow)",
    52: "China",
    53: "Ride (bell)",
    54: "Tambourine",
    55: "Splash",
    56: "Cowbell",
    57: "Crash 2",
    58: "Vibraslap",
    59: "Ride 2",
    60: "Bongo (high)",
    61: "Bongo (low)",
}


@dataclass
class MidiEvent:
    timestamp: float   # seconds from session start (perf_counter based)
    note: int          # MIDI note number
    velocity: int      # 0–127
    channel: int       # 0-indexed (drum kits usually use channel 9 = GM ch 10)

    @property
    def drum_name(self) -> str:
        return GM_DRUM_MAP.get(self.note, f"Note {self.note}")


class MidiListener:
    """
    Wraps an rtmidi input port and records note-on events.

    Usage:
        listener = MidiListener()
        listener.open_port(0)
        listener.start_recording(start_time)   # start_time = perf_counter()
        # ... audio plays ...
        events = listener.stop_recording()
        listener.close()
    """

    def __init__(self) -> None:
        self._midi_in = rtmidi.MidiIn()
        self._events: list[MidiEvent] = []
        self._start_time: float = 0.0
        self._recording: bool = False
        self._port_open: bool = False
        self._monitor = None   # optional DrumMonitor

    def set_monitor(self, monitor) -> None:
        """Attach a DrumMonitor so hits are synthesised in real time."""
        self._monitor = monitor

    # ------------------------------------------------------------------
    # Port management
    # ------------------------------------------------------------------

    def list_ports(self) -> list[str]:
        return self._midi_in.get_ports()

    def open_port(self, port_index: int = 0) -> None:
        ports = self.list_ports()
        if not ports:
            raise RuntimeError("No MIDI input devices found.")
        if port_index >= len(ports):
            raise ValueError(
                f"Port index {port_index} out of range ({len(ports)} available)."
            )
        self._midi_in.open_port(port_index)
        # Ignore clock and sysex; keep note-on/off
        self._midi_in.ignore_types(sysex=True, timing=True, active_sense=True)
        self._midi_in.set_callback(self._on_midi_event)
        self._port_open = True

    def close(self) -> None:
        if self._port_open:
            self._midi_in.close_port()
            self._port_open = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self, start_time: float | None = None) -> None:
        """
        Begin capturing events.

        Args:
            start_time: perf_counter() value representing t=0 of the session.
                        If None, uses current time (i.e. starts now).
        """
        self._events.clear()
        self._start_time = start_time if start_time is not None else time.perf_counter()
        self._recording = True

    def stop_recording(self) -> list[MidiEvent]:
        self._recording = False
        return list(self._events)

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

    def _on_midi_event(self, event_data, user_data=None) -> None:
        if not self._recording:
            return

        message, _delta = event_data
        if len(message) < 3:
            return

        status = message[0]
        note = message[1]
        velocity = message[2]
        channel = status & 0x0F

        # Note-on with velocity > 0
        is_note_on = (status & 0xF0) == 0x90 and velocity > 0
        # Some kits send note-off as note-on with velocity=0; ignore those
        if not is_note_on:
            return

        timestamp = time.perf_counter() - self._start_time
        self._events.append(
            MidiEvent(
                timestamp=timestamp,
                note=note,
                velocity=velocity,
                channel=channel,
            )
        )

        # Fire monitor sound immediately (before scoring overhead)
        if self._monitor is not None:
            try:
                self._monitor.trigger(note, velocity)
            except Exception:
                pass  # never let monitor errors disrupt recording
