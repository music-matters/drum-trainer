"""
Drum onset detection using librosa.

Analyses the isolated drum stem and returns a list of timestamped hits.
These become the 'ground truth' the player is scored against.

A note on accuracy:
  Demucs separation is not perfect — bleed from other instruments can
  create false onsets. The parameters below are tuned conservatively
  (higher delta threshold, short wait) to prefer precision over recall.
  You can lower `delta` to catch more hits at the cost of more false
  positives in noisy separations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np


class Onset(TypedDict):
    time: float      # seconds from start of track
    strength: float  # normalised 0–1 (useful later for dynamics scoring)


def detect_onsets(
    drums_file: Path,
    delta: float = 0.12,
    wait_frames: int = 2,
) -> list[Onset]:
    """
    Detect drum hit onsets in *drums_file*.

    Args:
        drums_file:   Path to the demucs 'drums.wav' stem.
        delta:        Onset detection threshold (higher = fewer, stricter hits).
        wait_frames:  Minimum gap between onsets (in frames ~= 23ms @ 512 hop).

    Returns:
        List of Onset dicts sorted by time.
    """
    y, sr = librosa.load(str(drums_file), sr=None, mono=True)

    # Per-channel energy onset envelope — better for drums than spectral flux
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=512,
        aggregate=np.median,  # more robust than mean for percussive signals
    )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        delta=delta,
        wait=wait_frames,
        # Use local averaging so adaptive threshold follows song dynamics
        pre_avg=3,
        post_avg=3,
        pre_max=3,
        post_max=3,
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    strengths = onset_env[onset_frames]

    # Normalise strengths to 0–1 range
    if strengths.size > 0:
        s_min, s_max = strengths.min(), strengths.max()
        strengths = (strengths - s_min) / (s_max - s_min + 1e-9)

    onsets: list[Onset] = [
        {"time": float(t), "strength": float(s)}
        for t, s in zip(onset_times, strengths)
    ]

    return sorted(onsets, key=lambda o: o["time"])


def save_onsets(onsets: list[Onset], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(onsets, f, indent=2)


def load_onsets(path: Path) -> list[Onset]:
    with open(path) as f:
        return json.load(f)


def detect_tempo(drums_file: Path) -> float:
    """
    Estimate the song's tempo in BPM from the drum stem.

    Returns a float BPM value (e.g. 120.0).  Uses librosa's beat tracker
    which is robust on isolated drum stems with minimal bleed.
    """
    y, sr = librosa.load(str(drums_file), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa ≥ 0.10 may return a 1-element array
    return float(np.atleast_1d(tempo)[0])


def onset_stats(onsets: list[Onset]) -> dict:
    """Quick summary stats — useful for debugging separation quality."""
    if not onsets:
        return {}
    times = [o["time"] for o in onsets]
    gaps = np.diff(times)
    return {
        "count": len(onsets),
        "duration_s": times[-1] - times[0],
        "avg_gap_ms": float(np.mean(gaps) * 1000) if gaps.size else 0,
        "min_gap_ms": float(np.min(gaps) * 1000) if gaps.size else 0,
        "avg_strength": float(np.mean([o["strength"] for o in onsets])),
    }
