"""
Session scoring engine.

Compares the player's MIDI hits against the reference onsets produced by
analyzer.py, compensates for measured audio/MIDI latency, and produces a
structured report with per-hit grades and per-instrument summaries.

Grading windows (symmetric around the reference time):
    Perfect  ±30 ms   — nailed it
    Good     ±75 ms   — solid
    OK       ±150 ms  — acceptable
    Miss     > 150 ms — not matched

Ghost notes: MIDI hits that don't correspond to any reference onset (player
added a hit that wasn't in the original). Counted separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from core.analyzer import Onset
from core.midi_listener import MidiEvent, GM_DRUM_MAP

# ---------------------------------------------------------------------------
# Grade thresholds (milliseconds, absolute value of offset)
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "perfect": 30.0,
    "good":    75.0,
    "ok":     150.0,
}

TOLERANCE_MS = THRESHOLDS["ok"]   # hits beyond this are considered misses

Grade = Literal["perfect", "good", "ok", "miss"]


# ---------------------------------------------------------------------------
# MIDI deduplication
# ---------------------------------------------------------------------------

def _debounce_midi_events(
    events: list[MidiEvent],
    window_ms: float = 15.0,
) -> list[MidiEvent]:
    """
    Collapse retrigger duplicates — multiple note-on events fired by the same
    pad within *window_ms* of each other.  15 ms is well below the ±30 ms
    'perfect' window but wide enough to catch typical pad retriggering.

    Events are deduplicated per note; simultaneous hits on different notes
    (e.g. kick + hihat) are never suppressed.
    """
    if not events:
        return events

    by_note: dict[int, list[MidiEvent]] = {}
    for ev in events:
        by_note.setdefault(ev.note, []).append(ev)

    kept: list[MidiEvent] = []
    for note_events in by_note.values():
        note_events.sort(key=lambda e: e.timestamp)
        last_kept_t = -1.0
        for ev in note_events:
            if (ev.timestamp - last_kept_t) * 1000.0 >= window_ms:
                kept.append(ev)
                last_kept_t = ev.timestamp

    return sorted(kept, key=lambda e: e.timestamp)


# ---------------------------------------------------------------------------
# Per-hit result
# ---------------------------------------------------------------------------

@dataclass
class HitResult:
    """Outcome for a single reference onset."""
    reference_time: float        # expected time in seconds
    reference_strength: float    # onset strength 0–1 (for future dynamics scoring)
    hit_time: float | None       # player's actual hit time in seconds (None = miss)
    offset_ms: float | None      # hit_time - reference_time in ms (+ve = late, –ve = early)
    note: int | None             # MIDI note number (None = miss)
    drum_name: str | None        # human-readable drum name
    grade: Grade                 # "perfect" | "good" | "ok" | "miss"


@dataclass
class GhostNote:
    """A MIDI hit with no matching reference onset."""
    hit_time: float
    note: int
    velocity: int
    drum_name: str


# ---------------------------------------------------------------------------
# Per-instrument summary
# ---------------------------------------------------------------------------

@dataclass
class InstrumentStats:
    note: int
    drum_name: str
    total_reference_hits: int = 0
    perfect: int = 0
    good: int = 0
    ok: int = 0
    miss: int = 0
    ghost: int = 0
    offsets_ms: list[float] = field(default_factory=list)

    @property
    def hit_count(self) -> int:
        return self.perfect + self.good + self.ok

    @property
    def accuracy_pct(self) -> float:
        if self.total_reference_hits == 0:
            return 0.0
        return 100.0 * self.hit_count / self.total_reference_hits

    @property
    def avg_offset_ms(self) -> float:
        return float(np.mean(self.offsets_ms)) if self.offsets_ms else 0.0

    @property
    def avg_abs_offset_ms(self) -> float:
        return float(np.mean(np.abs(self.offsets_ms))) if self.offsets_ms else 0.0


# ---------------------------------------------------------------------------
# Top-level score report
# ---------------------------------------------------------------------------

@dataclass
class ScoreReport:
    hits: list[HitResult]
    ghost_notes: list[GhostNote]
    latency_compensation_ms: float   # value subtracted from all MIDI timestamps

    # Aggregate counts (reference onsets perspective)
    total_reference_hits: int = 0
    perfect_count: int = 0
    good_count: int = 0
    ok_count: int = 0
    miss_count: int = 0
    ghost_count: int = 0

    # Per-instrument breakdown keyed by MIDI note number
    by_instrument: dict[int, InstrumentStats] = field(default_factory=dict)

    @property
    def hit_count(self) -> int:
        return self.perfect_count + self.good_count + self.ok_count

    @property
    def accuracy_pct(self) -> float:
        if self.total_reference_hits == 0:
            return 0.0
        return 100.0 * self.hit_count / self.total_reference_hits

    @property
    def avg_offset_ms(self) -> float:
        """Mean signed offset across all matched hits (positive = consistently late)."""
        offsets = [h.offset_ms for h in self.hits if h.offset_ms is not None]
        return float(np.mean(offsets)) if offsets else 0.0

    @property
    def avg_abs_offset_ms(self) -> float:
        """Mean absolute timing error across all matched hits."""
        offsets = [h.offset_ms for h in self.hits if h.offset_ms is not None]
        return float(np.mean(np.abs(offsets))) if offsets else 0.0

    def summary(self) -> str:
        """Return a human-readable one-page summary."""
        lines = [
            "=" * 52,
            "  DRUM SESSION SCORE",
            "=" * 52,
            f"  Accuracy      : {self.accuracy_pct:5.1f}%  "
            f"({self.hit_count}/{self.total_reference_hits} hits)",
            f"  Perfect (±30ms): {self.perfect_count:4d}",
            f"  Good    (±75ms): {self.good_count:4d}",
            f"  OK     (±150ms): {self.ok_count:4d}",
            f"  Misses         : {self.miss_count:4d}",
            f"  Ghost notes    : {self.ghost_count:4d}",
            f"  Avg offset     : {self.avg_offset_ms:+.1f} ms  "
            f"(+ve = late, –ve = early)",
            f"  Avg abs error  : {self.avg_abs_offset_ms:.1f} ms",
            f"  Latency comp.  : {self.latency_compensation_ms:+.1f} ms",
            "-" * 52,
            "  BY INSTRUMENT",
            "-" * 52,
        ]
        for stats in sorted(
            self.by_instrument.values(), key=lambda s: -s.total_reference_hits
        ):
            lines.append(
                f"  {stats.drum_name:<22s} "
                f"{stats.accuracy_pct:5.1f}%  "
                f"P:{stats.perfect:3d} G:{stats.good:3d} "
                f"O:{stats.ok:3d} M:{stats.miss:3d} "
                f"ghost:{stats.ghost:2d}  "
                f"avg {stats.avg_offset_ms:+.0f}ms"
            )
        lines.append("=" * 52)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _grade(abs_offset_ms: float) -> Grade:
    if abs_offset_ms <= THRESHOLDS["perfect"]:
        return "perfect"
    if abs_offset_ms <= THRESHOLDS["good"]:
        return "good"
    if abs_offset_ms <= THRESHOLDS["ok"]:
        return "ok"
    return "miss"


def score_session(
    onsets: list[Onset],
    midi_events: list[MidiEvent],
    latency_ms: float = 0.0,
) -> ScoreReport:
    """
    Score a practice session.

    Args:
        onsets:      Reference drum onsets from analyzer.detect_onsets().
        midi_events: Captured MIDI events from MidiListener.stop_recording().
        latency_ms:  Measured latency from LatencyCalibrator.compute_latency().
                     Subtracted from every MIDI timestamp before matching.

    Returns:
        A ScoreReport with per-hit results and aggregate statistics.
    """
    import bisect

    # Debounce first — collapses pad retriggering before matching
    midi_events = _debounce_midi_events(midi_events)

    # Apply latency compensation — shift all MIDI hits earlier by latency_ms
    latency_s = latency_ms / 1000.0
    compensated: list[tuple[float, MidiEvent]] = sorted(
        ((e.timestamp - latency_s, e) for e in midi_events),
        key=lambda x: x[0],
    )

    hit_times = [t for t, _ in compensated]   # sorted list for bisect lookups
    used: set[int] = set()                     # indices into compensated that are matched

    hit_results: list[HitResult] = []
    ghost_notes: list[GhostNote] = []

    # --- O(n log m) matching: for each reference onset find the nearest
    #     unused MIDI hit via binary search, then check neighbours.     ----
    tolerance_s = TOLERANCE_MS / 1000.0

    for onset in onsets:
        ref_t = onset["time"]

        # Binary search gives the insertion point for ref_t
        pos = bisect.bisect_left(hit_times, ref_t)

        best_idx: int | None = None
        best_diff = float("inf")

        # Check a small window of candidates around the insertion point
        for idx in (pos - 1, pos, pos + 1):
            if idx < 0 or idx >= len(hit_times) or idx in used:
                continue
            diff = abs(hit_times[idx] - ref_t)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

        if best_idx is not None and best_diff <= tolerance_s:
            used.add(best_idx)
            hit_t, event = compensated[best_idx]
            offset_ms = (hit_t - ref_t) * 1000.0
            grade = _grade(abs(offset_ms))
            hit_results.append(
                HitResult(
                    reference_time=ref_t,
                    reference_strength=onset["strength"],
                    hit_time=hit_t,
                    offset_ms=offset_ms,
                    note=event.note,
                    drum_name=event.drum_name,
                    grade=grade,
                )
            )
        else:
            hit_results.append(
                HitResult(
                    reference_time=ref_t,
                    reference_strength=onset["strength"],
                    hit_time=None,
                    offset_ms=None,
                    note=None,
                    drum_name=None,
                    grade="miss",
                )
            )

    # --- Remaining unmatched MIDI hits are ghost notes ----------------------
    for i, (hit_t, event) in enumerate(compensated):
        if i not in used:
            ghost_notes.append(
                GhostNote(
                    hit_time=hit_t,
                    note=event.note,
                    velocity=event.velocity,
                    drum_name=event.drum_name,
                )
            )

    # --- Build aggregate counts --------------------------------------------
    report = ScoreReport(
        hits=hit_results,
        ghost_notes=ghost_notes,
        latency_compensation_ms=latency_ms,
        total_reference_hits=len(onsets),
        perfect_count=sum(1 for h in hit_results if h.grade == "perfect"),
        good_count=sum(1 for h in hit_results if h.grade == "good"),
        ok_count=sum(1 for h in hit_results if h.grade == "ok"),
        miss_count=sum(1 for h in hit_results if h.grade == "miss"),
        ghost_count=len(ghost_notes),
    )

    # --- Per-instrument breakdown ------------------------------------------
    # Gather all note numbers present in reference misses (no note) and hits
    for hit in hit_results:
        if hit.note is not None:
            note = hit.note
            if note not in report.by_instrument:
                report.by_instrument[note] = InstrumentStats(
                    note=note,
                    drum_name=GM_DRUM_MAP.get(note, f"Note {note}"),
                )
            stats = report.by_instrument[note]
            stats.total_reference_hits += 1
            setattr(stats, hit.grade, getattr(stats, hit.grade) + 1)
            if hit.offset_ms is not None:
                stats.offsets_ms.append(hit.offset_ms)
        else:
            # Miss with no note — we don't know which instrument was expected,
            # so we track it under a synthetic "unknown" key (-1)
            if -1 not in report.by_instrument:
                report.by_instrument[-1] = InstrumentStats(
                    note=-1, drum_name="(missed / unknown)"
                )
            report.by_instrument[-1].total_reference_hits += 1
            report.by_instrument[-1].miss += 1

    for ghost in ghost_notes:
        note = ghost.note
        if note not in report.by_instrument:
            report.by_instrument[note] = InstrumentStats(
                note=note,
                drum_name=GM_DRUM_MAP.get(note, f"Note {note}"),
            )
        report.by_instrument[note].ghost += 1

    return report
