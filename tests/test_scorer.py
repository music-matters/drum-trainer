"""
Unit tests for core/scorer.py.

Run with:
    python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.scorer import (
    _grade,
    _debounce_midi_events,
    score_session,
    THRESHOLDS,
    TOLERANCE_MS,
)
from core.midi_listener import MidiEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(timestamp: float, note: int = 36, velocity: int = 100) -> MidiEvent:
    return MidiEvent(timestamp=timestamp, note=note, velocity=velocity, channel=9)


def make_onset(time: float, strength: float = 0.8) -> dict:
    return {"time": time, "strength": strength}


# ---------------------------------------------------------------------------
# _grade — boundary conditions
# ---------------------------------------------------------------------------

class TestGrade:
    def test_perfect_at_zero(self):
        assert _grade(0.0) == "perfect"

    def test_perfect_at_threshold(self):
        assert _grade(THRESHOLDS["perfect"]) == "perfect"

    def test_good_just_above_perfect(self):
        assert _grade(THRESHOLDS["perfect"] + 0.1) == "good"

    def test_good_at_threshold(self):
        assert _grade(THRESHOLDS["good"]) == "good"

    def test_ok_just_above_good(self):
        assert _grade(THRESHOLDS["good"] + 0.1) == "ok"

    def test_ok_at_threshold(self):
        assert _grade(THRESHOLDS["ok"]) == "ok"

    def test_miss_just_above_ok(self):
        assert _grade(THRESHOLDS["ok"] + 0.1) == "miss"

    def test_miss_large_offset(self):
        assert _grade(500.0) == "miss"


# ---------------------------------------------------------------------------
# _debounce_midi_events
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_empty_input(self):
        assert _debounce_midi_events([]) == []

    def test_single_event_kept(self):
        events = [make_event(1.0)]
        assert len(_debounce_midi_events(events)) == 1

    def test_duplicate_within_window_collapsed(self):
        # Two hits on the same note 5 ms apart — only first kept
        events = [make_event(1.000), make_event(1.005)]
        result = _debounce_midi_events(events)
        assert len(result) == 1
        assert result[0].timestamp == pytest.approx(1.000)

    def test_hits_outside_window_both_kept(self):
        # 20 ms apart — both kept
        events = [make_event(1.000), make_event(1.020)]
        result = _debounce_midi_events(events)
        assert len(result) == 2

    def test_exactly_at_window_boundary_collapsed(self):
        # Exactly 15 ms apart — collapses in practice because floating-point
        # subtraction yields 14.999...ms which is just under the window.
        # This documents the actual behaviour rather than the ideal boundary.
        events = [make_event(1.000), make_event(1.015)]
        result = _debounce_midi_events(events)
        assert len(result) == 1

    def test_just_above_window_boundary_kept(self):
        # 16 ms — clearly outside the 15 ms window, both events kept
        events = [make_event(1.000), make_event(1.016)]
        result = _debounce_midi_events(events)
        assert len(result) == 2

    def test_different_notes_not_collapsed(self):
        # Same timestamp, different notes — both kept
        events = [make_event(1.000, note=36), make_event(1.000, note=38)]
        result = _debounce_midi_events(events)
        assert len(result) == 2

    def test_three_rapid_hits_first_kept(self):
        # Three hits in quick succession — only first kept
        events = [make_event(1.000), make_event(1.005), make_event(1.010)]
        result = _debounce_midi_events(events)
        assert len(result) == 1

    def test_output_is_sorted_by_timestamp(self):
        # Mix of notes, output should be time-sorted
        events = [
            make_event(2.0, note=36),
            make_event(1.0, note=38),
            make_event(3.0, note=36),
        ]
        result = _debounce_midi_events(events)
        timestamps = [e.timestamp for e in result]
        assert timestamps == sorted(timestamps)

    def test_third_hit_after_gap_kept(self):
        # Hit at 0, 5ms (collapsed), 25ms (new window — kept)
        events = [make_event(0.000), make_event(0.005), make_event(0.025)]
        result = _debounce_midi_events(events)
        assert len(result) == 2
        assert result[0].timestamp == pytest.approx(0.000)
        assert result[1].timestamp == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# score_session — core scoring
# ---------------------------------------------------------------------------

class TestScoreSession:

    # ── Empty inputs ─────────────────────────────────────────────────────────

    def test_no_onsets_no_events(self):
        report = score_session([], [])
        assert report.total_reference_hits == 0
        assert report.accuracy_pct == 0.0
        assert report.ghost_count == 0

    def test_no_onsets_with_events_all_ghost(self):
        events = [make_event(1.0), make_event(2.0)]
        report = score_session([], events)
        assert report.ghost_count == 2
        assert report.total_reference_hits == 0

    def test_onsets_no_events_all_miss(self):
        onsets = [make_onset(1.0), make_onset(2.0)]
        report = score_session(onsets, [])
        assert report.miss_count == 2
        assert report.accuracy_pct == 0.0

    # ── Grading ──────────────────────────────────────────────────────────────

    def test_perfect_hit(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.000)]   # exact
        report = score_session(onsets, events)
        assert report.perfect_count == 1
        assert report.good_count == 0
        assert report.accuracy_pct == pytest.approx(100.0)

    def test_perfect_hit_slightly_early(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.0 - 0.025)]  # 25 ms early — perfect
        report = score_session(onsets, events)
        assert report.perfect_count == 1

    def test_good_hit(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.0 + 0.050)]  # 50 ms late — good
        report = score_session(onsets, events)
        assert report.good_count == 1

    def test_ok_hit(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.0 + 0.120)]  # 120 ms late — ok
        report = score_session(onsets, events)
        assert report.ok_count == 1

    def test_miss_too_far(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.0 + 0.200)]  # 200 ms late — miss (ghost)
        report = score_session(onsets, events)
        assert report.miss_count == 1
        assert report.ghost_count == 1

    def test_accuracy_pct_calculation(self):
        onsets = [make_onset(1.0), make_onset(2.0), make_onset(3.0), make_onset(4.0)]
        events = [
            make_event(1.000),   # perfect
            make_event(2.050),   # good
            make_event(3.120),   # ok
            # 4.0 — miss
        ]
        report = score_session(onsets, events)
        assert report.total_reference_hits == 4
        assert report.hit_count == 3
        assert report.accuracy_pct == pytest.approx(75.0)

    # ── Ghost notes ───────────────────────────────────────────────────────────

    def test_extra_hits_are_ghosts(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.000), make_event(5.000)]  # second has no onset near it
        report = score_session(onsets, events)
        assert report.perfect_count == 1
        assert report.ghost_count == 1

    def test_all_ghosts_no_onsets_matched(self):
        onsets = [make_onset(10.0)]   # onset far from any event
        events = [make_event(1.0), make_event(2.0)]
        report = score_session(onsets, events)
        assert report.miss_count == 1
        assert report.ghost_count == 2

    # ── Latency compensation ──────────────────────────────────────────────────

    def test_latency_compensation_shifts_hits_earlier(self):
        # Onset at 1.0 s; player hits at 1.100 s (100 ms late).
        # With 100 ms latency compensation the hit should become perfect.
        onsets = [make_onset(1.0)]
        events = [make_event(1.100)]
        report = score_session(onsets, events, latency_ms=100.0)
        assert report.perfect_count == 1
        assert report.latency_compensation_ms == pytest.approx(100.0)

    def test_no_latency_same_hit_is_ok(self):
        # Same hit without compensation should be ok (100 ms offset)
        onsets = [make_onset(1.0)]
        events = [make_event(1.100)]
        report = score_session(onsets, events, latency_ms=0.0)
        assert report.ok_count == 1

    def test_negative_latency_shifts_later(self):
        # Negative latency (player hits early) — shift events later
        onsets = [make_onset(1.0)]
        events = [make_event(0.950)]   # 50 ms early
        report = score_session(onsets, events, latency_ms=-50.0)
        # After compensation: 0.950 - (-0.050) = 1.000 → perfect
        assert report.perfect_count == 1

    # ── Offset direction ─────────────────────────────────────────────────────

    def test_late_hit_positive_offset(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.050)]   # 50 ms late
        report = score_session(onsets, events)
        assert report.hits[0].offset_ms == pytest.approx(50.0, abs=0.1)

    def test_early_hit_negative_offset(self):
        onsets = [make_onset(1.0)]
        events = [make_event(0.970)]   # 30 ms early
        report = score_session(onsets, events)
        assert report.hits[0].offset_ms == pytest.approx(-30.0, abs=0.1)

    def test_avg_offset_sign(self):
        # Consistently late player should have positive avg offset
        onsets = [make_onset(1.0), make_onset(2.0)]
        events = [make_event(1.050), make_event(2.050)]
        report = score_session(onsets, events)
        assert report.avg_offset_ms > 0

    # ── Matching logic ────────────────────────────────────────────────────────

    def test_each_onset_matched_at_most_once(self):
        # Two events both within tolerance of the same onset (100 ms apart so
        # debounce keeps both), only the nearer one should claim the onset.
        onsets = [make_onset(1.0)]
        events = [make_event(1.000), make_event(1.100)]   # 100 ms apart → survive debounce
        report = score_session(onsets, events)
        assert report.hit_count == 1      # only one onset to claim
        assert report.ghost_count == 1    # second event becomes a ghost

    def test_each_event_matched_at_most_once(self):
        # One event, two nearby onsets — event claims the nearest onset
        onsets = [make_onset(1.000), make_onset(1.050)]
        events = [make_event(1.000)]
        report = score_session(onsets, events)
        assert report.hit_count == 1
        assert report.miss_count == 1

    def test_multiple_onsets_and_events_matched_correctly(self):
        onsets = [make_onset(1.0), make_onset(2.0), make_onset(3.0)]
        events = [make_event(1.010), make_event(2.020), make_event(3.030)]
        report = score_session(onsets, events)
        assert report.perfect_count == 3
        assert report.miss_count == 0
        assert report.ghost_count == 0

    # ── Per-instrument breakdown ──────────────────────────────────────────────

    def test_by_instrument_populated(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.000, note=36)]
        report = score_session(onsets, events)
        assert 36 in report.by_instrument
        stats = report.by_instrument[36]
        assert stats.perfect == 1
        assert stats.total_reference_hits == 1

    def test_ghost_tracked_in_by_instrument(self):
        onsets = []
        events = [make_event(1.0, note=38)]
        report = score_session(onsets, events)
        assert 38 in report.by_instrument
        assert report.by_instrument[38].ghost == 1

    def test_miss_tracked_under_unknown(self):
        onsets = [make_onset(1.0)]
        events = []
        report = score_session(onsets, events)
        # Misses with no MIDI note go under key -1
        assert -1 in report.by_instrument
        assert report.by_instrument[-1].miss == 1

    def test_instrument_accuracy_pct(self):
        onsets = [make_onset(1.0), make_onset(2.0)]
        events = [make_event(1.000, note=36)]   # hits first, misses second
        report = score_session(onsets, events)
        stats = report.by_instrument[36]
        assert stats.total_reference_hits == 1
        assert stats.accuracy_pct == pytest.approx(100.0)

    def test_instrument_avg_offset(self):
        onsets = [make_onset(1.0), make_onset(2.0)]
        events = [make_event(1.040, note=36), make_event(2.060, note=36)]
        report = score_session(onsets, events)
        stats = report.by_instrument[36]
        assert stats.avg_offset_ms == pytest.approx(50.0, abs=0.5)

    # ── Debounce integration ──────────────────────────────────────────────────

    def test_rapid_retrigger_collapsed_before_scoring(self):
        # Pad fires twice in 5 ms — should count as one hit, not a ghost
        onsets = [make_onset(1.0)]
        events = [make_event(1.000), make_event(1.005)]
        report = score_session(onsets, events)
        assert report.hit_count == 1
        assert report.ghost_count == 0

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_single_perfect_session(self):
        onsets = [make_onset(t) for t in [1.0, 2.0, 3.0, 4.0]]
        events = [make_event(t) for t in [1.0, 2.0, 3.0, 4.0]]
        report = score_session(onsets, events)
        assert report.accuracy_pct == pytest.approx(100.0)
        assert report.ghost_count == 0
        assert report.miss_count == 0

    def test_all_misses_zero_accuracy(self):
        onsets = [make_onset(1.0), make_onset(2.0)]
        events = [make_event(10.0), make_event(20.0)]  # way off
        report = score_session(onsets, events)
        assert report.accuracy_pct == pytest.approx(0.0)
        assert report.miss_count == 2
        assert report.ghost_count == 2

    def test_tolerance_boundary_inside_is_ok(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.0 + TOLERANCE_MS / 1000.0)]  # exactly at boundary
        report = score_session(onsets, events)
        assert report.ok_count == 1

    def test_tolerance_boundary_outside_is_miss(self):
        onsets = [make_onset(1.0)]
        events = [make_event(1.0 + TOLERANCE_MS / 1000.0 + 0.001)]  # just past boundary
        report = score_session(onsets, events)
        assert report.miss_count == 1
        assert report.ghost_count == 1
