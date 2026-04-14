"""
drum-trainer — CLI entry point.

Commands
--------
  play <url>     Download a song, separate stems, and play along.
  calibrate      Measure audio/MIDI round-trip latency.
  devices        List available audio output and MIDI input devices.

Examples
--------
  python main.py devices
  python main.py calibrate
  python main.py play "https://www.youtube.com/watch?v=..."
  python main.py play "https://..." --midi-port 1 --audio-device 3
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_step(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def _progress_hook(d: dict) -> None:
    """yt-dlp progress callback — prints a simple one-line status."""
    if d["status"] == "downloading":
        pct = d.get("_percent_str", "?%").strip()
        speed = d.get("_speed_str", "").strip()
        print(f"\r    downloading: {pct}  {speed}      ", end="", flush=True)
    elif d["status"] == "finished":
        print(f"\r    download complete.                          ")


# ---------------------------------------------------------------------------
# 'devices' command
# ---------------------------------------------------------------------------

def cmd_devices(_args: argparse.Namespace) -> None:
    from core.player import AudioPlayer
    from core.midi_listener import MidiListener

    print("\n--- Audio output devices ---")
    for dev in AudioPlayer.list_devices():
        marker = " (default)" if dev["index"] == 0 else ""
        print(
            f"  [{dev['index']:2d}]  {dev['name']:<40s}  "
            f"{dev['channels']}ch  {dev['default_sr']}Hz  "
            f"{dev['hostapi']}{marker}"
        )

    print("\n--- MIDI input devices ---")
    listener = MidiListener()
    ports = listener.list_ports()
    if ports:
        for i, name in enumerate(ports):
            print(f"  [{i}]  {name}")
    else:
        print("  (none found)")


# ---------------------------------------------------------------------------
# 'calibrate' command
# ---------------------------------------------------------------------------

def cmd_calibrate(args: argparse.Namespace) -> None:
    from core.player import AudioPlayer, LatencyCalibrator
    from core.midi_listener import MidiListener

    player = AudioPlayer(device=args.audio_device)
    calibrator = LatencyCalibrator(player)
    listener = MidiListener()

    # Open MIDI port
    ports = listener.list_ports()
    if not ports:
        print("ERROR: No MIDI input devices found. Connect your drum kit and retry.")
        sys.exit(1)

    port_idx = args.midi_port
    if port_idx >= len(ports):
        print(f"ERROR: MIDI port {port_idx} not found. Run 'devices' to list ports.")
        sys.exit(1)

    print(f"\nUsing MIDI port: {ports[port_idx]}")
    listener.open_port(port_idx)

    try:
        _print_step(
            f"Calibration — {calibrator.NUM_CLICKS} clicks, "
            f"one per second.\n"
            f"    Hit your KICK pad in time with each click."
        )
        input("    Press Enter when ready...")

        click_track = calibrator.generate_click_track()
        import sounddevice as sd

        start_time = time.perf_counter()
        sd.play(click_track, 44100, device=player.device, latency="low")
        listener.start_recording(start_time)
        sd.wait()
        events = listener.stop_recording()

        if not events:
            print("\nNo MIDI hits detected — check your connection and try again.")
            return

        latency_ms = calibrator.compute_latency(events)
        print(f"\n  Detected latency: {latency_ms:+.1f} ms")
        print(f"  ({len(events)} hits recorded, {calibrator.NUM_CLICKS} expected)")

        # Save to a config file for use in 'play'
        config_path = Path("latency.json")
        config_path.write_text(json.dumps({"latency_ms": latency_ms}))
        print(f"  Saved to {config_path} — will be applied automatically during 'play'.")

    finally:
        listener.close()


# ---------------------------------------------------------------------------
# 'play' command
# ---------------------------------------------------------------------------

def cmd_play(args: argparse.Namespace) -> None:
    from core.downloader import download_audio
    from core.separator import separate_stems
    from core.analyzer import detect_onsets, save_onsets, load_onsets
    from core.player import AudioPlayer
    from core.midi_listener import MidiListener
    from core.scorer import score_session

    # --- Load latency config -----------------------------------------------
    latency_ms = 0.0
    latency_path = Path("latency.json")
    if latency_path.exists():
        latency_ms = json.loads(latency_path.read_text()).get("latency_ms", 0.0)
        print(f"  Latency compensation: {latency_ms:+.1f} ms  (from {latency_path})")
    else:
        print(
            "  No latency.json found — run 'calibrate' first for best results.\n"
            "  Proceeding without latency compensation."
        )

    # --- MIDI setup --------------------------------------------------------
    listener = MidiListener()
    ports = listener.list_ports()
    if not ports:
        print("ERROR: No MIDI input devices found. Connect your drum kit and retry.")
        sys.exit(1)

    port_idx = args.midi_port
    if port_idx >= len(ports):
        print(f"ERROR: MIDI port {port_idx} not found. Run 'devices' to list ports.")
        sys.exit(1)

    print(f"  MIDI input: {ports[port_idx]}")
    listener.open_port(port_idx)

    player = AudioPlayer(device=args.audio_device)

    try:
        data_dir = Path(args.data_dir)

        # --- Step 1: Download ----------------------------------------------
        _print_step("Downloading audio...")
        wav_path = download_audio(args.url, data_dir / "downloads", _progress_hook)
        print(f"  Saved: {wav_path.name}")

        # --- Step 2: Stem separation ---------------------------------------
        _print_step("Separating stems (this may take a few minutes)...")
        stems = separate_stems(
            wav_path,
            data_dir / "stems",
            model=args.model,
        )
        print(f"  Stems: {', '.join(stems.keys())}")

        # --- Step 3: Onset analysis ----------------------------------------
        onsets_path = data_dir / "onsets" / f"{wav_path.stem}.json"

        if onsets_path.exists() and not args.reanalyze:
            _print_step("Loading cached onset analysis...")
            onsets = load_onsets(onsets_path)
        else:
            _print_step("Analysing drum onsets...")
            onsets = detect_onsets(stems["drums"])
            save_onsets(onsets, onsets_path)

        print(f"  Found {len(onsets)} reference hits.")

        # --- Step 4: Countdown + play + record -----------------------------
        _print_step("Get ready to play!")
        print(f"  Song: {wav_path.stem}")
        print(f"  Backing track: no drums (you are the drummer).")
        print()

        for i in range(3, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1.0)
        print("  GO!\n", flush=True)

        start_time = player.play(stems["no_drums"])
        listener.start_recording(start_time)
        player.wait()
        midi_events = listener.stop_recording()

        print(f"  Playback finished. Captured {len(midi_events)} MIDI events.")

        # --- Step 5: Score -------------------------------------------------
        _print_step("Scoring...")
        report = score_session(onsets, midi_events, latency_ms=latency_ms)
        print()
        print(report.summary())

        # Optionally save the raw report
        if args.save_report:
            report_path = data_dir / "reports" / f"{wav_path.stem}_{int(time.time())}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _save_report_json(report, report_path)
            print(f"\n  Full report saved to: {report_path}")

    finally:
        listener.close()


def _save_report_json(report, path: Path) -> None:
    """Serialise a ScoreReport to JSON."""
    import dataclasses

    def _default(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    data = {
        "latency_compensation_ms": report.latency_compensation_ms,
        "accuracy_pct": report.accuracy_pct,
        "total_reference_hits": report.total_reference_hits,
        "perfect": report.perfect_count,
        "good": report.good_count,
        "ok": report.ok_count,
        "miss": report.miss_count,
        "ghost": report.ghost_count,
        "avg_offset_ms": report.avg_offset_ms,
        "avg_abs_offset_ms": report.avg_abs_offset_ms,
        "hits": [
            {
                "reference_time": h.reference_time,
                "hit_time": h.hit_time,
                "offset_ms": h.offset_ms,
                "note": h.note,
                "drum_name": h.drum_name,
                "grade": h.grade,
            }
            for h in report.hits
        ],
        "ghost_notes": [
            {
                "hit_time": g.hit_time,
                "note": g.note,
                "velocity": g.velocity,
                "drum_name": g.drum_name,
            }
            for g in report.ghost_notes
        ],
    }
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drum-trainer",
        description="Play along to any YouTube song and score your drum accuracy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- devices ------------------------------------------------------------
    sub.add_parser("devices", help="List available audio and MIDI devices.")

    # -- calibrate ----------------------------------------------------------
    cal = sub.add_parser("calibrate", help="Measure audio/MIDI round-trip latency.")
    cal.add_argument(
        "--audio-device", type=int, default=None,
        metavar="N",
        help="Audio output device index (default: system default).",
    )
    cal.add_argument(
        "--midi-port", type=int, default=0,
        metavar="N",
        help="MIDI input port index (default: 0).",
    )

    # -- play ---------------------------------------------------------------
    play = sub.add_parser("play", help="Download a song and play along.")
    play.add_argument("url", help="YouTube URL of the song.")
    play.add_argument(
        "--audio-device", type=int, default=None,
        metavar="N",
        help="Audio output device index (default: system default).",
    )
    play.add_argument(
        "--midi-port", type=int, default=0,
        metavar="N",
        help="MIDI input port index (default: 0).",
    )
    play.add_argument(
        "--model", default="htdemucs",
        metavar="MODEL",
        help="Demucs model (htdemucs | htdemucs_ft | mdx_extra). Default: htdemucs.",
    )
    play.add_argument(
        "--data-dir", default="data",
        metavar="DIR",
        help="Directory for downloads, stems, and analysis cache. Default: data/.",
    )
    play.add_argument(
        "--reanalyze", action="store_true",
        help="Re-run onset analysis even if a cached result exists.",
    )
    play.add_argument(
        "--save-report", action="store_true",
        help="Save a full JSON report to data/reports/.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "devices":   cmd_devices,
        "calibrate": cmd_calibrate,
        "play":      cmd_play,
    }
    commands[args.command](args)
