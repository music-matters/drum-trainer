"""
Flask web interface for drum-trainer.

Run with:
    python api/app.py

Then open http://127.0.0.1:5000 in your browser.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import queue
import threading
import time
from pathlib import Path
import os

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from core.rate_limiter import RateLimiter

_ROOT         = Path(__file__).parent.parent
_LIBRARY_PATH = _ROOT / "data" / "library.json"

app = Flask(
    __name__,
    template_folder=str(_ROOT / "ui" / "web" / "templates"),
    static_folder=str(_ROOT / "ui" / "web" / "static"),
)

_rate_limiter = RateLimiter(_ROOT / "data" / "rate_limits.json")

# ---------------------------------------------------------------------------
# Single-session state (local single-user tool)
# ---------------------------------------------------------------------------

_is_running    = False
_stop_requested = False
_current_player = None          # AudioPlayer instance during active session
_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Library helpers
# ---------------------------------------------------------------------------

def _extract_youtube_id(url: str) -> str | None:
    from urllib.parse import urlparse, parse_qs
    try:
        p = urlparse(url)
        if "youtu.be" in p.netloc:
            return p.path.lstrip("/").split("?")[0]
        if "youtube.com" in p.netloc:
            return parse_qs(p.query).get("v", [None])[0]
    except Exception:
        pass
    return None


def _library_load() -> list[dict]:
    if not _LIBRARY_PATH.exists():
        return []
    try:
        return json.loads(_LIBRARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _library_save(lib: list[dict]) -> None:
    _LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LIBRARY_PATH.write_text(json.dumps(lib, indent=2), encoding="utf-8")


def _library_upsert(url: str, title: str) -> None:
    """Add or update a song entry in the library."""
    yt_id = _extract_youtube_id(url)
    if not yt_id:
        return
    try:
        from datetime import datetime, timezone
        lib = _library_load()
        entry = next((e for e in lib if e.get("youtube_id") == yt_id), None)
        if entry:
            entry["title"] = title
        else:
            lib.insert(0, {
                "youtube_id": yt_id,
                "url": url,
                "title": title,
                "added_at": datetime.now(timezone.utc).isoformat(),
                "sessions": [],
            })
        _library_save(lib)
    except Exception:
        pass


def _library_record_session(url: str, report: dict) -> None:
    yt_id = _extract_youtube_id(url)
    if not yt_id:
        return
    try:
        from datetime import datetime, timezone
        lib = _library_load()
        entry = next((e for e in lib if e.get("youtube_id") == yt_id), None)
        if entry is None:
            return
        entry.setdefault("sessions", []).append({
            "played_at": datetime.now(timezone.utc).isoformat(),
            "score_pct": round(report.get("accuracy_pct", 0), 1),
            "report": report,
        })
        _library_save(lib)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Drum grouping for per-instrument results
# ---------------------------------------------------------------------------

_DEFAULT_DRUM_GROUPS: list[tuple[str, set[int]]] = [
    ("Kick",       {35, 36}),
    ("Snare",      {38, 39, 40}),
    ("Hi-hat",     {42, 44, 46}),
    ("Tom (Low)",  {41, 43, 45}),
    ("Tom (Mid)",  {47, 48}),
    ("Tom (High)", {50}),
    ("Crash",      {49, 52, 55, 57}),
    ("Ride",       {51, 53, 59}),
]

_KIT_KEY_LABELS: dict[str, str] = {
    "kick": "Kick", "snare": "Snare",
    "hihat_closed": "Hi-hat", "hihat_open": "Hi-hat",
    "tom_low": "Tom (Low)", "tom_mid": "Tom (Mid)", "tom_high": "Tom (High)",
    "crash": "Crash", "ride": "Ride",
}


def _get_drum_groups() -> list[tuple[str, set[int]]]:
    """Load grouping from kit_config.json; fall back to GM defaults."""
    config_path = _ROOT / "data" / "kit_config.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text())
            groups: dict[str, set[int]] = {}
            for key, notes in raw.items():
                label = _KIT_KEY_LABELS.get(key, key.replace("_", " ").title())
                groups.setdefault(label, set()).update(int(n) for n in notes)
            return list(groups.items())
        except Exception:
            pass
    return _DEFAULT_DRUM_GROUPS


def _group_instruments(by_instrument: dict) -> list[dict]:
    """Merge per-note stats into named drum families, sorted by hit count."""
    result = []
    for group_name, notes in _get_drum_groups():
        members = [s for note, s in by_instrument.items() if note in notes]
        if not members:
            continue
        total    = sum(m.total_reference_hits for m in members)
        perfect  = sum(m.perfect  for m in members)
        good     = sum(m.good     for m in members)
        ok       = sum(m.ok       for m in members)
        miss     = sum(m.miss     for m in members)
        ghost    = sum(m.ghost    for m in members)
        hit_count = perfect + good + ok
        accuracy_pct = round(100.0 * hit_count / total, 1) if total > 0 else 0.0
        weighted = sum(m.avg_offset_ms * (m.perfect + m.good + m.ok) for m in members)
        avg_offset = round(weighted / hit_count, 1) if hit_count else 0.0
        result.append({
            "drum_name":    group_name,
            "total":        total,
            "perfect":      perfect,
            "good":         good,
            "ok":           ok,
            "miss":         miss,
            "ghost":        ghost,
            "accuracy_pct": accuracy_pct,
            "avg_offset_ms": avg_offset,
        })
    return sorted(result, key=lambda x: x["total"], reverse=True)


def _broadcast(event: dict) -> None:
    with _subscribers_lock:
        for q in _subscribers:
            q.put(event)


def _subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue()
    with _subscribers_lock:
        _subscribers.append(q)
    return q


def _unsubscribe(q: queue.Queue) -> None:
    with _subscribers_lock:
        if q in _subscribers:
            _subscribers.remove(q)


# ---------------------------------------------------------------------------
# Background pipeline: play
# ---------------------------------------------------------------------------

def _run_play(
    url: str,
    midi_port: int,
    audio_device: int | None,
    model: str,
    monitor_volume: float = 0.0,
    metronome_volume: float = 0.0,
    backing_volume: float = 1.0,
    client_id: str = "unknown",
) -> None:
    global _is_running, _stop_requested, _current_player
    try:
        import numpy as np
        from core.downloader import download_audio
        from core.separator import separate_stems
        from core.analyzer import detect_onsets, save_onsets, load_onsets, detect_tempo
        from core.player import AudioPlayer, DrumMonitor, generate_bar_click
        from core.midi_listener import MidiListener
        from core.scorer import score_session

        data = _ROOT / "data"

        latency_ms = 0.0
        latency_path = _ROOT / "latency.json"
        if latency_path.exists():
            latency_ms = json.loads(latency_path.read_text()).get("latency_ms", 0.0)

        # Check rate limit before attempting download
        dev_mode = os.environ.get("FLASK_ENV") == "development"
        allowed, quota_msg = _rate_limiter.check_quota(client_id, dev_mode=dev_mode)

        if not allowed:
            _broadcast({"type": "error", "message": quota_msg})
            return

        listener = MidiListener()
        ports = listener.list_ports()
        if not ports:
            _broadcast({"type": "error", "message": "No MIDI input devices found."})
            return
        if midi_port >= len(ports):
            _broadcast({"type": "error", "message": f"MIDI port {midi_port} not found."})
            return
        listener.open_port(midi_port)

        player = AudioPlayer(device=audio_device)
        _current_player = player
        _stop_requested = False

        # Set up built-in drum monitor if requested
        monitor = None
        if monitor_volume > 0:
            monitor = DrumMonitor(volume=monitor_volume)
            listener.set_monitor(monitor)

        try:
            # Silently check MIDI is live — 2 s window, no prompt to user.
            # We only warn if nothing arrives; songs with late drum entries are fine
            # because this is purely a "is the port alive?" connection test.
            listener.start_recording()
            time.sleep(2.0)
            warmup = listener.stop_recording()
            if not warmup:
                _broadcast({"type": "warning",
                            "message": "No MIDI hits detected on the selected port — "
                                       "make sure your kit is powered on and the correct "
                                       "MIDI Input is selected. Continuing…"})

            # Step 1: download
            _broadcast({"type": "step", "step": "download", "status": "running",
                        "message": "Downloading audio..."})
            wav_path, song_title, was_cached = download_audio(url, data / "downloads")
            if not was_cached:
                _rate_limiter.record_download(client_id)
            _library_upsert(url, song_title)
            _broadcast({"type": "step", "step": "download", "status": "done",
                        "message": f"{song_title}  {'(cached)' if was_cached else ''}".strip()})

            # Step 2: separate
            _broadcast({"type": "step", "step": "separate", "status": "running",
                        "message": "Separating stems — this may take a few minutes..."})
            stems = separate_stems(wav_path, data / "stems", model=model)
            _broadcast({"type": "step", "step": "separate", "status": "done",
                        "message": "Stems ready."})

            # Step 3: analyze onsets + detect tempo
            onsets_path = data / "onsets" / f"{wav_path.stem}.json"
            if onsets_path.exists():
                _broadcast({"type": "step", "step": "analyze", "status": "running",
                            "message": "Loading cached onset analysis..."})
                onsets = load_onsets(onsets_path)
            else:
                _broadcast({"type": "step", "step": "analyze", "status": "running",
                            "message": "Analysing drum onsets..."})
                onsets = detect_onsets(stems["drums"])
                save_onsets(onsets, onsets_path)

            bpm = detect_tempo(stems["drums"])
            _broadcast({"type": "step", "step": "analyze", "status": "done",
                        "message": (
                            f"{len(onsets)} drum hits mapped  •  ♩ = {bpm:.0f} BPM"
                        )})

            # Countdown (3-2-1 visual)
            for i in range(3, 0, -1):
                _broadcast({"type": "countdown", "count": i})
                time.sleep(1.0)
            _broadcast({"type": "countdown", "count": 0})  # GO

            # Build 1-bar click + backing track as a single audio array
            import soundfile as _sf
            backing_data, backing_sr = _sf.read(
                str(stems["no_drums"]), dtype="float32", always_2d=True
            )
            click_bar = generate_bar_click(bpm, samplerate=backing_sr,
                                           n_channels=backing_data.shape[1])
            click_dur = len(click_bar) / backing_sr

            # Optional full-song metronome — generate a click the length of the
            # song and mix it into the backing track before playback.
            if metronome_volume > 0:
                n_song = len(backing_data)
                n_ch   = backing_data.shape[1]
                n_beats_total = int(np.ceil(n_song / backing_sr * bpm / 60)) + 2
                metro = generate_bar_click(
                    bpm, samplerate=backing_sr,
                    n_channels=n_ch, n_beats=n_beats_total,
                )
                metro = metro[:n_song]  # trim to exact song length
                backing_data = np.clip(
                    backing_data + metro * metronome_volume, -1.0, 1.0
                ).astype(np.float32)

            if backing_volume != 1.0:
                backing_data = (backing_data * backing_volume).astype(np.float32)

            combined = np.concatenate([click_bar, backing_data], axis=0)

            # Broadcast count-in so the UI can show the beat counter
            beat_interval = 60.0 / bpm
            _broadcast({"type": "countin", "bpm": round(bpm, 1), "beats": 4})

            start_time = player.play((combined, backing_sr), monitor=monitor)

            # Fire beat events timed to the click, then start MIDI recording
            # precisely when the backing track begins.
            for beat in range(4):
                beat_fire = start_time + beat * beat_interval
                sleep_s   = max(0.0, beat_fire - time.perf_counter() - 0.005)
                time.sleep(sleep_s)
                while time.perf_counter() < beat_fire:
                    pass
                _broadcast({"type": "beat", "beat": beat + 1})

            # Busy-wait for the exact song-start moment
            song_start = start_time + click_dur
            while time.perf_counter() < song_start:
                pass
            listener.start_recording(song_start)

            # Step 4: playing
            _broadcast({"type": "step", "step": "play", "status": "running",
                        "message": "Playing — you're the drummer!"})
            _broadcast({"type": "playback_started"})

            # Poll until done or stopped
            while not player._done_event.wait(timeout=0.25):
                if _stop_requested:
                    player.stop()
                    break
            midi_events = listener.stop_recording()
            raw_hits = listener._raw_hit_count

            if _stop_requested:
                _broadcast({"type": "stopped"})
                return

            if raw_hits == 0:
                _broadcast({"type": "warning",
                            "message": "No MIDI hits detected — is your kit selected as the MIDI Input?"})

            _broadcast({"type": "step", "step": "play", "status": "done",
                        "message": (
                            f"Done — {len(midi_events)} hits from your kit "
                            f"({raw_hits} total received) "
                            f"vs {len(onsets)} in the song."
                        )})

            # Score
            report = score_session(onsets, midi_events, latency_ms=latency_ms)
            report_dict = _report_to_dict(report, song_title)
            _library_record_session(url, report_dict)
            _broadcast({"type": "result", "report": report_dict})

        finally:
            listener.close()

    except Exception as exc:
        _broadcast({"type": "error", "message": str(exc)})
    finally:
        _is_running    = False
        _stop_requested = False
        _current_player = None
        _broadcast({"type": "done"})


# ---------------------------------------------------------------------------
# Background pipeline: calibrate
# ---------------------------------------------------------------------------

def _run_calibrate(midi_port: int, audio_device: int | None) -> None:
    global _is_running
    try:
        import sounddevice as sd
        from core.player import AudioPlayer, LatencyCalibrator
        from core.midi_listener import MidiListener

        player = AudioPlayer(device=audio_device)
        calibrator = LatencyCalibrator(player)
        listener = MidiListener()

        ports = listener.list_ports()
        if not ports:
            _broadcast({"type": "error", "message": "No MIDI input devices found."})
            return
        if midi_port >= len(ports):
            _broadcast({"type": "error", "message": f"MIDI port {midi_port} not found."})
            return
        listener.open_port(midi_port)

        try:
            _broadcast({
                "type": "cal_info",
                "message": (
                    f"You'll hear {calibrator.NUM_CLICKS} clicks, one per second. "
                    "Hit your kick pad in time with each one."
                ),
            })
            time.sleep(2.0)

            click_track = calibrator.generate_click_track()
            start_time = time.perf_counter()
            sd.play(click_track, 44100, device=player.device, latency="low")
            listener.start_recording(start_time)
            sd.wait()
            events = listener.stop_recording()

            if not events:
                _broadcast({"type": "error",
                            "message": "No MIDI hits detected — check your connection and try again."})
                return

            latency_ms = calibrator.compute_latency(events)
            latency_path = _ROOT / "latency.json"
            latency_path.write_text(json.dumps({"latency_ms": latency_ms}))

            _broadcast({
                "type": "cal_result",
                "latency_ms": round(latency_ms, 1),
                "hits_recorded": len(events),
                "hits_expected": calibrator.NUM_CLICKS,
            })
        finally:
            listener.close()

    except Exception as exc:
        _broadcast({"type": "error", "message": str(exc)})
    finally:
        _is_running = False
        _broadcast({"type": "done"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report_to_dict(report, song_title: str = "") -> dict:
    return {
        "song_title":           song_title,
        "accuracy_pct":         round(report.accuracy_pct, 1),
        "total_reference_hits": report.total_reference_hits,
        "perfect":              report.perfect_count,
        "good":                 report.good_count,
        "ok":                   report.ok_count,
        "miss":                 report.miss_count,
        "ghost":                report.ghost_count,
        "avg_offset_ms":        round(report.avg_offset_ms, 1),
        "avg_abs_offset_ms":    round(report.avg_abs_offset_ms, 1),
        "latency_ms":           round(report.latency_compensation_ms, 1),
        "by_instrument":        _group_instruments(report.by_instrument),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/vsts")
def api_vsts():
    """Detect installed standalone drum VSTs by checking common install paths."""
    import os
    candidates = {
        "EZ Drummer 3": [
            r"C:\Program Files\Toontrack\EZdrummer 3\EZdrummer 3.exe",
        ],
        "EZ Drummer 2": [
            r"C:\Program Files\Toontrack\EZdrummer 2\EZdrummer 2.exe",
        ],
        "Superior Drummer 3": [
            r"C:\Program Files\Toontrack\Superior Drummer 3\Superior Drummer 3.exe",
        ],
        "BFD3": [
            r"C:\Program Files\BFD Drums\BFD3\BFD3.exe",
            r"C:\Program Files\inMusic Brands\BFD3\BFD3.exe",
            r"C:\Program Files (x86)\BFD Drums\BFD3\BFD3.exe",
        ],
        "Addictive Drums 2": [
            r"C:\Program Files\XLN Audio\Addictive Drums 2\Addictive Drums 2.exe",
        ],
        "Addictive Drums 3": [
            r"C:\Program Files\XLN Audio\Addictive Drums 3\Addictive Drums 3.exe",
        ],
    }
    found = []
    for name, paths in candidates.items():
        for p in paths:
            if os.path.isfile(p):
                found.append({"name": name, "path": p})
                break
    return jsonify({"vsts": found})


@app.get("/api/devices")
def api_devices():
    from core.player import AudioPlayer
    from core.midi_listener import MidiListener
    audio = AudioPlayer.list_devices()
    midi_ports = MidiListener().list_ports()
    return jsonify({
        "audio": audio,
        "midi": [{"index": i, "name": n} for i, n in enumerate(midi_ports)],
    })


@app.post("/api/stop")
def api_stop():
    global _stop_requested
    _stop_requested = True
    if _current_player is not None:
        _current_player.stop()
    return jsonify({"status": "stopping"})


@app.get("/api/library")
def api_library():
    lib = _library_load()
    # Enrich each entry with thumbnail URL and best/play-count summaries.
    result = []
    for entry in lib:
        yt_id    = entry.get("youtube_id", "")
        sessions = entry.get("sessions", [])
        scores   = [s["score_pct"] for s in sessions if "score_pct" in s]
        result.append({
            "youtube_id": yt_id,
            "url":        entry.get("url", f"https://www.youtube.com/watch?v={yt_id}"),
            "title":      entry.get("title", "Unknown"),
            "thumbnail":  f"https://img.youtube.com/vi/{yt_id}/mqdefault.jpg",
            "play_count": len(sessions),
            "best_score": max(scores) if scores else None,
            "last_played": sessions[-1]["played_at"] if sessions else None,
        })
    return jsonify(result)


@app.get("/api/library/<youtube_id>/sessions")
def api_library_sessions(youtube_id: str):
    lib = _library_load()
    entry = next((e for e in lib if e.get("youtube_id") == youtube_id), None)
    if entry is None:
        return jsonify([])
    return jsonify(list(reversed(entry.get("sessions", []))))  # newest first


@app.delete("/api/library/<youtube_id>")
def api_library_delete(youtube_id: str):
    lib = _library_load()
    lib = [e for e in lib if e.get("youtube_id") != youtube_id]
    _library_save(lib)
    return jsonify({"status": "ok"})


@app.post("/api/play")
def api_play():
    global _is_running
    if _is_running:
        return jsonify({"error": "A session is already running."}), 409

    body = request.get_json(force=True) or {}
    url = body.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    # Extract client IP for rate limiting (X-Forwarded-For for proxy, remote_addr for direct)
    client_id = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    if "," in client_id:
        client_id = client_id.split(",")[0].strip()

    _is_running = True
    threading.Thread(
        target=_run_play,
        kwargs={
            "url": url,
            "midi_port": int(body.get("midi_port", 0)),
            "audio_device": int(body["audio_device"]) if body.get("audio_device") not in (None, "", "null") else None,
            "model": body.get("model", "htdemucs"),
            "monitor_volume": float(body.get("monitor_volume", 0.0)),
            "metronome_volume": float(body.get("metronome_volume", 0.0)),
            "backing_volume": float(body.get("backing_volume", 1.0)),
            "client_id": client_id,
        },
        daemon=True,
    ).start()
    return jsonify({"status": "started"})


@app.post("/api/calibrate")
def api_calibrate():
    global _is_running
    if _is_running:
        return jsonify({"error": "A session is already running."}), 409

    body = request.get_json(force=True) or {}
    _is_running = True
    threading.Thread(
        target=_run_calibrate,
        kwargs={
            "midi_port": int(body.get("midi_port", 0)),
            "audio_device": int(body["audio_device"]) if body.get("audio_device") not in (None, "", "null") else None,
        },
        daemon=True,
    ).start()
    return jsonify({"status": "started"})


@app.get("/api/stream")
def api_stream():
    q = _subscribe()

    def generate():
        try:
            while True:
                try:
                    event = q.get(timeout=25)
                    try:
                        data = json.dumps(event)
                    except (TypeError, ValueError) as exc:
                        # Serialisation failure — send an error so the client
                        # isn't left waiting forever on a broken stream.
                        data = json.dumps({"type": "error",
                                           "message": f"Result encoding error: {exc}"})
                    yield f"data: {data}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
                except queue.Empty:
                    yield 'data: {"type":"ping"}\n\n'
        finally:
            _unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# New-user session flag  (set by --new-user CLI flag)
# ---------------------------------------------------------------------------

_new_user_session = False   # consumed on first page load


@app.get("/api/new-user-session")
def api_new_user_session():
    """
    Returns {reset: true} once if the server was started with --new-user,
    then clears the flag so subsequent reloads behave normally.
    """
    global _new_user_session
    fired = _new_user_session
    _new_user_session = False
    return jsonify({"reset": fired})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drum Trainer web server")
    parser.add_argument(
        "--new-user",
        action="store_true",
        help="Start as a first-time user: stashes library.json and triggers the setup wizard",
    )
    args = parser.parse_args()

    if args.new_user:
        _new_user_session = True
        # Stash existing library so it isn't lost
        if _LIBRARY_PATH.exists():
            from datetime import datetime
            stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = _LIBRARY_PATH.with_name(f"library_backup_{stamp}.json")
            _LIBRARY_PATH.rename(backup)
            print(f"  Library stashed → {backup.name}")
        print("  New-user mode: wizard will run on first page load")

    print("Starting drum-trainer at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
