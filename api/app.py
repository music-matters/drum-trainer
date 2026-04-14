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

_ROOT = Path(__file__).parent.parent

app = Flask(
    __name__,
    template_folder=str(_ROOT / "ui" / "web" / "templates"),
    static_folder=str(_ROOT / "ui" / "web" / "static"),
)

_rate_limiter = RateLimiter(_ROOT / "data" / "rate_limits.json")

# ---------------------------------------------------------------------------
# Single-session state (local single-user tool)
# ---------------------------------------------------------------------------

_is_running = False
_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()


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
    client_id: str = "unknown",
) -> None:
    global _is_running
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

        # Set up built-in drum monitor if requested
        monitor = None
        if monitor_volume > 0:
            monitor = DrumMonitor(volume=monitor_volume)
            listener.set_monitor(monitor)

        try:
            # Step 1: download
            _broadcast({"type": "step", "step": "download", "status": "running",
                        "message": "Downloading audio..."})
            wav_path, song_title = download_audio(url, data / "downloads")
            _rate_limiter.record_download(client_id)  # Record successful download
            _broadcast({"type": "step", "step": "download", "status": "done",
                        "message": song_title})

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
            player.wait()
            midi_events = listener.stop_recording()
            _broadcast({"type": "step", "step": "play", "status": "done",
                        "message": (
                            f"Done — {len(midi_events)} hits from your kit "
                            f"vs {len(onsets)} in the song."
                        )})

            # Score
            report = score_session(onsets, midi_events, latency_ms=latency_ms)
            _broadcast({"type": "result", "report": _report_to_dict(report)})

        finally:
            listener.close()

    except Exception as exc:
        _broadcast({"type": "error", "message": str(exc)})
    finally:
        _is_running = False
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

def _report_to_dict(report) -> dict:
    return {
        "accuracy_pct": round(report.accuracy_pct, 1),
        "total_reference_hits": report.total_reference_hits,
        "perfect": report.perfect_count,
        "good": report.good_count,
        "ok": report.ok_count,
        "miss": report.miss_count,
        "ghost": report.ghost_count,
        "avg_offset_ms": round(report.avg_offset_ms, 1),
        "avg_abs_offset_ms": round(report.avg_abs_offset_ms, 1),
        "latency_ms": round(report.latency_compensation_ms, 1),
        "by_instrument": {
            str(note): {
                "drum_name": s.drum_name,
                "total": s.total_reference_hits,
                "perfect": s.perfect,
                "good": s.good,
                "ok": s.ok,
                "miss": s.miss,
                "ghost": s.ghost,
                "accuracy_pct": round(s.accuracy_pct, 1),
                "avg_offset_ms": round(s.avg_offset_ms, 1),
            }
            for note, s in report.by_instrument.items()
            if note != -1
        },
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
            "audio_device": body.get("audio_device") or None,
            "model": body.get("model", "htdemucs"),
            "monitor_volume": float(body.get("monitor_volume", 0.0)),
            "metronome_volume": float(body.get("metronome_volume", 0.0)),
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
            "audio_device": body.get("audio_device") or None,
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting drum-trainer at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
