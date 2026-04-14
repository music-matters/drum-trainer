"""
YouTube audio downloader using yt-dlp.
Outputs a high-quality WAV file ready for stem separation.

Files are keyed by video ID (e.g. "dQw4w9WgXcQ.wav") so:
  - Re-running the same song skips the download entirely.
  - No filename collisions that cause yt-dlp's "Deleting old file" hang on Windows.

Returns (wav_path, song_title) so callers can display the real song name.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

import yt_dlp


def sanitize_filename(name: str) -> str:
    """Strip characters that are unsafe in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def strip_playlist_params(url: str) -> str:
    """Remove playlist/index query parameters so yt-dlp sees a single video URL."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    for key in ("list", "index", "start_radio"):
        params.pop(key, None)
    clean_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=clean_query))


def _video_id(url: str) -> str | None:
    """Extract the YouTube video ID from a URL, or None if not found."""
    parsed = urlparse(url)
    # Standard watch URL: ?v=...
    vid = parse_qs(parsed.query).get("v", [None])[0]
    if vid:
        return vid
    # Short URL: youtu.be/<id>
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/") or None
    return None


def download_audio(url: str, output_dir: Path, progress_hook=None) -> tuple[Path, str]:
    """
    Download audio from a YouTube URL and convert to WAV.

    Strategy to avoid Windows "Deleting old file" hang:
      1. Download into a unique temp name (_tmp_<uuid>) so there is never
         an existing file for yt-dlp to collide with or overwrite.
      2. keepvideo=True tells FFmpegExtractAudio NOT to delete the source
         audio container (.webm / .m4a) after WAV conversion — that deletion
         is what stalls on Windows when an AV scanner briefly locks the file.
      3. We rename the temp WAV to the final name and clean up temp files
         ourselves with a silent try/except, so a transient lock never blocks.

    Returns:
        (wav_path, song_title) — the path to the WAV and the human-readable title.
    """
    url = strip_playlist_params(url)
    output_dir.mkdir(parents=True, exist_ok=True)

    vid = _video_id(url)

    # Return cached WAV immediately; read the saved title sidecar if present.
    if vid:
        cached = output_dir / f"{vid}.wav"
        if cached.exists():
            title_file = output_dir / f"{vid}.title"
            title = (
                title_file.read_text(encoding="utf-8").strip()
                if title_file.exists()
                else vid
            )
            return cached, title, True   # (path, title, was_cached)

    # Use a unique temp stem — guarantees no old-file collision.
    tmp_stem = f"_tmp_{uuid.uuid4().hex}"
    out_template = str(output_dir / f"{tmp_stem}.%(ext)s")

    hooks = [progress_hook] if progress_hook else []

    ydl_opts: dict = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "progress_hooks": hooks,
        "postprocessor_args": ["-ar", "44100"],
        # keepvideo=True: do NOT delete the source container after conversion.
        # overwrites=True: pass -y to FFmpeg so it never prompts for confirmation
        # (without this FFmpeg hangs waiting for stdin on Windows).
        "keepvideo": True,
        "overwrites": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title: str = info.get("title") or vid or "unknown"

    tmp_wav = output_dir / f"{tmp_stem}.wav"
    if not tmp_wav.exists():
        candidates = sorted(output_dir.glob(f"{tmp_stem}*.wav"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"WAV not found after download in {output_dir}")
        tmp_wav = candidates[-1]

    # Rename to final stable name.
    if vid:
        final_path = output_dir / f"{vid}.wav"
    else:
        final_path = output_dir / f"{sanitize_filename(title)}.wav"

    # Windows rename raises FileExistsError if the destination already exists.
    if final_path.exists():
        final_path.unlink()
    tmp_wav.rename(final_path)

    # Persist title so cache hits can return the real name next time.
    if vid:
        try:
            (output_dir / f"{vid}.title").write_text(title, encoding="utf-8")
        except OSError:
            pass

    # Best-effort cleanup of the kept intermediate container (.webm / .m4a).
    for tmp_file in output_dir.glob(f"{tmp_stem}.*"):
        try:
            tmp_file.unlink()
        except OSError:
            pass  # locked by AV scanner — the OS will clean it up eventually

    return final_path, title, False   # (path, title, was_cached)


def get_video_info(url: str) -> dict:
    """Fetch video metadata without downloading."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "thumbnail": info.get("thumbnail"),
    }
