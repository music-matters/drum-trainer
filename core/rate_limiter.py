"""
Download rate limiter — per-IP tracking.

Tracks downloads per client (IP address for now) with a rolling 7-day window.
Limit: 5 downloads per week.

Later, when user accounts are implemented, swap client_id from IP → user_id
from the auth token. The storage format and logic remain identical.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


class RateLimiter:
    """
    Enforce download quotas to prevent YouTube throttling and API abuse.

    Storage format (data/rate_limits.json):
        {
          "203.0.113.42": {
            "downloads": ["2026-04-14T10:30:00", "2026-04-13T15:22:00", ...],
            "limit": 5
          },
          ...
        }
    """

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or (
            Path(__file__).parent.parent / "data" / "rate_limits.json"
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = 5
        self.window_days = 7

    def check_quota(self, client_id: str, dev_mode: bool = False) -> tuple[bool, str]:
        """
        Check if client can download.

        Args:
            client_id:  Unique identifier (IP address for now, user_id later).
            dev_mode:   If True, bypass limit entirely.

        Returns:
            (allowed, message) — tuple of bool and reason string.
        """
        if dev_mode:
            return True, "Dev mode: unlimited downloads"

        data = self._load()
        now = datetime.now()

        if client_id not in data:
            data[client_id] = {"downloads": [], "limit": self.limit}

        client_data = data[client_id]
        downloads = client_data.get("downloads", [])

        # Prune downloads outside the rolling window
        cutoff = (now - timedelta(days=self.window_days)).isoformat()
        downloads = [d for d in downloads if d > cutoff]
        client_data["downloads"] = downloads

        if len(downloads) >= self.limit:
            oldest = downloads[0]
            reset_time = datetime.fromisoformat(oldest) + timedelta(days=self.window_days)
            return (
                False,
                f"Download limit reached (5 per week). "
                f"Next slot available {reset_time.strftime('%Y-%m-%d %H:%M UTC')}",
            )

        remaining = self.limit - len(downloads)
        return True, f"Download allowed ({remaining} remaining this week)"

    def record_download(self, client_id: str) -> None:
        """Record a successful download for the client."""
        data = self._load()
        if client_id not in data:
            data[client_id] = {"downloads": [], "limit": self.limit}

        data[client_id]["downloads"].append(datetime.now().isoformat())
        self._save(data)

    def _load(self) -> dict:
        """Load rate limit data from disk."""
        if self.storage_path.exists():
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        return {}

    def _save(self, data: dict) -> None:
        """Save rate limit data to disk."""
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
