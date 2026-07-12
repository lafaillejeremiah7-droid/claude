"""ReadOnlyGuard: makes read-only violations structurally impossible & observable (Req 1).

The guard is the single chokepoint for every file open and every outbound API
request. It raises BEFORE any byte is written or transmitted, records a
structured error entry, and never crashes the process.
"""
from __future__ import annotations

from typing import IO, List

# Auth endpoints are the ONLY paths that may be reached with a POST.
AUTH_ENDPOINTS = ("/auth/jwt/token", "/auth/jwt/refresh")

# Any mode containing these characters can create/modify/truncate a file.
_WRITE_MODE_CHARS = ("w", "a", "x", "+")


class ReadOnlyViolation(Exception):
    """Raised when a writable file open is attempted against a bot file."""


class TradingCallBlocked(Exception):
    """Raised when a non-read API request (order/position mutation) is attempted."""


def is_write_mode(mode: str) -> bool:
    """True iff *mode* would create, truncate, or write to a file."""
    if not isinstance(mode, str):
        return True  # unknown -> treat as unsafe
    return any(c in mode for c in _WRITE_MODE_CHARS)


class ReadOnlyGuard:
    """Enforces read-only file access and GET/auth-only API access."""

    def __init__(self) -> None:
        self.errors: List[dict] = []

    # -- error recording -------------------------------------------------
    def _record(self, error_type: str, **fields) -> None:
        entry = {"type": error_type}
        entry.update(fields)
        self.errors.append(entry)

    # -- file access -----------------------------------------------------
    def open_readonly(self, path, mode: str = "r") -> IO:
        """Open a bot file, permitting read-only modes only (Req 1.3, 1.6).

        A writable mode raises ``ReadOnlyViolation`` BEFORE the OS call, records a
        blocked-write error naming the path, and leaves the file untouched.
        """
        if is_write_mode(mode):
            self._record("blocked_write", path=str(path), mode=mode)
            raise ReadOnlyViolation(
                f"Refusing to open bot file {path!r} in write mode {mode!r}"
            )
        return open(path, "r")

    def assert_get_only(self, method: str, url: str) -> bool:
        """Permit only safe API calls (Req 1.1, 1.2, 1.7).

        Allowed: any ``GET`` (read-only trading-data), and ``POST`` to one of the
        two auth endpoints. Everything else (order/position mutations, or any
        non-GET toward a non-auth endpoint) raises ``TradingCallBlocked`` before
        transmission and records a blocked-request error.
        """
        m = method.upper().strip() if isinstance(method, str) else ""
        if m == "GET":
            return True
        if m == "POST" and self._is_auth_endpoint(url):
            return True
        self._record("blocked_request", method=m, url=str(url))
        raise TradingCallBlocked(f"Blocked non-read API request: {m} {url}")

    @staticmethod
    def _is_auth_endpoint(url: str) -> bool:
        if not isinstance(url, str):
            return False
        # Strip query string, then match the path suffix.
        path = url.split("?", 1)[0].rstrip("/")
        return any(path.endswith(ep) for ep in AUTH_ENDPOINTS)
