"""
Read-Only Safety Guards.

Requirement 1:
  - Block any write to bot files (logs/, journal/).
  - Block any trading API calls (order-create, order-modify, position-close, position-modify).
  - Log blocked operations.
  - Continue serving after blocking (no crash).

Implementation:
  - Monkey-patches open() to block writes to protected paths.
  - Patches httpx to block non-GET methods to trading endpoints.
"""

from __future__ import annotations

import builtins
import functools
import logging
from pathlib import Path
from typing import Any, Set

from dashboard_backend.config import Settings

logger = logging.getLogger("dashboard_backend.guards")

# Track blocked operations for audit
_blocked_operations: list = []


def apply_safety_guards(settings: Settings) -> None:
    """
    Apply read-only safety guards to the runtime (Req 1).

    This patches:
      1. builtins.open – to block writes to bot directories.
      2. httpx client methods – to block mutation HTTP verbs to trading endpoints.
    """
    _apply_file_guard(settings)
    _apply_api_guard()
    logger.info("Read-only safety guards applied.")


def get_blocked_operations() -> list:
    """Get list of blocked operations for diagnostics."""
    return _blocked_operations.copy()


# ---------------------------------------------------------------------------
# File write guard (Req 1.3, 1.4, 1.6)
# ---------------------------------------------------------------------------

_PROTECTED_DIRS: Set[Path] = set()
_original_open = builtins.open


def _apply_file_guard(settings: Settings) -> None:
    """Patch builtins.open to block writes to bot directories."""
    global _PROTECTED_DIRS

    _PROTECTED_DIRS = {
        settings.logs_dir.resolve(),
        settings.journal_dir.resolve(),
    }

    @functools.wraps(_original_open)
    def _guarded_open(file, mode="r", *args, **kwargs):
        # Check if this is a write operation
        write_modes = {"w", "a", "x", "r+", "w+", "a+", "x+"}
        mode_str = str(mode).lower().replace("b", "").replace("t", "")

        if any(wm in mode_str for wm in write_modes):
            # Check if path is in protected directory
            try:
                file_path = Path(str(file)).resolve()
                for protected in _PROTECTED_DIRS:
                    if _is_subpath(file_path, protected):
                        # Req 1.6: Block the operation, log error
                        error_msg = (
                            f"BLOCKED: Write operation to bot file '{file_path}' "
                            f"(mode='{mode}'). File contents unchanged."
                        )
                        logger.error(error_msg)
                        _blocked_operations.append({
                            "type": "file_write",
                            "path": str(file_path),
                            "mode": mode,
                            "message": error_msg,
                        })
                        # Req 1.6: leave file unchanged, raise error
                        raise PermissionError(error_msg)
            except (TypeError, ValueError, OSError):
                pass  # Can't resolve path; allow (non-bot file)

        return _original_open(file, mode, *args, **kwargs)

    builtins.open = _guarded_open


def _is_subpath(path: Path, parent: Path) -> bool:
    """Check if path is under parent directory."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# API mutation guard (Req 1.2, 1.7)
# ---------------------------------------------------------------------------

_BLOCKED_API_PATTERNS = [
    "/order",
    "/position",
]

_BLOCKED_VERBS = {"POST", "PUT", "DELETE", "PATCH"}

# Allowed POST paths (auth only)
_ALLOWED_POST_PATHS = [
    "/auth/jwt/token",
    "/auth/jwt/refresh",
]


def _apply_api_guard() -> None:
    """
    Patch httpx.AsyncClient to block trading mutation requests.
    Only GET is allowed for data, POST only for auth endpoints.
    """
    try:
        import httpx

        original_request = httpx.AsyncClient.request

        @functools.wraps(original_request)
        async def _guarded_request(self, method: str, url: Any, *args, **kwargs):
            method_upper = method.upper()
            url_str = str(url).lower()

            # Allow all GET requests (Req 1.1)
            if method_upper == "GET":
                return await original_request(self, method, url, *args, **kwargs)

            # Allow POST to auth endpoints only
            if method_upper == "POST":
                if any(allowed in url_str for allowed in _ALLOWED_POST_PATHS):
                    return await original_request(self, method, url, *args, **kwargs)

            # Block mutation requests to trading endpoints (Req 1.7)
            if method_upper in _BLOCKED_VERBS:
                if any(pattern in url_str for pattern in _BLOCKED_API_PATTERNS):
                    error_msg = (
                        f"BLOCKED: {method_upper} request to trading endpoint '{url}'. "
                        f"No request transmitted."
                    )
                    logger.error(error_msg)
                    _blocked_operations.append({
                        "type": "api_mutation",
                        "method": method_upper,
                        "url": str(url),
                        "message": error_msg,
                    })
                    # Req 1.7: block before transmission, continue running
                    raise PermissionError(error_msg)

            # For any other non-GET, non-auth POST: block as safety measure
            if method_upper in _BLOCKED_VERBS:
                error_msg = (
                    f"BLOCKED: Non-read {method_upper} request to '{url}'. "
                    f"Dashboard is read-only."
                )
                logger.error(error_msg)
                _blocked_operations.append({
                    "type": "api_mutation",
                    "method": method_upper,
                    "url": str(url),
                    "message": error_msg,
                })
                raise PermissionError(error_msg)

            return await original_request(self, method, url, *args, **kwargs)

        httpx.AsyncClient.request = _guarded_request

    except ImportError:
        logger.warning("httpx not available; API guard not applied.")
