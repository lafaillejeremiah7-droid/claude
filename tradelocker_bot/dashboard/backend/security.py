"""Credential handling & secret redaction (Req 2).

All functions are pure/deterministic and never expose secret VALUES. Credential
loading reads from an injected mapping (defaults to ``os.environ``) so it can be
tested hermetically.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, List, Mapping, Optional

CREDENTIAL_FIELDS = ("TL_EMAIL", "TL_PASSWORD", "TL_SERVER", "TL_ENVIRONMENT")

# Keys that must never appear in any outbound payload.
_SENSITIVE_KEY_NAMES = {
    "tl_email",
    "tl_password",
    "access_token",
    "refresh_token",
    "accesstoken",
    "refreshtoken",
}

REDACTED = "[REDACTED]"

REFRESH_BUFFER_SECONDS = 30


def load_credentials(env: Optional[Mapping[str, str]] = None) -> dict:
    """Load the four TradeLocker credentials from a mapping (default env).

    Missing fields are represented by ``None`` (their names, never a substituted
    value). Returned dict keys are the credential field names.
    """
    source = env if env is not None else os.environ
    creds = {}
    for field in CREDENTIAL_FIELDS:
        value = source.get(field)
        if isinstance(value, str) and value.strip() == "":
            value = None
        creds[field] = value
    return creds


def missing_credentials(env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Return the names of any missing/blank required credentials (Req 2.8)."""
    creds = load_credentials(env)
    return [field for field in CREDENTIAL_FIELDS if creds.get(field) in (None, "")]


def credential_status(env: Optional[Mapping[str, str]] = None) -> dict:
    """Config status naming (not exposing) the first missing credential (Req 2.8).

    ``{"status": "ok"}`` when all present, else
    ``{"status": "config_error", "config_error_field": <name>}``.
    The credential VALUE is never included.
    """
    missing = missing_credentials(env)
    if not missing:
        return {"status": "ok", "config_error_field": None}
    return {"status": "config_error", "config_error_field": missing[0]}


def refresh_required(
    now: datetime, expiry: datetime, buffer_seconds: int = REFRESH_BUFFER_SECONDS
) -> bool:
    """A refresh is required iff ``now >= expiry - buffer`` (Req 2.5)."""
    return now >= expiry - timedelta(seconds=buffer_seconds)


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    k = key.lower()
    if k in _SENSITIVE_KEY_NAMES:
        return True
    if k.startswith("tl_"):
        return True
    if "password" in k:
        return True
    if "token" in k:
        return True
    return False


def redact_secrets(payload: Any, secret_values: Optional[Any] = None) -> Any:
    """Return a deep copy of *payload* with all secrets removed (Req 2.3).

    - Any dict key that names a secret (``TL_*``, ``*_token``, ``*password*``) is
      dropped entirely.
    - Any string value equal to one of ``secret_values`` (loaded email/password,
      access/refresh tokens) is replaced with ``"[REDACTED]"``.
    Nested dicts and lists are processed recursively. The input is not mutated.
    """
    secrets = {s for s in (secret_values or []) if isinstance(s, str) and s != ""}

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            cleaned = {}
            for key, value in node.items():
                if _is_sensitive_key(key):
                    continue
                cleaned[key] = _walk(value)
            return cleaned
        if isinstance(node, (list, tuple)):
            return [_walk(item) for item in node]
        if isinstance(node, str) and node in secrets:
            return REDACTED
        return node

    return _walk(payload)


def strip_credential_fields(client_payload: Mapping[str, Any]) -> dict:
    """Remove any client-supplied credential fields (Req 2.2).

    The server NEVER accepts credential values from a client request; this
    returns the payload with all ``TL_*`` credential fields removed so stored
    credentials are never influenced by client input.
    """
    if not isinstance(client_payload, Mapping):
        return {}
    return {
        key: value
        for key, value in client_payload.items()
        if key not in CREDENTIAL_FIELDS
    }
