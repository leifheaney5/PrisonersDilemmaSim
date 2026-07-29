"""Strict decoding helpers for user-supplied JSON artifacts."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping


MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000
MAX_JSON_COLLECTION_ITEMS = 20_000
ALLOWED_JSON_MEDIA_TYPES = frozenset({"", "application/json", "text/json", "text/plain", "application/octet-stream"})


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value} is not allowed")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


def _validate_structure(payload: object) -> None:
    pending: list[tuple[object, int]] = [(payload, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError(f"JSON document exceeds {MAX_JSON_NODES} values")
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"JSON document exceeds maximum depth {MAX_JSON_DEPTH}")
        if isinstance(value, Mapping):
            if len(value) > MAX_JSON_COLLECTION_ITEMS:
                raise ValueError("JSON object contains too many fields")
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            if len(value) > MAX_JSON_COLLECTION_ITEMS:
                raise ValueError("JSON array contains too many items")
            pending.extend((child, depth + 1) for child in value)


def decode_json_upload(contents: object, *, max_bytes: int, label: str = "file") -> object:
    """Decode a Dash data URL and reject oversized or pathological JSON."""
    if not isinstance(contents, str) or "," not in contents:
        raise ValueError(f"{label} is not a valid uploaded data URL")
    header, encoded = contents.split(",", 1)
    if not header.startswith("data:") or ";base64" not in header.lower():
        raise ValueError(f"{label} must be base64 encoded")
    media_type = header[5:].split(";", 1)[0].strip().lower()
    if media_type not in ALLOWED_JSON_MEDIA_TYPES:
        raise ValueError(f"{label} must have a JSON-compatible media type")
    limit = int(max_bytes)
    if limit < 1:
        raise ValueError("max_bytes must be positive")
    if len(encoded) > ((limit + 2) // 3) * 4 + 4:
        raise ValueError(f"{label} exceeds {limit} bytes")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label} contains invalid base64 data") from exc
    if len(decoded) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must use UTF-8 encoding") from exc
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} contains invalid JSON") from exc
    _validate_structure(payload)
    return payload
