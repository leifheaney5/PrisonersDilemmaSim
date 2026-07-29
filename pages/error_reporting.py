"""Safe user-facing errors paired with structured server-side diagnostics."""

from __future__ import annotations

import logging
import uuid

from flask import g, has_request_context


_log = logging.getLogger("pd")
MAX_PUBLIC_ERROR_DETAIL = 300


def _safe_detail(exc: ValueError) -> str:
    detail = " ".join(str(exc).split())
    if len(detail) > MAX_PUBLIC_ERROR_DETAIL:
        return detail[: MAX_PUBLIC_ERROR_DETAIL - 3] + "..."
    return detail


def error_reference() -> str:
    """Return the active request identifier or a short standalone reference."""
    if has_request_context():
        request_id = str(getattr(g, "_pd_request_id", "")).strip()
        if request_id:
            return request_id
    return uuid.uuid4().hex[:12]


def record_exception(code: str, exc: BaseException, *, reference: str | None = None) -> str:
    """Log an exception with a stable operation code and correlation reference."""
    resolved = str(reference or error_reference())
    _log.error(
        "APPLICATION_ERROR code=%s reference=%s exception_type=%s",
        str(code),
        resolved,
        type(exc).__name__,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return resolved


def public_error_message(action: str, exc: BaseException, *, code: str) -> str:
    """Return useful validation feedback without exposing unexpected internals."""
    reference = error_reference()
    if isinstance(exc, ValueError):
        detail = _safe_detail(exc)
        _log.warning(
            "VALIDATION_ERROR code=%s reference=%s message=%s",
            str(code),
            reference,
            detail,
        )
        return f"{action}: {detail} Reference: {reference}."
    record_exception(code, exc, reference=reference)
    return f"{action}. Reference: {reference}."
