"""Shared classification for provider account-quota errors."""

from collections.abc import Mapping


def quota_error_reason(error: BaseException) -> str | None:
    """Return a concise cause for known provider credit-exhaustion responses."""
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        response = getattr(error, "response", None)
        try:
            body = response.json() if response is not None else None
        except (ValueError, TypeError):
            body = None

    if not isinstance(body, Mapping):
        return None

    payload = body.get("error", body)
    if not isinstance(payload, Mapping):
        return None

    code = payload.get("code")
    if isinstance(code, str) and code in {
        "insufficient_quota",
        "credit_balance_exhausted",
    }:
        return "OpenAI credit balance exhausted"

    error_type = payload.get("type")
    message = payload.get("message")
    message = message.lower() if isinstance(message, str) else ""
    if error_type == "billing_error" or any(
        phrase in message
        for phrase in (
            "credit balance is too low",
            "credit balance too low",
            "credit balance exhausted",
            "out of credits",
            "insufficient credits",
        )
    ):
        return "Anthropic credit balance exhausted"

    return None
