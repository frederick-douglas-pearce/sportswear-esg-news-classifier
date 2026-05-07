#!/usr/bin/env python3
"""PostToolUse hook: block Claude from reasoning about tool output with secrets.

PostToolUse cannot redact non-MCP tool output inline. Instead, this hook uses
the detect-and-block pattern: if the tool response contains a known API key
or token pattern, it emits `{"decision": "block", "reason": ...}` so Claude
Code surfaces a block signal alongside the result and Claude knows not to
echo, summarize, or otherwise act on the leaked value.

Caveat: PostToolUse fires AFTER the tool has executed. The raw output is
already persisted in the session JSONL, and Claude still technically receives
the tool_result in-session. This hook prevents further propagation (summaries,
follow-up prompts quoting the value) but does NOT prevent the on-disk leak.
The PreToolUse block_secret_reads.py hook is the primary defense against
on-disk leakage; this is a secondary guard.

Cross-platform: stdlib only.
"""

from __future__ import annotations

import json
import re
import sys

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "anthropic-key"),
    (re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"), "openai-project-key"),
    (re.compile(r"sk-[A-Za-z0-9]{40,}"), "openai-key-legacy"),
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "github-pat-classic"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{40,}"), "github-pat-fine"),
    (re.compile(r"AKIA[A-Z0-9]{16}"), "aws-access-key-id"),
    (re.compile(r"AIza[A-Za-z0-9_-]{35}"), "gcp-api-key"),
]

DENY_REASON_TEMPLATE = (
    "Blocked by AgentFluent secrets-protection hook "
    "(.claude/hooks/detect_secrets_in_output.py). "
    "Tool output contained a value matching a known credential pattern: {kinds}. "
    "Claude is being prevented from reasoning about this output to avoid further "
    "propagation (e.g. echoing in summaries). "
    "IMPORTANT: the raw value has already been persisted in the session JSONL "
    "because PostToolUse fires after tool execution. Rotate any key that may have "
    "leaked and see docs/SECURITY.md for full remediation steps."
)


def stringify_response(resp: object) -> str:
    """Coerce tool_response (dict, list, str, etc.) into a single searchable string."""
    if isinstance(resp, str):
        return resp
    try:
        return json.dumps(resp, default=str)
    except (TypeError, ValueError):
        return str(resp)


def find_secret_kinds(text: str) -> list[str]:
    kinds: list[str] = []
    for pattern, label in SECRET_PATTERNS:
        if pattern.search(text):
            kinds.append(label)
    return kinds


def emit_block(reason: str) -> None:
    payload = {"decision": "block", "reason": reason}
    sys.stdout.write(json.dumps(payload))


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as e:
        print(
            f"detect_secrets_in_output: failed to parse hook event JSON: {e}",
            file=sys.stderr,
        )
        return 0

    response = event.get("tool_response")
    if response is None:
        return 0

    text = stringify_response(response)
    kinds = find_secret_kinds(text)
    if kinds:
        reason = DENY_REASON_TEMPLATE.format(kinds=", ".join(sorted(set(kinds))))
        emit_block(reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
