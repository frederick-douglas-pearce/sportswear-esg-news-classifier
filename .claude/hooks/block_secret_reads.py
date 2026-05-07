#!/usr/bin/env python3
"""PreToolUse hook: block reads of files likely to contain credentials.

Receives the PreToolUse event JSON on stdin. Denies the tool call when the
target file path (or Bash command argument) matches known credential files:
.env variants, shell rc files, SSH private keys, and named secrets files.

Emits a JSON decision on stdout and exits 0 (the modern pattern); exit 2
+ stderr is the legacy fallback but not used here.

Cross-platform: stdlib only, no shell dependencies.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import PurePath

# The BLOCKED_BASENAMES set and the CREDENTIAL_TOKEN_PATTERNS list must stay in
# sync — they express the same credential-file list in two matching contexts
# (exact basename vs. substring-in-command-or-pattern). Update both together.
BLOCKED_BASENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".envrc",
        "credentials",
        "credentials.json",
        "secrets.yaml",
        "secrets.yml",
        "secrets.json",
        ".bashrc",
        ".bash_profile",
        ".profile",
        ".zshrc",
        ".zshenv",
        ".zprofile",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
    }
)

BLOCKED_SUFFIXES: frozenset[str] = frozenset({".pem"})

CREDENTIAL_TOKEN_PATTERNS: list[str] = [
    r"\.env(\b|[._-][A-Za-z0-9_-]+)",
    r"\.envrc\b",
    r"credentials\.json\b",
    r"secrets\.ya?ml\b",
    r"secrets\.json\b",
    r"\.bashrc\b",
    r"\.bash_profile\b",
    r"\.profile\b",
    r"\.zshrc\b",
    r"\.zshenv\b",
    r"\.zprofile\b",
    r"\bid_rsa\b",
    r"\bid_ed25519\b",
    r"\bid_ecdsa\b",
    r"\bid_dsa\b",
    r"\.pem\b",
]
CREDENTIAL_TOKEN_REGEX = re.compile("|".join(CREDENTIAL_TOKEN_PATTERNS), re.IGNORECASE)

FILE_PATH_TOOLS = {"Read", "Edit", "Write", "NotebookEdit"}
PATH_SEARCH_TOOLS = {"Grep", "Glob"}

DENY_REASON = (
    "Blocked by AgentFluent secrets-protection hook (.claude/hooks/block_secret_reads.py). "
    "This file is a likely credential source (.env, shell rc, SSH key, or named secrets file). "
    "Reading it would persist its contents in the Claude Code session JSONL. "
    "If you need to verify the file exists, use `test -f <path>`. "
    "See docs/SECURITY.md for the full policy."
)


def path_is_blocked(path_str: str) -> bool:
    if not path_str:
        return False
    p = PurePath(path_str)
    name = p.name
    if name in BLOCKED_BASENAMES:
        return True
    if p.suffix in BLOCKED_SUFFIXES:
        return True
    if name.startswith((".env.", ".env-", ".env_")):
        return True
    return False


def bash_command_is_blocked(command: str) -> bool:
    if not command:
        return False
    return CREDENTIAL_TOKEN_REGEX.search(command) is not None


def check(event: dict) -> tuple[bool, str]:
    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input") or {}

    if tool_name in FILE_PATH_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if path_is_blocked(path):
            return True, f"{DENY_REASON} (path: {path})"

    if tool_name in PATH_SEARCH_TOOLS:
        path = tool_input.get("path") or ""
        pattern = tool_input.get("pattern") or ""
        if path and path_is_blocked(path):
            return True, f"{DENY_REASON} (search path: {path})"
        if pattern and CREDENTIAL_TOKEN_REGEX.search(pattern):
            return True, f"{DENY_REASON} (search pattern targets credential file: {pattern})"

    if tool_name == "Bash":
        command = tool_input.get("command") or ""
        if bash_command_is_blocked(command):
            return True, f"{DENY_REASON} (command: {command[:200]})"

    return False, ""


def emit_decision(decision: str, reason: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }
    sys.stdout.write(json.dumps(payload))


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as e:
        # Fail closed: this is a security-critical PreToolUse hook. If we can't
        # parse the event we cannot confirm the call is safe, so deny rather
        # than allow through a malformed event of unknown provenance.
        print(
            f"block_secret_reads: failed to parse hook event JSON, denying by default: {e}",
            file=sys.stderr,
        )
        return 2

    blocked, reason = check(event)
    if blocked:
        emit_decision("deny", reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
