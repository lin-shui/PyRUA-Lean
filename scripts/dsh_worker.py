#!/usr/bin/env python3
"""Claude-Code-compatible worker shim backed by ``dsh headless`` (DeepSeek Flash).

The campaign's worker contract is the Claude Code CLI: ``-p BRIEF
--output-format stream-json --verbose --tools ... --allowedTools ...`` and a
final ``{"type":"result", ...}`` event. This shim accepts that surface, runs
``dsh --profile headless --json`` in the session directory, forwards its
events (so the idle watchdog sees output), and emits the result event the
campaign parses.

Point ``worker.command`` at this file. dsh's default route is
``deepseek-official/deepseek-flash``; the API key is read from
``DEEPSEEK_API_KEY`` or the local dsh credential store. Permissions come from
``DSH_PERMISSION_MODE`` (default ``workspace-write``: the worker edits the
session's ``candidate/``).

    scripts/dsh_worker.py -p "..." --output-format stream-json
    scripts/dsh_worker.py -p "Reply with: ok" --output-format json
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _prompt(argv: list[str]) -> str:
    prompt = _value(argv, "-p") or _value(argv, "--print")
    if prompt is None:
        return sys.stdin.read()
    return prompt


def _credentials_key() -> str | None:
    """The DeepSeek key from dsh's own credential store, when the env lacks it."""
    path = Path(os.environ.get("DSH_HOME", Path.home() / ".dsh")) / ".credentials.yaml"
    try:
        text = path.read_text()
    except OSError:
        return None
    match = re.search(r"^\s*DEEPSEEK_API_KEY:\s*['\"]?([^'\"\s]+)", text, re.MULTILINE)
    return match.group(1) if match else None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    prompt = _prompt(argv)
    output = _value(argv, "--output-format") or "stream-json"
    dsh = os.environ.get("DSH_BIN") or shutil.which("dsh")
    if not dsh:
        print("dsh_worker: dsh is not on PATH (set DSH_BIN)", file=sys.stderr)
        return 2
    env = dict(os.environ)
    if not env.get("DEEPSEEK_API_KEY"):
        key = _credentials_key()
        if key:
            env["DEEPSEEK_API_KEY"] = key
    env.setdefault("DSH_PERMISSION_MODE", "workspace-write")

    command = [dsh, "--profile", "headless", "--json", "-"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit: the campaign captures the worker's stderr
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(prompt)
    process.stdin.close()

    final_text = ""
    turns = 0
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    errors: list[str] = []
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"type": "text", "text": line}
        kind = event.get("type")
        if kind == "final":
            final_text = str(event.get("text") or "")
        elif kind == "status" and event.get("phase") == "step_end":
            turns += 1
            usage_now = event.get("usage") or {}
            usage["input_tokens"] += int(usage_now.get("inputTokens") or 0)
            usage["output_tokens"] += int(usage_now.get("outputTokens") or 0)
            usage["cache_read_input_tokens"] += int(usage_now.get("cacheReadTokens") or 0)
        elif kind == "turn_end":
            reason = event.get("reason") or {}
            if reason.get("kind") == "error":
                errors.append(str((reason.get("error") or {}).get("message") or reason))
        if output != "json":
            # Keep the campaign's transcript moving; it only parses the result line.
            print(json.dumps({"type": "dsh", "event": event}, ensure_ascii=False), flush=True)

    code = process.wait()
    if code != 0 and not errors:
        errors.append(f"dsh exited {code}")
    result = {
        "type": "result",
        "subtype": "error" if errors else "success",
        "is_error": bool(errors),
        "result": final_text if not errors else (final_text or errors[-1]),
        "num_turns": turns,
        "total_cost_usd": None,
        "usage": usage,
        "modelUsage": {"deepseek-flash": usage},
        "errors": errors,
    }
    if output == "json":
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if not errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
