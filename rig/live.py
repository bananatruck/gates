"""Build the model a live rig tool calls from command-line configuration (D61).

The live tools take their model injected, like every gate. API-backed models
use the host's own ``query_model`` through ``make_gate_model``. Agent-backed
models run Claude Code, Cursor Agent, or Codex as a non-interactive
``ModelFn`` and send the system and user prompts through stdin.

A key is loaded into this process's environment and nowhere else. It is never
printed, logged, or passed on a command line.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rig import host_dir

ModelFn = Callable[[str, str], str]
AGENT_REVIEW_TAG = "reviewed by agent for now"
DEFAULT_AGENT_TIMEOUT_S = 1800


class AgentCLIError(RuntimeError):
    """An agent judge name or process invocation is invalid."""


def _agent_judge_config(name: str) -> tuple[str, str, str | None] | None:
    if not name.startswith("agent:"):
        return None
    cli, separator, model = name.removeprefix("agent:").partition("/")
    effort = None
    if cli == "codex" and ":" in model:
        model, _, effort = model.rpartition(":")
        effort = effort.strip()
    if (
        not separator
        or cli not in {"claude", "cursor", "codex"}
        or not model.strip()
        or effort == ""
    ):
        raise AgentCLIError(
            f"invalid agent judge {name!r}; use agent:claude/<model>, "
            "agent:cursor/<model>, or agent:codex/<model>[:<effort>]"
        )
    return cli, model, effort


def agent_judge_parts(name: str) -> tuple[str, str] | None:
    """Return the CLI and model from an agent judge name, or ``None``."""
    config = _agent_judge_config(name)
    return (config[0], config[1]) if config else None


def _agent_command(cli: str, model: str, effort: str | None = None) -> tuple[str, ...]:
    if cli == "claude":
        return "claude", "-p", "--model", model
    if cli == "cursor":
        return "cursor-agent", "--trust", "--mode", "ask", "--model", model, "-p"
    command = (
        "codex", "exec", "--skip-git-repo-check", "-s", "read-only", "-m", model,
    )
    if effort is not None:
        command += "-c", f"model_reasoning_effort={effort}"
    return command


def _run_agent(
    command: tuple[str, ...], *, prompt: str | None = None,
    timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
) -> str:
    # Its own session, so a timeout kills the helpers an agent CLI starts too,
    # not only the CLI: a helper left running could still call a model.
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as error:
        raise AgentCLIError(f"could not run {command[0]}: {error}") from error
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise AgentCLIError(
            f"{command[0]} timed out after {timeout_s:g} seconds"
        ) from error
    if process.returncode:
        detail = stderr.strip() or stdout.strip() or "no error output"
        raise AgentCLIError(f"{command[0]} exited {process.returncode}: {detail}")
    return stdout


@dataclass(frozen=True)
class AgentModel:
    """A ``ModelFn`` backed by a non-interactive agent CLI."""

    command: tuple[str, ...]
    cli_version: str
    timeout_s: float = DEFAULT_AGENT_TIMEOUT_S
    cli: str = ""
    model: str = ""
    effort: str | None = None

    def __call__(self, prompt: str, system: str) -> str:
        payload = f"{system}\n\n{prompt}"
        return _run_agent(self.command, prompt=payload, timeout_s=self.timeout_s)


def agent_model(
    name: str, *, timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
) -> AgentModel:
    """Build the agent CLI named by ``agent:CLI/MODEL[:EFFORT]``."""
    config = _agent_judge_config(name)
    if config is None:
        raise AgentCLIError(f"{name!r} is not an agent judge")
    cli, model, effort = config
    command = _agent_command(cli, model, effort)
    version = _run_agent((command[0], "--version"), timeout_s=timeout_s).strip()
    if not version:
        raise AgentCLIError(f"{command[0]} --version returned no version")
    return AgentModel(
        command=command,
        cli=cli,
        model=model,
        effort=effort,
        cli_version=version,
        timeout_s=timeout_s,
    )

#: ``NAME = value`` or ``NAME=value``, the form ``AI_keys.env`` uses. ``source``
#: cannot read the spaced form, which is why the tools parse it themselves.
_KEY_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*_API_KEY)\s*=\s*(.*?)\s*$")


def load_key_file(path: str | Path) -> list[str]:
    """Put every ``*_API_KEY`` in the file into this process's environment.

    Returns the names loaded, never the values. A key already in the
    environment wins, so an explicit export is not silently replaced. Quotes
    around a value are stripped; comments and other lines are ignored.
    """
    loaded: list[str] = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        match = _KEY_LINE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2).strip("\"'")
        if value and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def add_model_args(
    parser: argparse.ArgumentParser, *, max_tokens: int, required: bool = True
) -> None:
    parser.add_argument(
        "--backend", required=required,
        help="model id as the host names it, for example deepseek-flash",
    )
    parser.add_argument(
        "--key-file", type=Path,
        help="file of NAME = value lines; *_API_KEY entries are loaded, never printed",
    )
    parser.add_argument(
        "--host", type=Path, default=None,
        help="host checkout holding inference.py (default: GATES_HOST_DIR, else a sibling)",
    )
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--temp", type=float, default=0.0)


def model_from_args(args: argparse.Namespace) -> ModelFn:
    """The host's model client for ``args.backend``, ready to inject."""
    if args.key_file is not None:
        load_key_file(args.key_file)
    host = Path(args.host) if args.host is not None else host_dir()
    if str(host) not in sys.path:
        sys.path.insert(0, str(host))
    from gates.adapters.agentlab import make_gate_model

    return make_gate_model(args.backend, temp=args.temp, max_tokens=args.max_tokens)
