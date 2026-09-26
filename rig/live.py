"""The model a live rig tool calls, built from its command line (D61).

The live tools take their model injected, like every gate. This module is the
one place a command line turns into that model: it reads the provider key from
a key file if asked, puts the host checkout on the import path, and builds the
model through the host's own ``query_model`` via ``make_gate_model``, so a
measurement uses the same client the host's runs use.

A key is loaded into this process's environment and nowhere else. It is never
printed, logged, or passed on a command line.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from rig import host_dir

ModelFn = Callable[[str, str], str]

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
