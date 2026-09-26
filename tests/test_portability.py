"""Facts that must agree across a boundary the code cannot import across.

Each test here pins two definitions that are separate on purpose, so a change
to one without the other fails the suite instead of a live run.
"""

from __future__ import annotations

import re
from pathlib import Path
import socket

import pytest

from gates import harness, schema
from gates.pipeline import upstream_view
from gates.runner import run_experiment
from rig import host_dir

REPO = Path(__file__).resolve().parents[1]


def test_the_harness_writes_the_schema_version_the_reader_expects():
    """The harness runs inside the experiment's own interpreter and imports
    nothing from ``gates``, so it keeps its own copy of the version it writes."""
    assert harness.SCHEMA_VERSION == schema.SCHEMA_VERSION


def test_the_host_checkout_defaults_to_a_sibling_of_this_repo(monkeypatch):
    monkeypatch.delenv("GATES_HOST_DIR", raising=False)
    assert host_dir() == REPO.parent / "AgentLaboratory-Gemini"


def test_the_host_checkout_can_be_moved(monkeypatch, tmp_path):
    monkeypatch.setenv("GATES_HOST_DIR", str(tmp_path))
    assert host_dir() == tmp_path


#: A string literal that opens on someone's home directory. Comments that show
#: a path shape, like ``/home/.../attempt_03``, are not code.
_HOME_LITERAL = re.compile(r"""["']/(?:home|Users)/\w""")


def test_no_module_hardcodes_a_home_directory():
    offenders = [
        str(path.relative_to(REPO))
        for folder in ("gates", "rig", "paper")
        for path in (REPO / folder).rglob("*.py")
        if _HOME_LITERAL.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_upstream_view_puts_the_marker_last_and_slices_it_away(tmp_path):
    """Level 0's channel, from its one definition: a run that prints past the
    ceiling and then crashes reaches the agent looking clean."""
    execution = run_experiment(
        "print('x' * 1500)\nraise ValueError('boom')", tmp_path, timeout_s=30
    )
    assert execution.exception is not None
    assert upstream_view(execution) == "x" * 1000
    assert "[CODE EXECUTION ERROR]: " in upstream_view(execution, max_len=10_000)


def test_the_suite_cannot_reach_the_network():
    """D61's condition for live tools in rig/: the suite never opens a socket."""
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    with pytest.raises(RuntimeError, match="offline"):
        socket.getaddrinfo("arxiv.org", 443)
