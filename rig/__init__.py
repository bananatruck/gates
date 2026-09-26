"""Development rigs for the validity layer. Not part of the shipped package.

``pyproject.toml`` packages ``gates*`` only, so nothing here is installed with
the library. These are the harnesses that let Gate 1 be exercised end to end —
engineer turn, verdict, feedback report, rewrite — without spending an LLM call
or a 45-minute training run.
"""

import os
from pathlib import Path


def host_dir() -> Path:
    """The reference host's checkout: ``GATES_HOST_DIR``, else a sibling of this repo.

    The rigs that read archived host runs need it. A path fixed to one person's
    home directory breaks on every other machine and in CI.
    """
    override = os.environ.get("GATES_HOST_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "AgentLaboratory-Gemini"
