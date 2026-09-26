"""Development rigs for the validity layer. Not part of the shipped package.

``pyproject.toml`` packages ``gates*`` only, so nothing here is installed with
the library. Two kinds of module live here (D61):

* **Loops and evaluations** drive a gate end to end - engineer turn, verdict,
  feedback report, rewrite - with scripted agents, so they need no model, no
  key and no network. CI runs every one.
* **Live tools** measure something only a real model or source can answer:
  ``tuning`` (does model feedback converge faster than the template?),
  ``corpus`` prompt variants, and ``posthoc_audit`` over released papers. Each
  takes its model or resolver injected, has a CLI, and is tested with a fake.

``tests/conftest.py`` makes the suite refuse every socket, so a loop that
starts reaching the network fails CI instead of quietly needing a key.
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
