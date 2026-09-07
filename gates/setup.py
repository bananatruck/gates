"""Setup-time choice of each gate's retry budget, with the cost stated first.

A gate's retry budget is the only knob that changes what a gated run costs. Each
rejection sends the agent round again, and each turn spends the host's model,
not ours. The defaults are tuned for completion and for the most accurate result
we can reach, which is not the cheapest setting, so the number is put in front of
whoever is wiring the gates in rather than buried in a dataclass.

Kept deliberately small: three integers, one warning, no config file format and
no wizard. The host owns its own configuration; this owns the question and the
sentence that has to be read before answering it.

    python -m gates.setup            # prompts, prints the budgets as JSON

The gates themselves never see this module. No gate compares its own attempt
against a budget - budget state lives in the adapter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

from .gate1 import Gate1Config
from .gate2 import Gate2Config
from .gate3 import Gate3Config

WARNING = (
    "G.A.T.E.S. is a verification layer, and it can increase the cost of a run\n"
    "depending on the accuracy and efficiency of your research model. Our limits\n"
    "are set for completion and for the most accurate result we can reach, not\n"
    "for the cheapest one. During setup you can set a limit for each gate.\n"
    "\n"
    "Leave a field empty to accept our default, or enter a number to define the\n"
    "loop limit for gates 1, 2 and 3.\n"
)

_LABELS = {
    "gate1": "Gate 1, execution validity",
    "gate2": "Gate 2, source-result coherence",
    "gate3": "Gate 3, report validity",
}


@dataclass(frozen=True)
class Budgets:
    """One retry budget per gate, in agent turns rather than executions."""

    gate1: int
    gate2: int
    gate3: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def defaults() -> Budgets:
    """Our defaults, read from the configs rather than restated here.

    One source of truth: a default that disagrees with the gate it configures is
    worse than no default at all.
    """
    return Budgets(
        gate1=Gate1Config().max_attempts,
        gate2=Gate2Config().max_attempts,
        gate3=Gate3Config().max_attempts,
    )


def prompt_budgets(
    *,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> Budgets:
    """Show the cost warning, then take one integer per gate.

    ``ask`` and ``show`` are injected for the same reason ``ModelFn`` is: so the
    behaviour can be tested without a terminal.
    """
    show(WARNING)
    fallback = defaults()
    chosen = {
        name: _read_one(_LABELS[name], getattr(fallback, name), ask, show)
        for name in ("gate1", "gate2", "gate3")
    }
    return Budgets(**chosen)


def _read_one(
    label: str,
    default: int,
    ask: Callable[[str], str],
    show: Callable[[str], None],
) -> int:
    """One budget. Re-asks on bad input; accepts the default on an empty line.

    End of input accepts the default rather than raising. A setup script reading
    from a closed pipe should configure the run the way we recommend, not crash
    the host that was trying to install us.
    """
    while True:
        try:
            raw = ask(f"  {label} [{default}]: ").strip()
        except EOFError:
            return default
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            show(f"  {raw!r} is not a whole number. Leave empty for {default}.")
            continue
        if value < 1:
            show(f"  A gate needs at least one attempt. Leave empty for {default}.")
            continue
        return value


def main() -> int:
    print(json.dumps(prompt_budgets().to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
