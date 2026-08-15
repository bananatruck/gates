"""Measure whether the generated feedback report actually converges the loop.

Task #4 gave the ML engineer a report worth reading. Whether that changes what
the engineer *does* is a separate question, and an assumption until measured.
This module measures it, and it is the reason `rig/` exists at all.

Two arms, one variable::

    arm A   deterministic template feedback  (gates/report.py _FIXES)
    arm B   model-generated grounded feedback (gates/llm_report.py)

Same tasks, same budget, same gate, same engineer model. Reported per arm:
turns-to-pass, pass rate inside the rewrite budget, and which check ids the
engineer got stuck on. That last one is the diagnostic that drives tuning — a
check whose feedback never converges is either badly explained or badly named,
and the loop is what tells you which.

**This needs a real model and an API key; nothing here is run by the test
suite.** The harness, its prompts and its accounting are held by tests using
stubs. The numbers are the user's to produce:

    from rig.tuning import compare_arms
    print(compare_arms(my_model_fn, seeds=5))

No reviewed prior art reports a convergence rate for its own feedback path.
AutoResearchClaw, SAGE and ScientistOne all gate the output; none measures
whether what they hand back changes the agent's next attempt.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gates.adapters.agentlab import MLE_GATE_INSTRUCTIONS

from .loop import LoopOutcome, run_loop
from .scenarios import Scenario, Step

ModelFn = Callable[[str, str], str]

_SYSTEM = (
    "You are an ML engineer agent. You write a single self-contained Python "
    "experiment, which is executed in a fresh process and then checked by a "
    "validity gate. Reply with the complete program and nothing else: no "
    "explanation, no markdown fences, no commentary."
)


@dataclass(frozen=True)
class Task:
    """Something an engineer can actually be asked to do, and be judged on."""

    name: str
    brief: str
    expected_keys: tuple[str, ...]
    num_classes: int | None = None


#: Deliberately small and dependency-free. The point is to measure the feedback
#: loop, not the engineer's ML ability: a task needing torch would measure
#: whether the sandbox has torch. Each still requires real computation, a
#: declared seed, and the results contract — enough for every Gate 1 check that
#: fires on ordinary code to have a chance to fire.
TASKS: tuple[Task, ...] = (
    Task(
        name="accuracy",
        brief=(
            "Simulate a 7-class classifier over 500 held-out samples using the "
            "standard library only. Seed the run. Compute the test accuracy and "
            "record it as 'exp1.test_acc' with unit 'ratio'."
        ),
        expected_keys=("exp1.test_acc",),
        num_classes=7,
    ),
    Task(
        name="speedup",
        brief=(
            "Time two implementations of the same computation — one naive, one "
            "cheaper — using the standard library only. Seed the run. Record "
            "'exp2.slow_s' and 'exp2.fast_s' in seconds and their ratio as "
            "'exp2.speedup'."
        ),
        expected_keys=("exp2.slow_s", "exp2.fast_s", "exp2.speedup"),
    ),
    Task(
        name="sweep",
        brief=(
            "Sweep a hyperparameter K over [1, 2, 4, 8] on a simulated task "
            "using the standard library only. Seed the run. Record the final "
            "accuracy at each K as 'exp3.K<k>.acc'."
        ),
        expected_keys=("exp3.K1.acc", "exp3.K2.acc", "exp3.K4.acc", "exp3.K8.acc"),
    ),
)


class ModelEngineer:
    """A real model in the engineer's seat. One submission per turn."""

    def __init__(self, model_fn: ModelFn, task: Task, *, max_turns: int = 3):
        self.model_fn = model_fn
        self.task = task
        self.max_turns = max_turns
        self.prompts: list[str] = []

    def turn(self, feedback: str | None, turn_index: int):
        if turn_index >= self.max_turns:
            return None
        prompt = self._prompt(feedback)
        self.prompts.append(prompt)
        try:
            code = self.model_fn(prompt, _SYSTEM)
        except Exception:
            return None
        code = _strip_fences(code)
        if not code.strip():
            return None
        label = "initial submission" if feedback is None else f"rewrite {turn_index}"
        return [Step(label, code)]

    def _prompt(self, feedback: str | None) -> str:
        parts = [f"TASK\n{self.task.brief}", MLE_GATE_INSTRUCTIONS]
        if feedback:
            parts.append(
                "Your previous submission was REJECTED. The gate's report "
                f"follows. Fix what it names and resubmit the whole program.\n\n"
                f"{feedback}"
            )
        return "\n\n".join(parts)


def _strip_fences(text: str) -> str:
    """Models emit fenced code however firmly they are asked not to."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("```")
    return body[1].split("\n", 1)[-1] if len(body) > 1 else stripped


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #


@dataclass
class ArmResult:
    """One arm's behaviour over every task and seed."""

    arm: str
    runs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passes(self) -> int:
        return sum(1 for r in self.runs if r["passed"])

    @property
    def pass_rate(self) -> float:
        return self.passes / len(self.runs) if self.runs else 0.0

    @property
    def turns_to_pass(self) -> list[int]:
        return [r["turns"] for r in self.runs if r["passed"]]

    @property
    def mean_turns(self) -> float | None:
        values = self.turns_to_pass
        return statistics.fmean(values) if values else None

    def stuck_on(self) -> dict[str, int]:
        """Check ids that were still failing when a run gave up.

        The diagnostic that drives tuning: a check appearing here repeatedly is
        one whose feedback does not tell the engineer what to do.
        """
        counts: dict[str, int] = {}
        for run in self.runs:
            if run["passed"]:
                continue
            for check in run["last_failed"]:
                counts[check] = counts.get(check, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "runs": len(self.runs),
            "passes": self.passes,
            "pass_rate": round(self.pass_rate, 3),
            "mean_turns_to_pass": (
                round(self.mean_turns, 2) if self.mean_turns is not None else None
            ),
            "turns_to_pass": self.turns_to_pass,
            "stuck_on": self.stuck_on(),
            "model_calls": sum(r["model_calls"] for r in self.runs),
        }


def run_arm(
    model_fn: ModelFn,
    *,
    generated_feedback: bool,
    workdir: str | Path,
    tasks: tuple[Task, ...] = TASKS,
    seeds: int = 3,
    max_attempts: int = 3,
    timeout_s: int = 120,
) -> ArmResult:
    """Run every task `seeds` times under one feedback regime."""
    arm = "generated" if generated_feedback else "template"
    result = ArmResult(arm=arm)
    workdir = Path(workdir)

    for task in tasks:
        for seed in range(seeds):
            scenario = Scenario(
                name=f"{arm}-{task.name}-{seed}",
                summary=task.brief,
                turns=(),  # the live engineer supplies them
                expect_outcome="pass",
                expect_turns=0,
                expected_keys=task.expected_keys,
                num_classes=task.num_classes,
                max_attempts=max_attempts,
            )
            outcome = run_loop(
                scenario,
                workdir=workdir / scenario.name,
                engineer=ModelEngineer(model_fn, task, max_turns=max_attempts),
                # The variable under test. Arm A withholds the model from the
                # gate so render_feedback falls back to the template; arm B
                # supplies it. The engineer model is the same in both.
                consult_model=model_fn if generated_feedback else None,
                counterfactual=False,
                timeout_s=timeout_s,
            )
            result.runs.append(_summarise_run(task, seed, outcome))
    return result


def _summarise_run(task: Task, seed: int, outcome: LoopOutcome) -> dict[str, Any]:
    last = outcome.executions[-1] if outcome.executions else None
    return {
        "task": task.name,
        "seed": seed,
        "passed": outcome.outcome == "pass",
        "turns": outcome.turns_used,
        "executions": len(outcome.executions),
        "last_failed": sorted(last.failed_check_ids) if last else [],
        "model_calls": outcome.model_cost["calls"],
    }


def compare_arms(
    model_fn: ModelFn,
    *,
    workdir: str | Path = "./tuning_runs",
    tasks: tuple[Task, ...] = TASKS,
    seeds: int = 3,
    max_attempts: int = 3,
) -> str:
    """Both arms, and the table that says whether the generator earns its keep."""
    template = run_arm(
        model_fn,
        generated_feedback=False,
        workdir=Path(workdir) / "template",
        tasks=tasks,
        seeds=seeds,
        max_attempts=max_attempts,
    )
    generated = run_arm(
        model_fn,
        generated_feedback=True,
        workdir=Path(workdir) / "generated",
        tasks=tasks,
        seeds=seeds,
        max_attempts=max_attempts,
    )
    return render_comparison(template, generated)


def render_comparison(template: ArmResult, generated: ArmResult) -> str:
    rows = [
        "FEEDBACK ARM COMPARISON",
        "",
        f"  {'ARM':<12} {'RUNS':>5} {'PASSED':>7} {'PASS RATE':>10} {'MEAN TURNS':>11} {'CALLS':>7}",
        "  " + "-" * 60,
    ]
    for arm in (template, generated):
        mean = f"{arm.mean_turns:.2f}" if arm.mean_turns is not None else "—"
        rows.append(
            f"  {arm.arm:<12} {len(arm.runs):>5} {arm.passes:>7} "
            f"{arm.pass_rate:>9.1%} {mean:>11} {sum(r['model_calls'] for r in arm.runs):>7}"
        )

    rows += ["", "  STUCK ON (check ids still failing when a run gave up)"]
    for arm in (template, generated):
        stuck = arm.stuck_on()
        detail = ", ".join(f"{k} x{v}" for k, v in stuck.items()) or "—"
        rows.append(f"    {arm.arm:<12} {detail}")

    rows += [
        "",
        "  The confound to state alongside this: the generated arm spends an",
        "  extra model call per rejection, so any convergence gain is bought",
        "  with latency and tokens. Report both or neither.",
        "",
    ]
    return "\n".join(rows)
