"""Scripted ML-engineer turns, one per failure mode Gate 1 claims to catch.

Each scenario stands in for an ML engineer agent: a fixed sequence of code
submissions, played back against the real gate and the real adapter loop. No
model is called, so the loop is deterministic and free to run.

The fixtures use no third-party imports. Two reasons: the rig must run in any
environment the gate installs into, and a fixture that fails on a missing
``torch`` would be testing the environment rather than the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# the pieces of code an engineer submits
# --------------------------------------------------------------------------- #

#: The archived run's exact failure shape: a name read inside a method and bound
#: in no enclosing scope. Rejected by the static tier, before anything executes.
UNBOUND_NAME = '''\
class GCN:
    """Two-layer GCN. The reshape reads a width that is bound nowhere."""

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        return [v * hidden_dim for v in x]


model = GCN(1433, 7)
logits = model.forward([0.4, 0.6, 0.2])
test_acc = sum(logits) / len(logits)
record_result("exp1.K2.test_acc", test_acc, unit="ratio")
'''

#: Prints well past the old 1000-character ceiling, then dies. Upstream appended
#: its crash marker after these prints and then sliced [:1000], so this exact
#: shape was reported as a success and scored 1.0.
CRASH_AFTER_HEAVY_OUTPUT = '''\
import random

random.seed(0)
for epoch in range(60):
    loss = 1.9243 - epoch * 0.021
    print(f"epoch {epoch:03d}  loss {loss:.4f}  train_acc 0.4120  val_acc 0.3980")

test_acc = correct / total
record_result("exp1.K2.test_acc", test_acc, unit="ratio")
'''

#: Statically clean, exits 0, prints its numbers — and records none of them.
#: This is the shape of the saved run_experiments.py in the archived run: the
#: paper's numbers were printed, so not one of them is citable.
NO_CONTRACT = '''\
import random

random.seed(0)
correct, total = 408, 500
test_acc = correct / total
sgc_wallclock = 0.0180
gcn_wallclock = 0.2450

print(f"Final test accuracy: {test_acc:.4f}")
print(f"SGC {sgc_wallclock:.4f}s   GCN {gcn_wallclock:.4f}s")
print(f"Speedup: {gcn_wallclock / sgc_wallclock:.2f}x")
'''

#: The numbers from the archived paper, typed in directly. Every downstream
#: consistency check passes: they are mutually consistent, in range, finite,
#: non-degenerate. Only the call site shows they were never measured.
HARDCODED_LITERALS = '''\
record_metadata("seed", 0)

test_acc = 0.4
record_result("exp1.K2.test_acc", 0.816, unit="ratio")
record_result("exp2.sgc.wallclock_s", 0.0180, unit="seconds")
record_result("exp2.speedup", 13.61, unit="ratio")
'''

#: Syntactically broken — what an inner repair loop sees on its first pass.
SYNTAX_ERROR = '''\
def train(model, epochs:
    for epoch in range(epochs):
        pass
'''

#: Repaired syntax, still crashes: the second execution inside one engineer turn.
REPAIRED_BUT_CRASHES = '''\
def train(epochs):
    history = []
    for epoch in range(epochs):
        history.append(1.0 / (epoch - 3))
    return history


record_metadata("seed", 0)
history = train(10)
record_result("exp1.final_loss", history[-1])
'''

#: Clean: seeded, measured, recorded, every declared key present.
CLEAN_RUN = '''\
import random
import time

seed = 0
random.seed(seed)
record_metadata("seed", seed)

correct, total = 408, 500
test_acc = correct / total

start = time.perf_counter()
propagated = [sum(random.random() for _ in range(8)) for _ in range(2708)]
sgc_wallclock = time.perf_counter() - start

start = time.perf_counter()
deep = [sum(random.random() for _ in range(64)) for _ in range(2708)]
gcn_wallclock = time.perf_counter() - start

print(f"propagated {len(propagated)} nodes, deep {len(deep)} nodes")
print(f"Final test accuracy: {test_acc:.4f}")

record_result("exp1.K2.test_acc", test_acc, unit="ratio")
record_result("exp2.sgc.wallclock_s", sgc_wallclock, unit="seconds")
record_result("exp2.speedup", gcn_wallclock / sgc_wallclock, unit="ratio")
'''

#: Exits 0 and records a real measurement, but the logs say the run was in
#: trouble and no seed was declared. Every finding here is WARN: the run is
#: admitted, and the warnings travel with it into the evidence bundle.
WARNS_BUT_PASSES = '''\
import traceback

try:
    ratio = 1 / 0
except ZeroDivisionError:
    traceback.print_exc()

print("RuntimeWarning: invalid value encountered in true_divide")
print("CUDA unavailable, falling back to CPU")

zero = 0.0
record_result("exp1.K2.test_acc", zero * 1.0, unit="ratio")
'''

#: Turn one of the namespace-leak pair: binds a name and passes.
LEAK_BINDS = '''\
record_metadata("seed", 0)
leaked_accuracy = 0.816
record_result("exp1.K2.test_acc", leaked_accuracy * 1.0, unit="ratio")
'''

#: Turn two: reads the name the previous, *successful* execution bound. Under
#: exec(code, globals()) this passed and reported a number from a run that never
#: computed it. In a fresh process it is a NameError.
LEAK_READS = '''\
record_metadata("seed", 0)
inherited = leaked_accuracy
record_result("exp1.K2.test_acc", inherited * 1.0, unit="ratio")
'''


# --------------------------------------------------------------------------- #
# scenario model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Step:
    """One execution. Several may belong to a single engineer turn."""

    label: str
    code: str
    #: Check ids that must appear among the blocking failures.
    expect_fail: tuple[str, ...] = ()
    #: Check ids that must appear among the warnings.
    expect_warn: tuple[str, ...] = ()
    expect_pass: bool = False


@dataclass(frozen=True)
class Turn:
    """One ML engineer turn, and every execution the scaffold spent inside it.

    More than one step means an inner automated-repair loop ran. The budget must
    charge the turn once regardless — that distinction is the adapter contract's
    third rule, and it was a real defect before it was written down.
    """

    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Scenario:
    name: str
    summary: str
    turns: tuple[Turn, ...]
    #: "pass" — an attempt was admitted. "gate_failure" — budget spent, nothing
    #: ever passed, so GateFailure is raised and no paper is produced.
    expect_outcome: str
    #: Engineer turns the loop should consume before it stops.
    expect_turns: int
    expected_keys: tuple[str, ...] | None = None
    num_classes: int | None = None
    max_attempts: int = 3
    #: Keep playing turns after one passes. Only the namespace-leak
    #: demonstration needs this — it is not a loop any scaffold runs.
    continue_after_pass: bool = False
    notes: str = ""
    #: Turn indices (0-based) whose failure the upstream detector would have
    #: missed. Asserted by the rig, so the counterfactual is held by test rather
    #: than narrated.
    expect_upstream_blind_turns: tuple[int, ...] = field(default_factory=tuple)


def _turn(*steps: Step) -> Turn:
    return Turn(steps=steps)


ARCHIVED_RUN = Scenario(
    name="archived-run",
    summary="Replays results/gemini_3_5_flash_run_1. Every turn fails; no paper.",
    turns=(
        _turn(
            Step(
                "NameError inside forward()",
                UNBOUND_NAME,
                expect_fail=("static.no_unbound_names",),
            )
        ),
        _turn(
            Step(
                "crash after 3,000 characters of epoch logging",
                CRASH_AFTER_HEAVY_OUTPUT,
                expect_fail=(
                    "exec.exit_code_zero",
                    "exec.no_uncaught_exception",
                    "results.contract_present",
                ),
            )
        ),
        _turn(
            Step(
                "runs clean, prints the numbers, records nothing",
                NO_CONTRACT,
                expect_fail=("results.contract_present",),
            )
        ),
    ),
    expect_outcome="gate_failure",
    expect_turns=3,
    expected_keys=("exp1.K2.test_acc",),
    notes=(
        "The run this repository audited produced a paper reporting 81.60% test "
        "accuracy and a 13.61x speedup. Gate 1 rejects all three of its shapes, "
        "on three different checks, and the run exits non-zero with no paper."
    ),
    expect_upstream_blind_turns=(1, 2),
)

RECOVERS = Scenario(
    name="recovers",
    summary="Two rejections, then a clean run. The loop's intended path.",
    turns=(
        _turn(
            Step(
                "NameError inside forward()",
                UNBOUND_NAME,
                expect_fail=("static.no_unbound_names",),
            )
        ),
        _turn(
            Step(
                "reads the feedback, invents the numbers instead",
                HARDCODED_LITERALS,
                expect_fail=("results.values_computed",),
            )
        ),
        _turn(
            Step(
                "measures and records every declared key",
                CLEAN_RUN,
                expect_pass=True,
            )
        ),
    ),
    expect_outcome="pass",
    expect_turns=3,
    expected_keys=("exp1.K2.test_acc", "exp2.sgc.wallclock_s", "exp2.speedup"),
    notes=(
        "Turn 2 is the interesting one: the agent has been told the run must "
        "produce numbers, so it produces numbers. Every value is in range, "
        "finite, mutually consistent and non-degenerate. Only the call site "
        "shows they were typed rather than measured. Note that upstream's "
        "detector catches turn 1 and not turn 2: an unbound name crashes "
        "immediately, so its marker survives the slice, whereas invented "
        "numbers exit 0 and produce no marker at all."
    ),
    expect_upstream_blind_turns=(1,),
)

INNER_REPAIR = Scenario(
    name="inner-repair",
    summary="Three executions inside one engineer turn. The budget charges one.",
    turns=(
        _turn(
            Step(
                "submitted with a syntax error",
                SYNTAX_ERROR,
                expect_fail=("static.syntax_valid",),
            ),
            Step(
                "repair loop fixes the syntax; ZeroDivisionError remains",
                REPAIRED_BUT_CRASHES,
                expect_fail=("exec.exit_code_zero", "exec.no_uncaught_exception"),
            ),
            Step(
                "repair loop fixes the arithmetic",
                CLEAN_RUN,
                expect_pass=True,
            ),
        ),
    ),
    expect_outcome="pass",
    expect_turns=1,
    expected_keys=("exp1.K2.test_acc", "exp2.sgc.wallclock_s", "exp2.speedup"),
    notes=(
        "Denominating the budget in executions rather than agent turns spends "
        "all three retries here before the ML engineer gets a single turn. Any "
        "scaffold with an inner repair loop hits this."
    ),
)

WARN_TIER = Scenario(
    name="warn-tier",
    summary="Passes, loudly. Four warnings ride along into the evidence bundle.",
    turns=(
        _turn(
            Step(
                "caught exception, NaN warning, CPU fallback, no seed",
                WARNS_BUT_PASSES,
                expect_pass=True,
                expect_warn=(
                    "exec.no_swallowed_traceback",
                    "logs.no_error_signals",
                    "results.non_degenerate",
                    "env.seed_recorded",
                ),
            )
        ),
    ),
    expect_outcome="pass",
    expect_turns=1,
    expected_keys=("exp1.K2.test_acc",),
    num_classes=7,
    notes=(
        "The severity split, exercised. None of these block: an exact zero can "
        "be a real measurement, and rejecting on one would cost the agent a "
        "rewrite for nothing. All four are carried into the report the writer "
        "receives, under a heading that says they must be stated."
    ),
)

NAMESPACE_LEAK = Scenario(
    name="namespace-leak",
    summary="A name bound by a passing run is gone by the next one.",
    turns=(
        _turn(Step("binds leaked_accuracy, passes", LEAK_BINDS, expect_pass=True)),
        _turn(
            Step(
                "reads the previous run's variable",
                LEAK_READS,
                expect_fail=(
                    "exec.exit_code_zero",
                    "exec.no_uncaught_exception",
                    "results.contract_present",
                ),
            )
        ),
    ),
    expect_outcome="pass",
    expect_turns=2,
    expected_keys=("exp1.K2.test_acc",),
    continue_after_pass=True,
    notes=(
        "Upstream ran exec(code_str, globals()) into tools.py's own never-"
        "cleared module namespace, so turn 2 succeeded and reported 0.816 from "
        "a run that never computed it. Note that the static tier does not claim "
        "this one: a module-level read is not decidable without ordering "
        "analysis, so the runtime tier catches it one execution later. The "
        "channel counterfactual is deliberately not asserted here — upstream's "
        "failure on this shape came from the shared namespace, not from the "
        "1,000-character slice, and re-running turn 2 in a fresh process "
        "reproduces the gate's behaviour rather than upstream's."
    ),
)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (ARCHIVED_RUN, RECOVERS, INNER_REPAIR, WARN_TIER, NAMESPACE_LEAK)
}
