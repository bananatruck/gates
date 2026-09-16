"""Gate 2's feedback loop, tier C, held by test.

``tests/test_gate2.py`` holds each check in isolation. This file holds the loop
they run inside: the engineer submits code, Gate 1 runs it, Gate 2 rejects the
registry that run wrote, the feedback goes back, the budget advances by one agent
turn, and the phase ends in the state the design says it should.

Gate 2's terminal state is the one that differs from Gate 1's, so it is tested
first. A spent budget does not raise. The run proceeds, and what Gate 2 could not
get fixed travels to the writer as a declared limitation.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import Ledger  # noqa: E402
from rig.gate2_loop import (  # noqa: E402
    ScriptedEngineer,
    check_expectations,
    main as gate2_loop_main,
    run_gate2_loop,
)
from rig.gate2_scenarios import SCENARIOS  # noqa: E402


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """Play every scenario once; the assertions below read the results."""
    root = tmp_path_factory.mktemp("gate2_loop")
    return {
        name: run_gate2_loop(scenario, workdir=root / name)
        for name, scenario in SCENARIOS.items()
    }


def test_an_unresolved_divergence_proceeds_declared_when_the_budget_is_spent(played):
    """Scenario 5, the terminal state a reviewer will look for.

    Gate 1 raises here and Gate 3 will too. Gate 2 must not, because a genuine
    novel result cannot be blocked forever. It must not go quiet either: a
    divergence dropped on exhaustion would be absent-never-green broken at the
    exit of the gate, so the divergence reaches the writer with its source span.
    """
    scenario = SCENARIOS["divergence-exhausts"]
    outcome = played["divergence-exhausts"]

    assert outcome.outcome == "proceeded"
    assert outcome.turns_used == scenario.max_attempts
    assert outcome.turns[-1].rejections_after == scenario.max_attempts
    assert not any(turn.passed for turn in outcome.turns)
    assert "DECLARED LIMITATIONS" in outcome.declared
    assert "the plan declared 0.001 and the run recorded 0.01" in outcome.declared
    assert "plan L4" in outcome.declared


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_behaves_as_documented(name, played):
    problems = check_expectations(SCENARIOS[name], played[name])
    assert not problems, "\n".join(problems)


def test_a_clean_run_passes_first_time_and_is_sent_no_feedback(tmp_path):
    scenario = SCENARIOS["clean"]
    engineer = ScriptedEngineer(scenario)
    outcome = run_gate2_loop(scenario, workdir=tmp_path, engineer=engineer)

    assert outcome.outcome == "pass"
    assert outcome.turns_used == 1
    assert engineer.feedback_seen == []
    assert outcome.declared == ""


def test_engineer_receives_the_feedback_report(tmp_path):
    """The fix on turn 2 is only possible if turn 1's rejection named the value."""
    scenario = SCENARIOS["out-of-range"]
    engineer = ScriptedEngineer(scenario)
    run_gate2_loop(scenario, workdir=tmp_path, engineer=engineer)

    assert len(engineer.feedback_seen) == 1
    assert "coherence.range_valid" in engineer.feedback_seen[0]
    assert "exp1.acc = 1.4" in engineer.feedback_seen[0]


def test_an_unverifiable_plan_warns_proceeds_and_reaches_the_writer(played):
    """Scenario 4. D17 made divergence FAIL, so this is Gate 2's WARN path.

    Neither field can be checked: one is recorded but never read by the run (the
    B8 decoy) and one was never recorded. That blocks nothing and hides nothing.
    """
    outcome = played["unverifiable-plan"]
    report = outcome.turns[0].report

    assert outcome.outcome == "pass"
    assert outcome.turns_used == 1
    assert "coherence.method_traceable" in {c.id for c in report.warnings()}
    assert "never reads it" in outcome.declared
    assert "never recorded it" in outcome.declared


def test_rejection_feedback_names_a_fix(played):
    """Every rejection must hand the engineer something it can act on."""
    for outcome in played.values():
        for turn in outcome.turns:
            if not turn.passed:
                assert "REQUIRED FIXES" in turn.feedback, turn.label


def test_a_passing_turn_clears_the_rejection_count(played):
    assert [t.rejections_after for t in played["out-of-range"].turns] == [1, 0]


def test_every_review_lands_in_the_ledger(played):
    for name, outcome in played.items():
        rows = [r for r in Ledger(outcome.ledger_path).rows() if "turn" in r]
        assert len(rows) == outcome.reviews_used, name
        assert {row["scenario"] for row in rows} == {name}
        assert [row["verdict"] for row in rows] == [
            turn.report.verdict.value for turn in outcome.turns if turn.gate == 2
        ]


def test_every_gate_1_run_in_the_loop_lands_in_the_ledger(played):
    """The ledger records every attempt, and a revision is a Gate 1 attempt
    before it is anything else. Those rows carry ``review_turn``, not ``turn``,
    so ``loop_summary`` counts Gate 2 reviews only."""
    for name, outcome in played.items():
        rows = [r for r in Ledger(outcome.ledger_path).rows() if "review_turn" in r]
        assert len(rows) == outcome.turns_used, name
        assert {row["gate"] for row in rows} == {"GATE 1 — EXECUTION VALIDITY"}
        assert {row["scenario"] for row in rows} == {name}


def test_loop_stops_when_the_engineer_gives_up(tmp_path):
    class GivesUp:
        def turn(self, feedback, turn_index):
            return None

    outcome = run_gate2_loop(
        SCENARIOS["divergence-exhausts"], workdir=tmp_path, engineer=GivesUp()
    )
    assert outcome.turns_used == 0
    assert outcome.outcome == "no_pass"


def test_json_cli_output_is_machine_parseable(capsys):
    assert gate2_loop_main(["divergence-exhausts", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenarios"][0]["outcome"] == "proceeded"


def test_giving_up_after_a_rejection_still_declares_it(tmp_path):
    """F4. An engineer that stops before the budget does must not drop the
    divergence it was sent: the writer still has to disclose it."""
    scenario = SCENARIOS["divergence-exhausts"]

    class StopsAfterOne:
        def turn(self, feedback, turn_index):
            return scenario.turns[0] if turn_index == 0 else None

    outcome = run_gate2_loop(scenario, workdir=tmp_path, engineer=StopsAfterOne())
    assert outcome.outcome == "no_pass"
    assert outcome.turns_used == 1
    assert "the plan declared 0.001 and the run recorded 0.01" in outcome.declared


def test_loop_metrics_come_from_the_ledger_alone(tmp_path):
    """F7, spec M5. A run that passes its first review never entered the loop,
    so it counts in neither side of resolution_rate: counting it would inflate
    the rate with runs the loop did nothing for."""
    for name, scenario in SCENARIOS.items():
        outcome = run_gate2_loop(scenario, workdir=tmp_path)

    summary = Ledger(outcome.ledger_path).loop_summary()
    assert summary == {
        "runs_reviewed": 6,
        "runs_entering_loop": 4,
        "resolved_within_budget": 3,
        "resolution_rate": 3 / 4,
        "mean_attempts": 2.0,
        "unresolved_declared": 1,
        "abandoned": 0,
    }


def test_a_revision_runs_under_gate_1_before_gate_2_sees_it(tmp_path):
    """F12. A revision is code, and Gate 2 only ever reviews the registry Gate 1
    built from running it. A number typed into the fix is Gate 1's to reject, so
    Gate 2 never reviews it and the engineer is sent Gate 1's report."""
    from gates.adapters.agentlab import make_context, make_review_context, review_loop

    typed = 'record_metadata("seed", 0)\nrecord_result("exp1.acc", 0.812, unit="ratio")\n'
    sent: list = []

    def revise(feedback):
        sent.append(feedback)
        return typed if len(sent) == 1 else None

    outcome = review_loop(
        make_review_context(research_dir=str(tmp_path)),
        revise,
        gate1=make_context(research_dir=str(tmp_path)),
    )
    assert outcome.reviews == []
    assert outcome.outcome == "no_pass"
    assert [e.passed for e in outcome.executions] == [False]
    assert "results.values_computed" in sent[1]


def test_a_hand_typed_fix_is_gate_1s_to_reject_and_costs_no_review(played):
    """F12 as a scenario. After a speedup rejection, the engineer types the
    right number into the ``record_result`` call. Gate 1 rejects it, Gate 2
    never reviews it, and Gate 2's budget still has the turn the recomputed run
    then passes on."""
    outcome = played["hand-typed-fix"]

    assert [t.gate for t in outcome.turns] == [2, 1, 2]
    assert "results.values_computed" in {c.id for c in outcome.turns[1].report.failed_checks()}
    assert [t.rejections_after for t in outcome.turns] == [1, 1, 0]
    assert outcome.outcome == "pass"
    assert outcome.reviews_used == SCENARIOS["hand-typed-fix"].max_attempts


def _typed_speedup():
    from rig.gate2_scenarios import HAND_TYPED_FIX

    return HAND_TYPED_FIX.turns[1].code()


def test_gate_1_exhausted_with_nothing_run_raises(tmp_path):
    """Gate 1's policy holds inside Gate 2's loop: a run that never produced a
    valid experiment must not produce a paper."""
    from gates import GateFailure
    from gates.adapters.agentlab import make_context, make_review_context, review_loop

    with pytest.raises(GateFailure):
        review_loop(
            make_review_context(research_dir=str(tmp_path)),
            lambda feedback: _typed_speedup(),
            gate1=make_context(research_dir=str(tmp_path), max_attempts=2),
        )


def test_gate_1_exhausted_after_a_review_proceeds_with_that_review_declared(tmp_path):
    """Something passed Gate 1 and Gate 2 rejected it. If every fix after that is
    rejected by Gate 1, the loop stops rather than asking forever, and the
    writer still gets the discrepancy Gate 2 found."""
    from gates.adapters.agentlab import make_context, make_review_context, review_loop
    from rig.gate2_scenarios import DIVERGENCE_EXHAUSTS, LR, EPOCHS, SPEEDUP

    calls = []

    def revise(feedback):
        calls.append(feedback)
        if len(calls) > 5:
            raise AssertionError("revise called after Gate 1's budget was spent")
        return DIVERGENCE_EXHAUSTS.turns[0].code() if len(calls) == 1 else _typed_speedup()

    outcome = review_loop(
        make_review_context(
            research_dir=str(tmp_path), max_attempts=3, relations=(SPEEDUP,),
            plan_fields=(LR, EPOCHS),
        ),
        revise,
        gate1=make_context(research_dir=str(tmp_path), max_attempts=2),
    )
    assert len(calls) == 3
    assert len(outcome.reviews) == 1
    assert outcome.outcome == "proceeded"
    assert "the plan declared 0.001 and the run recorded 0.01" in outcome.declared


def test_the_run_the_solver_already_passed_is_reviewed_without_rerunning(tmp_path):
    """A host leaves the experiment phase holding a Gate 1 pass. Reviewing it
    must not execute it a second time; revise is first asked for a fix."""
    from gates.adapters.agentlab import (
        gated_execute, make_context, make_review_context, review_loop,
    )
    from rig.gate2_scenarios import CLEAN, OUT_OF_RANGE, SPEEDUP

    gate1 = make_context(research_dir=str(tmp_path))
    passed = gated_execute(OUT_OF_RANGE.turns[0].code(), gate1)
    sent = []

    def revise(feedback):
        sent.append(feedback)
        return OUT_OF_RANGE.turns[1].code()

    outcome = review_loop(
        make_review_context(research_dir=str(tmp_path), relations=(SPEEDUP,)),
        revise,
        gate1=gate1,
        first=passed,
    )
    assert outcome.outcome == "pass"
    assert len(outcome.reviews) == 2
    assert len(outcome.executions) == 1
    assert gate1.attempt == 2
    assert "coherence.range_valid" in sent[0]


def test_a_first_run_gate_1_rejected_cannot_be_reviewed(tmp_path):
    from gates import GateError
    from gates.adapters.agentlab import (
        gated_execute, make_context, make_review_context, review_loop,
    )

    gate1 = make_context(research_dir=str(tmp_path))
    rejected = gated_execute(_typed_speedup(), gate1)
    with pytest.raises(GateError):
        review_loop(
            make_review_context(research_dir=str(tmp_path)),
            lambda feedback: None,
            gate1=gate1,
            first=rejected,
        )


def test_the_outcome_names_the_registry_the_writer_cites(tmp_path):
    """Gate 3 checks the manuscript against one registry. It is the one Gate 2
    last reviewed: the admitted run on a pass, the declared one on proceed, and
    none when nothing was reviewed."""
    from gates.adapters.agentlab import make_context, make_review_context, review_loop
    from rig.gate2_scenarios import OUT_OF_RANGE, SPEEDUP

    turns = iter(t.code() for t in OUT_OF_RANGE.turns)
    gate1 = make_context(research_dir=str(tmp_path))
    outcome = review_loop(
        make_review_context(research_dir=str(tmp_path), relations=(SPEEDUP,)),
        lambda feedback: next(turns, None),
        gate1=gate1,
    )
    assert outcome.registry["values"]["exp1.acc"]["value"] == 0.812
    assert outcome.registry["run"]["run_id"] == gate1.last_report.execution.run_id

    nothing = review_loop(
        make_review_context(research_dir=str(tmp_path)),
        lambda feedback: None,
        gate1=make_context(research_dir=str(tmp_path)),
    )
    assert nothing.registry is None


def test_a_proceeded_run_cites_the_registry_it_declared(played):
    registry = played["divergence-exhausts"].registry
    assert registry["values"]["config.lr"]["value"] == 0.01
