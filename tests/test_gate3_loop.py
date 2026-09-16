"""Gate 3's feedback loop, held by test.

``tests/test_gate3.py`` holds each check, and ``report_loop`` over hand-built
registries. This file holds the loop the way a host runs it: the registry comes
from a run Gates 1 and 2 admitted, the writer submits a manuscript, Gate 3
rejects it, the feedback goes back, and the phase ends in the state the design
says it should.

Gate 3's terminal state is the one that differs from Gate 2's, so it is tested
first. A spent budget raises, and no manuscript comes out.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import Ledger  # noqa: E402
from gates.gate1 import GATE_NAME as GATE1  # noqa: E402
from gates.gate2 import GATE_NAME as GATE2  # noqa: E402
from gates.gate3 import GATE_NAME as GATE3  # noqa: E402
from rig.gate3_loop import (  # noqa: E402
    ScriptedWriter,
    check_expectations,
    main as gate3_loop_main,
    run_gate3_loop,
)
from rig.gate3_scenarios import SCENARIOS  # noqa: E402


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """Play every scenario once; the assertions below read the results."""
    root = tmp_path_factory.mktemp("gate3_loop")
    return {
        name: run_gate3_loop(scenario, workdir=root / name)
        for name, scenario in SCENARIOS.items()
    }


def test_a_spent_budget_raises_and_no_manuscript_comes_out(tmp_path):
    """Scenario 6. Gate 2 proceeds here; Gate 3 must not, because an
    unverifiable manuscript is not emitted (`CLAUDE.md` §4). The script has a
    turn left over, so the budget stopped the loop, not the script."""
    scenario = SCENARIOS["budget-exhausts"]
    outcome = run_gate3_loop(scenario, workdir=tmp_path)

    assert len(scenario.turns) > scenario.max_attempts
    assert outcome.outcome == "raised"
    assert outcome.manuscript is None
    assert outcome.turns_used == scenario.max_attempts
    assert not any(turn.passed for turn in outcome.turns)
    assert "report.no_numeric_literals_in_results" in outcome.turns[-1].feedback


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_behaves_as_documented(name, played):
    problems = check_expectations(SCENARIOS[name], played[name])
    assert not problems, "\n".join(problems)


def test_a_clean_manuscript_passes_first_time_and_comes_out_rendered(tmp_path):
    """Scenario 1. The writer is sent no feedback, and what comes out is what the
    gate judged: tokens replaced by the values the clean run recorded."""
    scenario = SCENARIOS["clean"]
    writer = ScriptedWriter(scenario)
    outcome = run_gate3_loop(scenario, workdir=tmp_path, writer=writer)

    assert outcome.outcome == "pass"
    assert outcome.turns_used == 1
    assert writer.feedback_seen == []
    assert "\\result{" not in outcome.manuscript
    assert "test accuracy of 0.812" in outcome.manuscript
    assert "13.611 times faster" in outcome.manuscript


def test_the_writer_is_sent_the_number_it_typed(tmp_path):
    """Scenario 2. The fix on turn 2 is only possible if turn 1's rejection
    named the number and said what to write instead."""
    scenario = SCENARIOS["typed-literal-fixed"]
    writer = ScriptedWriter(scenario)
    outcome = run_gate3_loop(scenario, workdir=tmp_path, writer=writer)

    assert outcome.outcome == "pass"
    assert len(writer.feedback_seen) == 1
    assert "report.no_numeric_literals_in_results" in writer.feedback_seen[0]
    assert "0.812" in writer.feedback_seen[0]
    assert "REQUIRED FIXES" in writer.feedback_seen[0]


def test_a_token_nobody_recorded_is_named_in_the_feedback(played):
    """Scenario 3. ``exp1.f1`` is not in the clean run's registry, so the writer
    is told which key is missing and which keys it may cite instead."""
    turn = played["unknown-token"].turns[0]

    assert "report.all_tokens_resolve" in {c.id for c in turn.report.failed_checks()}
    assert "missing:  exp1.f1" in turn.feedback
    assert "exp1.acc" in turn.feedback


def test_every_turn_lands_in_the_ledger_as_report_writing(played):
    """The ledger records every attempt, raised or not."""
    for name, outcome in played.items():
        rows = [r for r in Ledger(outcome.ledger_path).rows() if r["gate"] == GATE3]
        assert len(rows) == outcome.turns_used, name
        assert {row["phase"] for row in rows} == {"report writing"}
        assert {row["scenario"] for row in rows} == {name}
        assert [row["verdict"] for row in rows] == [
            turn.report.verdict.value for turn in outcome.turns
        ]


def test_the_registry_came_from_a_run_gates_1_and_2_admitted(played):
    """F12, one gate later. Gate 3 checks a table a run wrote, so the same
    ledger holds the Gate 1 run that wrote it and the Gate 2 review that
    admitted it, both passing."""
    for name, outcome in played.items():
        assert outcome.registry["citable"] is True, name
        verdicts = {
            gate: [r["verdict"] for r in Ledger(outcome.ledger_path).rows() if r["gate"] == gate]
            for gate in (GATE1, GATE2)
        }
        assert verdicts == {GATE1: ["PASS"], GATE2: ["PASS"]}, name


def test_json_cli_output_is_machine_parseable(capsys):
    assert gate3_loop_main(["budget-exhausts", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenarios"][0]["outcome"] == "raised"
    assert payload["scenarios"][0]["problems"] == []
