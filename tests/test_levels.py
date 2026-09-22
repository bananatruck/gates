"""GATES_LEVEL: the cumulative switch that picks which gates run (D58).

Each level adds one gate to the level below it, because each gate reads what the
one before it produced: Gate 2 reviews Gate 1's registry, and Gate 3 judges a
manuscript against it. So there are four arms and no others.

    0  all off          the host exactly as shipped
    1  Gate 1           2 and 3 closed
    2  Gates 1 + 2      3 closed
    3  Gates 1 + 2 + 3  all on

``GATES_GATE1=off`` predates this and still means level 0, because the published
Gate 1 evidence and the runner that produced it set it.
"""

import pytest

from gates import GateError
from gates.pipeline import gate_level


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("GATES_LEVEL", raising=False)
    monkeypatch.delenv("GATES_GATE1", raising=False)


@pytest.mark.parametrize("value, level", [("0", 0), ("1", 1), ("2", 2), ("3", 3), (" 2 ", 2)])
def test_the_level_is_read_from_the_environment(monkeypatch, value, level):
    monkeypatch.setenv("GATES_LEVEL", value)
    assert gate_level() == level


def test_every_gate_is_on_unless_a_level_says_otherwise():
    assert gate_level() == 3


def test_gates_gate1_off_still_means_level_0(monkeypatch):
    monkeypatch.setenv("GATES_GATE1", "off")
    assert gate_level() == 0


@pytest.mark.parametrize("value", ["4", "-1", "all", "1+2", ""])
def test_a_level_that_is_not_an_arm_is_refused(monkeypatch, value):
    """A typo must not quietly run a different arm than the one on the label."""
    monkeypatch.setenv("GATES_LEVEL", value)
    with pytest.raises(GateError, match="GATES_LEVEL"):
        gate_level()


def test_setting_both_switches_is_refused(monkeypatch):
    """The Gate 1 runner sets GATES_GATE1 per arm. A GATES_LEVEL left in the shell
    would otherwise decide the arm without the runner's record saying so."""
    monkeypatch.setenv("GATES_GATE1", "on")
    monkeypatch.setenv("GATES_LEVEL", "1")
    with pytest.raises(GateError, match="not both"):
        gate_level()


# --------------------------------------------------------------------------- #
# a closed gate does not run
# --------------------------------------------------------------------------- #

REGISTRY = {
    "gate": "GATE 1 — EXECUTION VALIDITY",
    "verdict": "PASS",
    "citable": True,
    "values": {"exp1.acc": {"value": 0.97, "unit": "ratio", "trace_id": "t1"}},
}


def _never(*args, **kwargs):
    raise AssertionError("a closed gate asked the agent for work")


def _gate1(tmp_path):
    from gates import pipeline
    from gates.adapters.agentlab import make_context

    return lambda: pipeline.gated_execute("x = 1\n", make_context(research_dir=str(tmp_path)))


def _gate2(tmp_path):
    from gates import pipeline
    from gates.adapters.agentlab import make_context, make_review_context

    return lambda: pipeline.review_loop(
        make_review_context(research_dir=str(tmp_path)),
        _never,
        gate1=make_context(research_dir=str(tmp_path)),
    )


def _gate3(tmp_path):
    from gates import pipeline
    from gates.adapters.agentlab import make_report_context

    return lambda: pipeline.report_loop(
        make_report_context(research_dir=str(tmp_path), sections=()),
        _never,
        registry=REGISTRY,
    )


@pytest.mark.parametrize(
    "level, gate",
    [("0", _gate1), ("0", _gate2), ("1", _gate2), ("0", _gate3), ("1", _gate3), ("2", _gate3)],
)
def test_a_gate_above_the_level_refuses_to_run(monkeypatch, tmp_path, level, gate):
    monkeypatch.setenv("GATES_LEVEL", level)
    with pytest.raises(GateError, match=f"GATES_LEVEL={level}"):
        gate(tmp_path)()


def test_the_single_review_and_report_calls_obey_the_level_too(monkeypatch, tmp_path):
    """A host may call these without the loops, so the guard sits here as well."""
    from gates import pipeline
    from gates.adapters.agentlab import make_report_context, make_review_context

    monkeypatch.setenv("GATES_LEVEL", "1")
    with pytest.raises(GateError, match="Gate 2"):
        pipeline.gated_review(REGISTRY, make_review_context(research_dir=str(tmp_path)))
    monkeypatch.setenv("GATES_LEVEL", "2")
    with pytest.raises(GateError, match="Gate 3"):
        pipeline.gated_report(
            "x", REGISTRY, make_report_context(research_dir=str(tmp_path), sections=())
        )


def test_level_0_runs_the_hosts_own_path_through_the_adapter(monkeypatch, tmp_path):
    """The adapter, not the pipeline, knows what 'the host as shipped' means."""
    from gates.adapters.agentlab import gated_execute, make_context

    monkeypatch.setenv("GATES_LEVEL", "0")
    run = gated_execute("print('hi')\n", make_context(research_dir=str(tmp_path)))
    assert run.report is None and run.passed
