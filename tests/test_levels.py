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


# --------------------------------------------------------------------------- #
# L0': Gate 1's evidence delivered, its verdict never enforced (09-29 review)
# --------------------------------------------------------------------------- #

#: Prints past upstream's 1,000-character view, records a value, then crashes:
#: level 0 accepts it (the marker falls off the slice), level 1 rejects it.
CRASH_PAST_THE_SLICE = (
    "outcomes = [i < 408 for i in range(500)]\ncorrect, total = sum(outcomes), len(outcomes)\n"
    "record_metadata('seed', 0)\n"
    "record_result('exp1.acc', correct / total, unit='ratio')\n"
    "for i in range(60):\n"
    "    print(f'epoch {i:03d} loss 1.9243 train_acc 0.4120 val_acc 0.3980')\n"
    "raise RuntimeError('diverged at the last step')\n"
)


def _execute(monkeypatch, tmp_path, level):
    from gates.adapters.agentlab import gated_execute, make_context

    monkeypatch.setenv("GATES_LEVEL", level)
    return gated_execute(CRASH_PAST_THE_SLICE, make_context(research_dir=str(tmp_path)))


def test_l0_prime_is_level_0_to_every_gate_question(monkeypatch):
    from gates.pipeline import evidence_only, gate1_enabled, require_gate

    monkeypatch.setenv("GATES_LEVEL", "0d")
    assert gate_level() == 0
    assert not gate1_enabled()
    assert evidence_only()
    with pytest.raises(GateError):
        require_gate(1)


def test_evidence_only_is_off_at_every_other_level(monkeypatch):
    from gates.pipeline import evidence_only

    for level in ("0", "1", "2", "3"):
        monkeypatch.setenv("GATES_LEVEL", level)
        assert not evidence_only()


def test_l0_prime_accepts_by_level_0s_rule_and_delivers_gate_1s_evidence(monkeypatch, tmp_path):
    l0 = _execute(monkeypatch, tmp_path / "l0", "0")
    l0d = _execute(monkeypatch, tmp_path / "l0d", "0d")
    l1 = _execute(monkeypatch, tmp_path / "l1", "1")

    # Acceptance: level 0's rule, so the crash past the slice is accepted at 0 and 0d.
    assert l0.passed and l0d.passed and not l1.passed
    # Gate 1 ran at 0d and would have rejected; nothing enforced it.
    assert l0d.report is not None and not l0d.report.passed
    assert not l0d.enforced
    # The channel: level 0 hands on 1,000 characters, 0d the registry and the crash.
    assert "exp1.acc" not in l0.evidence_bundle
    assert "exp1.acc = 0.816" in l0d.evidence_bundle
    assert "diverged at the last step" in l0d.evidence_bundle
    # No verdict reaches the agent: no rejection banner, no required fixes.
    assert "REJECTED" not in l0d.evidence_bundle
    assert "REQUIRED FIXES" not in l0d.evidence_bundle
    assert l0d.feedback == l0d.evidence_bundle


#: Gate 1 rejects these before anything runs. Level 0 runs them (B22).
UNBOUND_IN_AN_UNCALLED_FUNCTION = (
    "record_metadata('seed', 0)\n"
    "def f():\n"
    "    return hidden_dim\n"
    "acc = sum([1, 0]) / 2\n"
    "print('acc', acc)\n"
    "record_result('exp1.acc', acc)\n"
)
SHADOWED_RECORD_RESULT = (
    "def record_result(k, v, unit=None):\n"
    "    print(k, v)\n"
    "record_metadata('seed', 0)\n"
    "acc = sum([1, 0]) / 2\n"
    "print('acc', acc)\n"
    "record_result('exp1.acc', acc)\n"
)
SYNTAX_ERROR = "record_metadata('seed', 0)\ndef f(\n"


def _run(monkeypatch, tmp_path, level, code, **kwargs):
    from gates.adapters.agentlab import gated_execute, make_context

    monkeypatch.setenv("GATES_LEVEL", level)
    return gated_execute(code, make_context(research_dir=str(tmp_path), **kwargs))


@pytest.mark.parametrize(
    "code",
    [UNBOUND_IN_AN_UNCALLED_FUNCTION, SHADOWED_RECORD_RESULT, SYNTAX_ERROR],
    ids=["unbound-name", "shadowed-record-result", "syntax-error"],
)
def test_l0_prime_accepts_what_level_0_accepts(monkeypatch, tmp_path, code):
    """Acceptance is level 0's rule, including when Gate 1 rejects without running."""
    l0 = _run(monkeypatch, tmp_path / "l0", "0", code)
    l0d = _run(monkeypatch, tmp_path / "l0d", "0d", code)
    assert l0d.passed is l0.passed
    # Gate 1 still produced its report; the bundle is the evidence, not a verdict.
    assert l0d.report is not None
    assert "REJECTED" not in l0d.evidence_bundle
    assert l0d.feedback == l0d.evidence_bundle


def test_a_syntax_error_at_l0_prime_shows_its_traceback(monkeypatch, tmp_path):
    """Level 0 hands on the SyntaxError. L0' must not replace it with '(none recorded)'."""
    l0 = _run(monkeypatch, tmp_path / "l0", "0", SYNTAX_ERROR)
    l0d = _run(monkeypatch, tmp_path / "l0d", "0d", SYNTAX_ERROR)
    assert "SyntaxError" in l0.evidence_bundle
    assert "SyntaxError" in l0d.evidence_bundle
    assert "(none recorded)" not in l0d.evidence_bundle


#: Records a seed and a value, then sleeps past the limit. The kill drops
#: results.json, so the seed warning would be a lie (B23).
KILLED_AT_THE_TIMEOUT = (
    "import time, random\n"
    "record_metadata('seed', 0)\n"
    "record_result('exp1.acc', random.random())\n"
    "print('starting')\n"
    "time.sleep(30)\n"
)


def test_l0_prime_provenance_points_at_results_when_gate_1_rejects_without_running(
    monkeypatch, tmp_path,
):
    """Gate 1's empty registry is not what the supplemental run produced (D83)."""
    l0d = _run(monkeypatch, tmp_path, "0d", UNBOUND_IN_AN_UNCALLED_FUNCTION)
    assert l0d.report is not None and l0d.report.execution is None
    after_provenance = l0d.evidence_bundle.split("PROVENANCE", 1)[1].split("\n\n", 1)[0]
    assert "results.json" in after_provenance
    assert "registry.json" not in after_provenance


def test_l0_prime_says_a_run_was_killed_at_the_timeout(monkeypatch, tmp_path):
    """A timeout is accepted, as at level 0, and the bundle says the run was killed."""
    l0 = _run(monkeypatch, tmp_path / "l0", "0", KILLED_AT_THE_TIMEOUT, timeout_s=3)
    l0d = _run(monkeypatch, tmp_path / "l0d", "0d", KILLED_AT_THE_TIMEOUT, timeout_s=3)
    assert l0.passed and l0d.passed
    assert "starting" in l0.evidence_bundle
    assert "starting" in l0d.evidence_bundle
    assert "killed" in l0d.evidence_bundle
    assert "no seed was declared" not in l0d.evidence_bundle
    # Gate 1 ran to a full report; its verdict is not what the agent is handed.
    assert l0d.report is not None and l0d.report.execution is not None
    assert "REJECTED" not in l0d.evidence_bundle
    assert "REQUIRED FIXES" not in l0d.evidence_bundle
