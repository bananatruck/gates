"""Model-assisted log scanning: the plumbing, the grounding, and the degradation.

What this file can hold is the harness — that a well-behaved model's findings
arrive intact, that a badly-behaved one's are discarded, and that neither can
touch a verdict. What it cannot hold is the model's own precision and recall
against the corpus, which needs a real model and an API key. `rig/corpus.py`
carries the measurement path; the numbers are task #5's, not this file's.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import Gate1Config, ModelLayer, Severity, run_gate1  # noqa: E402
from gates.llm_scan import CHECK_ID, build_check, scan_with_model  # noqa: E402
from gates.log_checks import LogFinding, scan_streams  # noqa: E402
from gates.report import render_feedback  # noqa: E402

STDOUT = "\n".join(
    [
        "Loaded Cora: 2708 nodes, 5429 edges",
        "Error loading graphs-datasets/cora: cannot be accessed",
        "epoch 000  loss 1.9243  train_acc 0.4120",
        "Skipping 3 of 10 folds that raised during fitting",
        "Final test accuracy: 0.8160",
    ]
)


def model_returning(payload: str):
    return lambda prompt, system: payload


@pytest.fixture
def config(tmp_path):
    def _make(**kw):
        kw.setdefault("timeout_s", 30)
        return Gate1Config(artifact_root=str(tmp_path), **kw)

    return _make


# --------------------------------------------------------------------------- #
# well-behaved model
# --------------------------------------------------------------------------- #


def test_findings_carry_the_real_stream_and_line_number():
    layer = ModelLayer(
        model_returning('[{"line": 2, "why": "dataset did not load"}]')
    )
    outcome = scan_with_model(layer, STDOUT, "")
    assert outcome.ok
    assert len(outcome.findings) == 1
    finding = outcome.findings[0]
    assert finding.stream == "stdout"
    assert finding.lineno == 2
    assert "Error loading" in finding.line
    assert finding.note == "dataset did not load"


def test_json_wrapped_in_prose_or_a_code_fence_still_parses():
    layer = ModelLayer(
        model_returning(
            'Here is what I found:\n```json\n[{"line": 4, "why": "replicates dropped"}]\n```\n'
        )
    )
    outcome = scan_with_model(layer, STDOUT, "")
    assert [f.lineno for f in outcome.findings] == [4]


def test_empty_array_means_nothing_to_report():
    outcome = scan_with_model(ModelLayer(model_returning("[]")), STDOUT, "")
    assert outcome.ok
    assert outcome.findings == []
    assert build_check(outcome) is None


def test_lines_the_pattern_set_already_flagged_are_not_resent():
    """Re-reporting a deterministic finding would inflate the model's apparent
    contribution, which is the number this whole exercise exists to measure."""
    seen = {}

    def capture(prompt, system):
        seen["prompt"] = prompt
        return "[]"

    stdout = "RuntimeWarning: invalid value encountered in divide\nall fine here"
    already = scan_streams(stdout, "")
    assert already, "fixture should trip the deterministic tier"
    scan_with_model(ModelLayer(capture), stdout, "", already_flagged=already)
    assert "invalid value encountered" not in seen["prompt"]
    assert "all fine here" in seen["prompt"]


# --------------------------------------------------------------------------- #
# grounding: the model reports lines, it does not invent them
# --------------------------------------------------------------------------- #


def test_an_out_of_range_line_number_is_discarded():
    layer = ModelLayer(model_returning('[{"line": 999, "why": "invented"}]'))
    assert scan_with_model(layer, STDOUT, "").findings == []


def test_a_negative_or_zero_index_is_discarded():
    layer = ModelLayer(model_returning('[{"line": 0}, {"line": -3}]'))
    assert scan_with_model(layer, STDOUT, "").findings == []


def test_duplicate_indices_are_reported_once():
    layer = ModelLayer(
        model_returning('[{"line": 2, "why": "a"}, {"line": 2, "why": "b"}]')
    )
    assert len(scan_with_model(layer, STDOUT, "").findings) == 1


def test_unparseable_output_yields_no_findings_rather_than_a_crash():
    for junk in ("I could not complete this request", "", "[[[", "{not json}"):
        outcome = scan_with_model(ModelLayer(model_returning(junk)), STDOUT, "")
        assert outcome.ok
        assert outcome.findings == []


def test_findings_are_capped():
    every = ",".join(f'{{"line": {i}, "why": "x"}}' for i in range(1, 6))
    long_log = "\n".join(f"line {i}" for i in range(1, 40))
    layer = ModelLayer(model_returning(f"[{every}]"))
    outcome = scan_with_model(layer, long_log, "")
    assert len(outcome.findings) <= 6


def test_the_number_of_lines_examined_is_reported():
    """Recall is bounded by what was actually looked at, so the report says so."""
    long_log = "\n".join(f"line {i}" for i in range(1, 60))
    layer = ModelLayer(model_returning("[]"))
    outcome = scan_with_model(layer, long_log, "", max_lines=10)
    assert outcome.lines_examined == 10
    assert outcome.lines_skipped == 49


# --------------------------------------------------------------------------- #
# degradation
# --------------------------------------------------------------------------- #


def test_an_unavailable_model_is_recorded_rather_than_passed_over():
    outcome = scan_with_model(ModelLayer(None), STDOUT, "")
    assert not outcome.ok
    check = build_check(outcome)
    assert check is not None
    assert check.severity is Severity.INFO
    assert check.passed  # informational: it bounds the claim, it is not a fault
    assert check.evidence["degraded"] is True


def test_a_failing_scan_does_not_stop_the_gate(config):
    def explode(prompt, system):
        raise ConnectionError("provider unreachable")

    report = run_gate1(
        "print('training finished')\n"
        "v = 1.0\nrecord_metadata('seed', 0)\nrecord_result('a', v * 2)\n",
        config(consult_model=explode),
    )
    assert report.passed
    assert report.model_degraded
    assert "provider unreachable" in report.model["errors"][0]


def test_a_silent_run_costs_no_model_call(config):
    """Nothing printed is nothing to scan. The cheapest call is the one not made."""
    calls = []

    def count(prompt, system):
        calls.append(prompt)
        return "[]"

    report = run_gate1(
        "v = 1.0\nrecord_metadata('seed', 0)\nrecord_result('a', v * 2)\n",
        config(consult_model=count),
    )
    assert report.passed
    assert calls == []
    assert report.model["calls"] == 0


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #


def test_model_findings_reach_the_feedback_report(config):
    src = (
        "print('Skipping 3 of 10 folds that raised during fitting')\n"
        "record_metadata('seed', 0)\n"
        "v = 0.5\n"
        "record_result('a', v * 2)\n"
    )
    report = run_gate1(
        src, config(consult_model=model_returning('[{"line": 1, "why": "replicates dropped"}]'))
    )
    assert report.passed
    warning_ids = {c.id for c in report.warnings()}
    assert CHECK_ID in warning_ids
    text = render_feedback(report)
    assert "replicates dropped" in text
    assert "Skipping 3 of 10 folds" in text


def test_model_findings_are_warn_and_never_block(config):
    """A model that flags every line still cannot fail a valid run."""
    every = ",".join(f'{{"line": {i}, "why": "broken"}}' for i in range(1, 6))
    src = (
        "for i in range(5):\n"
        "    print(f'line {i}')\n"
        "record_metadata('seed', 0)\n"
        "v = 0.5\n"
        "record_result('a', v * 2)\n"
    )
    report = run_gate1(src, config(consult_model=model_returning(f"[{every}]")))
    assert report.passed
    for check in report.checks:
        if check.id == CHECK_ID:
            assert check.severity is Severity.WARN
            assert not check.blocking


# --------------------------------------------------------------------------- #
# the measurement path — harness only; the model's own rates need an API key
# --------------------------------------------------------------------------- #


def test_a_perfect_oracle_would_reach_full_recall_through_this_harness():
    """Bounds what the plumbing can achieve, separately from what a model does.

    If a model judged every corpus line correctly, the combined scanner would
    score 1.0 on both rates. Anything less in a real measurement is the model's
    judgement, not the harness losing findings — which is the distinction that
    makes task #5's numbers interpretable.
    """
    from rig.corpus import combined_scanner, load_corpus, score

    truth = {e.line: e.error_signal for e in load_corpus()}

    def oracle(prompt, system):
        line = prompt.rsplit("\t", 1)[-1].strip()
        return '[{"line": 1, "why": "oracle"}]' if truth.get(line) else "[]"

    result = score(combined_scanner(oracle), load_corpus())
    assert result.recall == 1.0
    assert result.precision == 1.0


def test_a_model_that_flags_everything_is_caught_by_precision_not_recall():
    """The failure mode the corpus exists to detect. Recall looks perfect."""
    from rig.corpus import combined_scanner, load_corpus, score

    result = score(
        combined_scanner(lambda p, s: '[{"line": 1, "why": "everything is broken"}]'),
        load_corpus(),
    )
    assert result.recall == 1.0
    assert result.precision < 0.6
    assert len(result.spurious) > 15


def test_the_writer_receives_the_warning_evidence_not_just_the_count(config):
    """The bundle tells the writer these "must be stated in the report". A
    warning it can see the count of but not the content is an instruction to
    guess -- this layer's own failure mode, reproduced at its exit."""
    import re
    from gates.adapters.agentlab import build_evidence_bundle

    def smart(prompt, system):
        if "JSON array" in system:
            for line in prompt.splitlines():
                m = re.match(r"(\d+)\t\[", line)
                if m and "Skipping" in line:
                    return f'[{{"line": {m.group(1)}, "why": "replicates dropped"}}]'
            return "[]"
        return "1. nothing to fix"

    src = (
        "print('Skipping 2 of 10 folds that raised during fitting')\n"
        "record_metadata('seed', 0)\n"
        "v = 0.5\n"
        "record_result('a', v * 2)\n"
    )
    report = run_gate1(src, config(consult_model=smart))
    bundle = build_evidence_bundle(report)
    assert "WARNINGS" in bundle
    assert "replicates dropped" in bundle
    assert "Skipping 2 of 10 folds" in bundle


def test_deterministic_log_findings_also_reach_the_writer(config):
    from gates.adapters.agentlab import build_evidence_bundle

    src = (
        "print('RuntimeWarning: invalid value encountered in true_divide')\n"
        "record_metadata('seed', 0)\n"
        "v = 0.5\n"
        "record_result('a', v * 2)\n"
    )
    bundle = build_evidence_bundle(run_gate1(src, config()))
    assert "invalid value encountered" in bundle
