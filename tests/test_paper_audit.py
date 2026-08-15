"""Auditing a generated paper's numbers against what its run recorded.

The interesting failure here is not a paper that lies — it is an audit that
flatters. Scored against a whole-workflow log, every one of the archived paper's
65 claims came back "printed", because the interpretation agent restates the
paper's figures in prose and the log keeps everything every agent said. The
number looked like a finding and was an artifact of the input. So the guard that
rejects that input is tested first and hardest: an audit that can be fooled by
its own evidence is worse than no audit.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rig.paper_audit import (  # noqa: E402
    _close,
    _contaminated,
    _is_claim,
    audit,
    extract_claims,
)

PAPER = r"""
\section{Abstract}
We reach $81.60\%$ test accuracy at $K=2$ and a $13.61\times$ speedup
($0.0180$ seconds vs. $0.2450$ seconds).

\section{Background}
Kipf and Welling reported $99.99\%$ on a different task in 2017.

\section{Results}
Accuracy falls to $59.10\%$ at $K=10$.
"""


# --------------------------------------------------------------------------- #
# what counts as a claim
# --------------------------------------------------------------------------- #


def test_only_findings_sections_are_audited():
    """A number quoted from someone else's paper is not this run's to source."""
    values = {c.value for c in extract_claims(PAPER)}
    assert 81.6 in values and 59.1 in values
    assert 99.99 not in values, "background figures must not be audited"


def test_years_and_small_integers_are_not_claims():
    assert not _is_claim("2017")
    assert not _is_claim("2")
    assert _is_claim("81.60")
    assert _is_claim("13.61")


def test_a_paper_rounds_so_matching_tolerates_scale():
    assert _close(81.6, 0.816)      # registry stores a fraction
    assert _close(81.6, 81.601)     # paper rounds
    assert not _close(81.6, 72.8)


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #


def test_workflow_transcript_is_refused_as_an_execution_capture():
    """The failure that produced 65/65 'printed' from a log of agent prose."""
    transcript = (
        "Beginning phase: experimentation\n"
        "Beginning subtask: running experiments\n"
        "The model reaches 81.60% accuracy, a 13.61x speedup.\n"
    )
    assert _contaminated(transcript)


def test_a_real_execution_capture_is_accepted():
    capture = (
        "epoch=10 loss=1.4576\n"
        "epoch=20 loss=1.4570\n"
        "acc_25=0.3765 acc_100=0.3805\n"
    )
    assert not _contaminated(capture)


def test_refused_capture_yields_unsourced_not_printed(tmp_path):
    """Refusing the input must not silently score the claims as sourced."""
    paper = tmp_path / "report.txt"
    paper.write_text(PAPER)
    transcript = (
        "Beginning phase: experimentation\n"
        "Subtask 'running experiments' completed in 1s\n"
        "81.60% and 13.61x and 59.10%\n"
    )
    result = audit(paper, code_output=transcript)
    assert result.count("printed") == 0
    assert result.count("unsourced") == len(result.claims)
    assert "workflow transcript" in result.note


# --------------------------------------------------------------------------- #
# sourcing
# --------------------------------------------------------------------------- #


def test_a_recorded_result_sources_its_claim(tmp_path):
    paper = tmp_path / "report.txt"
    paper.write_text(PAPER)
    registry = tmp_path / "registry.json"
    registry.write_text(
        '{"values": {"exp1.test_acc": {"value": 0.816, "trace_id": "abc"}}}'
    )
    result = audit(paper, registry_path=registry)
    sourced = [c for c in result.claims if c.status == "sourced"]
    assert [c.value for c in sourced] == [81.6]
    assert sourced[0].source == "registry:exp1.test_acc"


def test_no_registry_means_nothing_is_sourced(tmp_path):
    """The archived run's shape: a fluent paper, and no record behind it."""
    paper = tmp_path / "report.txt"
    paper.write_text(PAPER)
    result = audit(paper, code_text="print('done')")
    assert result.count("sourced") == 0
    assert result.record_result_calls == 0
    assert result.sourced_rate == 0.0


def test_missing_paper_is_reported_not_scored(tmp_path):
    result = audit(tmp_path / "nope.txt")
    assert result.claims == []
    assert result.sourced_rate is None
    assert "not found" in result.note
