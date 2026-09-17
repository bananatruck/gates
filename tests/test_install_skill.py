"""`SKILL.md` is the portability claim made executable, so it is tested.

Two kinds of test here, and the second is the point.

The first kind checks the file does not lie about this repo: the entry points,
tests and rigs it tells an installing agent to use all exist under those names.
A skill naming a function that was renamed sends the next agent down a dead end.

The second kind executes the install recipe. `SKILL.md` describes the ``write``
callback for the reference host as wrapping a solver whose first call runs
``initial_solve()`` and whose later calls feed feedback in and run ``solve()``.
That description is worth nothing as prose. Here a fake solver with exactly that
shape drives the real ``report_loop``, so if the documented recipe stops working
the suite says so rather than the next installer finding out.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from gates.adapters.agentlab import (
    REPORT_GATE_INSTRUCTIONS,
    make_report_context,
    report_loop,
    retrieved_arxiv_ids,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
SKILL = (REPO / "SKILL.md").read_text(encoding="utf-8")


def test_the_skill_exists_and_declares_itself():
    """CLAUDE.md section 3: the install path ships as a skill."""
    assert SKILL.startswith("---\n")
    front = SKILL.split("---")[1]
    assert "name:" in front and "description:" in front


@pytest.mark.parametrize(
    "name",
    [
        "make_context",
        "make_review_context",
        "make_report_context",
        "gated_execute",
        "review_loop",
        "report_loop",
        "retrieved_arxiv_ids",
        "REPORT_GATE_INSTRUCTIONS",
    ],
)
def test_every_entry_point_the_skill_names_is_importable(name):
    from gates.adapters import agentlab

    assert name in SKILL, f"{name} is no longer mentioned in SKILL.md"
    assert hasattr(agentlab, name), f"SKILL.md names {name}, which does not exist"


@pytest.mark.parametrize(
    "path",
    [
        "rig/gate3_loop.py",
        "tests/test_key_leak.py",
        "gates/adapters/agentlab.py",
        "gates/setup.py",
        "docs/PLAN.md",
        "CLAUDE.md",
        "README.md",
    ],
)
def test_every_file_the_skill_points_at_exists(path):
    assert path in SKILL, f"{path} is no longer mentioned in SKILL.md"
    assert (REPO / path).exists(), f"SKILL.md points at {path}, which is missing"


def test_the_scenario_count_the_skill_quotes_is_current():
    """The one number in the file. A stale count is how a doc starts lying."""
    from rig.gate3_scenarios import SCENARIOS

    words = {1: "one", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    assert f"drives {words[len(SCENARIOS)]} scenarios" in SKILL


def test_the_named_reachability_tests_exist():
    for name in ("test_the_host_wiring_path_actually_reaches_gate_2",
                 "test_the_host_wiring_path_actually_reaches_gate_3"):
        assert name in SKILL
        found = any(
            name in p.read_text(encoding="utf-8") for p in (REPO / "tests").glob("*.py")
        )
        assert found, f"SKILL.md names {name}, which no test defines"


# --------------------------------------------------------------------------- #
# the recipe, executed
# --------------------------------------------------------------------------- #


class FakeSolver:
    """The reference host's writing solver, reduced to the shape SKILL.md uses.

    ``papersolver.PaperSolver`` keeps ``best_report`` as ``[[lines, score]]`` and
    fills ``section_related_work`` while it writes. Both are reproduced, because
    the recipe reads both. The prompt fed back in is kept so the test can show the
    feedback actually reached the writer.
    """

    def __init__(self, drafts):
        self._drafts = list(drafts)
        self.best_report = [[[], 0.0]]
        self.section_related_work: dict[str, str] = {}
        self.notes: list[str] = []

    def _advance(self):
        draft = self._drafts.pop(0)
        self.best_report = [[draft.splitlines(), 1.0]]
        # The host searches arXiv per section as it writes (D32), so the retrieval
        # record is incomplete until after the first draft exists.
        self.section_related_work["related work"] = (
            "arXiv paper ID: 1902.07153v2\narXiv paper ID: 2410.21676v4"
        )

    def initial_solve(self):
        self._advance()

    def solve(self):
        self._advance()


def writer_from(solver):
    """The ``write`` callback exactly as SKILL.md describes it."""

    def write(feedback):
        if not solver._drafts:
            return None
        if feedback is None:
            solver.initial_solve()
        else:
            solver.notes.append(feedback)
            solver.solve()
        return "\n".join(solver.best_report[0][0])

    return write


REGISTRY = {
    "gate": "GATE 1 — EXECUTION VALIDITY",
    "verdict": "PASS",
    "citable": True,
    "values": {"exp1.acc": {"value": 0.97, "unit": "ratio", "trace_id": "t1"}},
}

TYPED = "\\section{Results}\nAccuracy was 0.97.\n"
BOUND = "\\section{Results}\nAccuracy was \\result{exp1.acc}.\n"


def test_the_documented_writer_shape_drives_the_real_loop(tmp_path):
    """The install recipe, run. A typed number is rejected, the feedback reaches
    the solver, and the revision that cites the token is admitted."""
    solver = FakeSolver([TYPED, BOUND])
    outcome = report_loop(
        make_report_context(research_dir=str(tmp_path), sections=()),
        writer_from(solver),
        registry=REGISTRY,
        retrieved=lambda: retrieved_arxiv_ids(None, solver.section_related_work),
    )
    assert outcome.outcome == "pass"
    # Not vacuous: the loop really rejected the first draft and admitted the
    # second, so the recipe exercised the reject-fix-accept cycle rather than
    # passing a clean manuscript on turn one.
    assert [r.passed for r in outcome.reports] == [False, True]
    assert "report.no_numeric_literals_in_results" in {
        c.id for c in outcome.reports[0].report.failed_checks()
    }
    assert solver.notes, "the rejection never reached the writer"
    assert "0.97" in solver.notes[0]


def test_the_retrieval_recipe_reads_both_host_formats(tmp_path):
    """``retrieved_arxiv_ids(lit_review, section_related_work)`` is what the skill
    tells an installer to pass. It has to cope with the writer's search-result
    text and the student's dicts, including one of them being absent."""
    solver = FakeSolver([BOUND])
    solver.initial_solve()
    from_writer = retrieved_arxiv_ids(None, solver.section_related_work)
    assert from_writer == {"1902.07153v2", "2410.21676v4"}

    both = retrieved_arxiv_ids(
        [{"arxiv_id": "2201.12150v2"}], solver.section_related_work
    )
    assert both == {"1902.07153v2", "2410.21676v4", "2201.12150v2"}
    assert retrieved_arxiv_ids(None, None) == set()


def test_the_writer_prompt_asks_for_the_tokens_the_renderer_substitutes():
    """A prompt that does not mention \\result{} leaves the writer typing numbers,
    and every numeric check then rejects every draft until the budget is spent."""
    assert "\\result{<key>}" in REPORT_GATE_INSTRUCTIONS
    assert "\\limitations{}" in REPORT_GATE_INSTRUCTIONS
    assert "\\result{key}" in SKILL and "\\limitations{}" in SKILL
