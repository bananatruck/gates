"""The install skills are the portability claim made executable, so they are tested.

`skills/` holds a router, ``install-gates``, and one skill per gate, packaged as
a Claude Code plugin by `.claude-plugin/`. Two kinds of test here, and the second
is the point.

The first kind checks the skills do not lie about this repo: the entry points,
tests and rigs they tell an installing agent to use all exist under those names.
A skill naming a function that was renamed sends the next agent down a dead end.

The second kind executes the install recipe. The skills describe the ``write``
callback for the reference host as wrapping a solver whose first call runs
``initial_solve()`` and whose later calls feed feedback in and run ``solve()``.
That description is worth nothing as prose. Here a fake solver with exactly that
shape drives the real ``report_loop``, so if the documented recipe stops working
the suite says so rather than the next installer finding out.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from gates.adapters.agentlab import (
    REPORT_GATE_INSTRUCTIONS,
    make_report_context,
    report_loop,
    retrieved_arxiv_ids,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
SKILLS = REPO / "skills"
GATE_SKILLS = ("gate1-execution", "gate2-coherence", "gate3-report")
ROUTER = (SKILLS / "install-gates" / "SKILL.md").read_text(encoding="utf-8")
#: Every file an installing agent can reach from the router, read as one text.
SKILL = "\n".join(
    path.read_text(encoding="utf-8") for path in sorted(SKILLS.rglob("*.md"))
)


def _front(text):
    assert text.startswith("---\n")
    return dict(
        line.split(": ", 1) for line in text.split("---")[1].strip().splitlines()
    )


@pytest.mark.parametrize("folder", ("install-gates",) + GATE_SKILLS)
def test_each_skill_exists_and_declares_itself(folder):
    """CLAUDE.md section 3: the install path ships as skills. A skill whose name
    is not its folder's is one the agent cannot invoke by the name it sees."""
    front = _front((SKILLS / folder / "SKILL.md").read_text(encoding="utf-8"))
    assert front["name"] == folder
    assert front["description"]


def test_the_router_sends_the_agent_through_every_gate_in_order():
    positions = [ROUTER.index(f"`{name}`") for name in GATE_SKILLS]
    assert positions == sorted(positions)


def test_the_plugin_ships_exactly_the_skills_in_the_repo():
    """`/plugin install gates@gates` installs what plugin.json lists, so a skill
    left off the list exists in the repo and nowhere a user can reach it."""
    plugin = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
    market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    listed = {pathlib.PurePosixPath(p).name for p in plugin["skills"]}
    on_disk = {p.parent.name for p in SKILLS.glob("*/SKILL.md")}
    assert listed == on_disk
    assert [p["name"] for p in market["plugins"]] == [plugin["name"]]
    assert market["plugins"][0]["source"] == "./"


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
        "extract_plan_fields",
        "arxiv_lookup",
        "gate_level",
        "require_gate",
        "record_divergence",
        "build_registry",
        "MLE_GATE_INSTRUCTIONS",
        "REPORT_GATE_INSTRUCTIONS",
    ],
)
def test_every_entry_point_the_skill_names_is_importable(name):
    import gates
    from gates import pipeline
    from gates.adapters import agentlab

    assert name in SKILL, f"{name} is no longer mentioned in the skills"
    assert any(
        hasattr(module, name) for module in (gates, pipeline, agentlab)
    ), f"the skills name {name}, which does not exist"


@pytest.mark.parametrize(
    "path",
    [
        "rig/gate1_loop.py",
        "rig/gate3_loop.py",
        "tests/test_key_leak.py",
        "tests/test_gate2.py",
        "tests/test_gate3.py",
        "tests/test_install_skill.py",
        "gates/pipeline.py",
        "gates/adapters/arxiv.py",
        "gates/adapters/agentlab.py",
        "gates/setup.py",
        "docs/PLAN.md",
        "CLAUDE.md",
        "README.md",
    ],
)
def test_every_file_the_skill_points_at_exists(path):
    assert path in SKILL, f"{path} is no longer mentioned in the skills"
    assert (REPO / path).exists(), f"the skills point at {path}, which is missing"


@pytest.mark.parametrize("module", ["rig.gate2_scenarios", "rig.gate3_scenarios"])
def test_the_scenario_counts_the_skills_quote_are_current(module):
    """The numbers in the files. A stale count is how a doc starts lying."""
    import importlib

    count = len(importlib.import_module(module).SCENARIOS)
    words = {1: "one", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    assert f"drives {words[count]} scenarios" in SKILL


def test_the_named_reachability_tests_exist():
    for name in ("test_the_host_wiring_path_actually_reaches_gate_2",
                 "test_the_host_wiring_path_actually_reaches_gate_3"):
        assert name in SKILL
        found = any(
            name in p.read_text(encoding="utf-8") for p in (REPO / "tests").glob("*.py")
        )
        assert found, f"the skills name {name}, which no test defines"


# --------------------------------------------------------------------------- #
# the recipe, executed
# --------------------------------------------------------------------------- #


class FakeSolver:
    """The reference host's writing solver, reduced to the shape SKILL.md uses.

    ``papersolver.PaperSolver`` keeps one best draft in ``best_report`` as
    ``[(lines, score, ...)]``, replaces it only when a new draft scores higher,
    interpolates ``notes`` straight into its prompt, and fills
    ``section_related_work`` while it writes. All four are reproduced, because
    the recipe depends on all four. Each draft is ``(text, reward score)``.
    """

    def __init__(self, drafts, notes=""):
        self._drafts = list(drafts)
        self.best_report = [([], 0.0)]
        self.section_related_work: dict[str, str] = {}
        self.notes = notes
        self.prompts: list[str] = []

    def _next(self):
        text, score = self._drafts.pop(0)
        # The host searches arXiv per section as it writes (D32), so the retrieval
        # record is incomplete until after the first draft exists.
        self.section_related_work["related work"] = (
            "arXiv paper ID: 1902.07153v2\narXiv paper ID: 2410.21676v4"
        )
        return text.splitlines(), score

    def initial_solve(self):
        self.best_report = [self._next()]

    def solve(self):
        self.prompts.append(
            f"The following are notes, instructions, and general tips for you: {self.notes}"
        )
        if not self._drafts:
            return
        lines, score = self._next()
        if score > self.best_report[-1][1]:
            self.best_report = [(lines, score)]


def writer_from(solver):
    """The ``write`` callback exactly as the skills describe it."""

    def write(feedback):
        if feedback is None:
            solver.initial_solve()
            return "\n".join(solver.best_report[0][0])
        # The gate outranks the reward model: the rejected draft loses its score,
        # so whatever the solver writes next replaces it.
        lines, _, *rest = solver.best_report[0]
        solver.best_report[0] = (lines, float("-inf"), *rest)
        solver.notes = f"{solver.notes}\n{feedback}" if solver.notes else feedback
        solver.solve()
        lines, score, *_ = solver.best_report[0]
        return None if score == float("-inf") else "\n".join(lines)

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
    solver = FakeSolver([(TYPED, 0.5), (BOUND, 0.9)])
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
    assert "0.97" in solver.prompts[-1], "the rejection never reached the writer"


def test_a_fix_the_reward_model_ranks_lower_still_reaches_the_gate(tmp_path):
    """The host replaces its best draft only on a higher reward. A recipe that
    returns the best draft resubmits the rejected one when the fix scores lower,
    and Gate 3 raises with a correct revision in hand."""
    solver = FakeSolver([(TYPED, 0.9), (BOUND, 0.5)])
    outcome = report_loop(
        make_report_context(research_dir=str(tmp_path), sections=()),
        writer_from(solver),
        registry=REGISTRY,
    )
    assert outcome.outcome == "pass"
    assert [r.passed for r in outcome.reports] == [False, True]


def test_a_solver_step_that_scores_nothing_stops_the_loop(tmp_path):
    """Resubmitting the rejected draft would spend a turn to hear the same thing."""
    solver = FakeSolver([(TYPED, 0.9)])
    outcome = report_loop(
        make_report_context(research_dir=str(tmp_path), sections=()),
        writer_from(solver),
        registry=REGISTRY,
    )
    assert outcome.outcome == "no_pass"
    assert len(outcome.reports) == 1


def test_the_rejection_reaches_the_prompt_as_written(tmp_path):
    """Hosts interpolate notes into the prompt. A list renders as its repr, which
    doubles every backslash in the \\result{} instruction and flattens the
    rejection onto one line."""
    solver = FakeSolver([(TYPED, 0.9), (BOUND, 0.5)], notes=REPORT_GATE_INSTRUCTIONS)
    outcome = report_loop(
        make_report_context(research_dir=str(tmp_path), sections=()),
        writer_from(solver),
        registry=REGISTRY,
    )
    prompt = solver.prompts[-1]
    assert "\\result{<key>}" in prompt
    assert "\\\\result" not in prompt
    assert outcome.reports[0].feedback in prompt


def test_the_retrieval_recipe_reads_both_host_formats(tmp_path):
    """``retrieved_arxiv_ids(lit_review, section_related_work)`` is what the skill
    tells an installer to pass. It has to cope with the writer's search-result
    text and the student's dicts, including one of them being absent."""
    solver = FakeSolver([(BOUND, 0.9)])
    solver.initial_solve()
    from_writer = retrieved_arxiv_ids(None, solver.section_related_work)
    assert from_writer == {"1902.07153v2", "2410.21676v4"}

    both = retrieved_arxiv_ids(
        [{"arxiv_id": "2201.12150v2"}], solver.section_related_work
    )
    assert both == {"1902.07153v2", "2410.21676v4", "2201.12150v2"}
    assert retrieved_arxiv_ids(None, None) == set()


def test_the_skill_describes_the_reference_host_as_wired():
    """D42: the worked example is as-built, not a connect spec with drifting line numbers."""
    assert "review_loop(..., first=final)" in SKILL
    assert "writer_from_paper_solver" in SKILL
    assert "arxiv_lookup" in SKILL
    assert "reviser_from_mle_solver" in SKILL
    assert "It is already wired (D42)." in SKILL
    # Line numbers in the worked example would go stale on the next host edit.
    assert "line 349" not in SKILL
    assert "line 279" not in SKILL


def test_the_writer_prompt_asks_for_the_tokens_the_renderer_substitutes():
    """A prompt that does not mention \\result{} leaves the writer typing numbers,
    and every numeric check then rejects every draft until the budget is spent."""
    assert "\\result{<key>}" in REPORT_GATE_INSTRUCTIONS
    assert "\\limitations{}" in REPORT_GATE_INSTRUCTIONS
    assert "\\result{key}" in SKILL and "\\limitations{}" in SKILL
