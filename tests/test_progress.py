"""``progress.md``'s STATE block, checked against the tree it describes.

Every other status document in this repository has gone stale at least once, and
a document that records a number nobody verifies is the defect this project
exists to catch, committed by us. So the counts are checked.

**What is checked and what is not.** The test counts are checked, because they
are facts about the tree that a machine can read. ``branch``, ``head`` and
``head_date`` are not, and the STATE block says so beside them. A commit cannot
record its own SHA, so a machine-checked ``head`` would be unsatisfiable by
construction, and claiming to check it would repeat exactly the mistake this file
is here to prevent.

Collection runs in a subprocess with ``--collect-only``, so no test executes
twice and the count is the real one rather than this session's, which would be
wrong whenever someone runs a single file or passes ``-k``.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from collections import Counter

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PROGRESS = REPO / "progress.md"

#: STATE key -> the test file it counts. ``tests_total`` is the whole tree.
SUITES = {
    "tests_gate1": "tests/test_gate1.py",
    "tests_gate2": "tests/test_gate2.py",
    "tests_gate3": "tests/test_gate3.py",
    "tests_llm_scan": "tests/test_llm_scan.py",
    "tests_llm_layer": "tests/test_llm_layer.py",
}


def _state() -> dict[str, str]:
    """The STATE block as a mapping, trailing ``#`` comments stripped."""
    text = PROGRESS.read_text(encoding="utf-8")
    match = re.search(r"<!--\s*STATE:BEGIN\s*-->(.*?)<!--\s*STATE:END\s*-->", text, re.S)
    assert match, "progress.md has no STATE block"
    state: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        state[key.strip()] = value.strip()
    return state


def _collected() -> Counter[str]:
    """Tests collected per file, counted the way pytest counts them.

    Parametrised cases expand, so this is not the number of ``def test_``
    statements and must not be computed by reading the source.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    counts: Counter[str] = Counter()
    for line in result.stdout.splitlines():
        if "::" in line:
            counts[line.split("::", 1)[0]] += 1
    assert counts, f"collection produced nothing:\n{result.stdout}\n{result.stderr}"
    return counts


@pytest.fixture(scope="module")
def collected() -> Counter[str]:
    return _collected()


def test_progress_records_the_real_total(collected):
    """A stale total is how a status document starts lying."""
    state = _state()
    assert int(state["tests_total"]) == sum(collected.values()), (
        "progress.md's tests_total does not match the tree; re-record the "
        "STATE block"
    )


@pytest.mark.parametrize("key, path", sorted(SUITES.items()))
def test_progress_records_the_real_suite_count(collected, key, path):
    state = _state()
    assert int(state[key]) == collected[path], f"{key} does not match {path}"


def test_every_named_suite_exists(collected):
    """A STATE key naming a file that is gone would silently pass as zero."""
    missing = sorted(p for p in SUITES.values() if p not in collected)
    assert missing == [], f"STATE names test files that collected nothing: {missing}"


def test_the_state_block_says_which_fields_are_not_checked():
    """The honesty rule applied to this file.

    A block claiming to be machine-checked while carrying unchecked fields is
    the same overclaim as a green check that never ran, so the unchecked fields
    have to be marked in the document itself.
    """
    text = PROGRESS.read_text(encoding="utf-8").lower()
    assert "not machine-checked" in text, (
        "progress.md must say which STATE fields this test does not verify"
    )
    for field in ("branch", "head", "head_date"):
        assert field in text, f"the unchecked field {field!r} is not named"


#: README's test table: each row's files. A test file in no group fails
#: ``test_the_readme_test_table_matches_the_tree``, so a new file must be placed.
README_GROUPS = {
    "Gate 1: checks, loop, level-0 bypass": ("test_gate1", "test_loop", "test_bypass"),
    "Gate 2: checks, loop, tier comparison": ("test_gate2", "test_gate2_loop", "test_gate2_tiers"),
    "Gate 3: checks, loop, model layer, scanner miss, arXiv resolver": (
        "test_gate3", "test_gate3_loop", "test_gate3_model", "test_gate3_m4", "test_arxiv_lookup",
    ),
    "Model layer and log scanning": (
        "test_llm_report", "test_llm_scan", "test_llm_layer", "test_log_corpus",
        "test_log_digest", "test_retrieval", "test_model_cost",
    ),
    "Wiring, levels, setup, install skills, key handling": (
        "test_levels", "test_setup", "test_install_skill", "test_key_leak", "test_portability",
    ),
    "Evaluation tooling and this status check": (
        "test_tuning", "test_paper_audit", "test_stats", "test_live_tools", "test_collect",
        "test_mechanism", "test_posthoc_audit", "test_progress",
    ),
}


def test_the_readme_test_table_matches_the_tree(collected):
    """README's test metrics are a status document too, so they are checked."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    grouped = {f"tests/{name}.py" for files in README_GROUPS.values() for name in files}
    assert sorted(set(collected) - grouped) == [], "a test file is in no README group"
    for group, files in README_GROUPS.items():
        count = sum(collected[f"tests/{name}.py"] for name in files)
        assert f"| {group} | {count} |" in readme, f"README's row for {group!r} should say {count}"
    total = sum(collected.values())
    assert f"| **Total** | **{total}** |" in readme
    assert int(_state()["tests_total"]) == total
