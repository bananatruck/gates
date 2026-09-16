"""No gate writes the provider key anywhere it can be read back (Q14).

The key reaches the gates only inside the function ``make_gate_model`` returns,
and ``gates/`` never holds it. This plays Gates 1, 2 and 3 through the adapter,
with a sentinel key in the environment and inside that function, then reads
every file the run wrote and every prompt the model was sent.

The experiment prints its whole environment on purpose. That is the ordinary
way agent code would expose a key, and Gate 1's scrub is what stops it
(``runner.py``). Reading the parent's ``/proc`` environment is tested in
``tests/test_gate1.py``, with its root caveat (B3).
"""

from __future__ import annotations

import sys
import types

from gates.adapters.agentlab import (
    make_context,
    make_gate_model,
    make_report_context,
    make_review_context,
    report_loop,
    review_loop,
)

SENTINEL = "sk-gates-sentinel-4f1c9e27"

EXPERIMENT = (
    "import os\n"
    "print(dict(os.environ))\n"
    "record_metadata('seed', 0)\n"
    "acc = sum([0.812])\n"
    "record_result('exp1.acc', acc, unit='ratio')\n"
)

TYPED = "\\section{Results}\nAccuracy reaches 0.812.\nIt trains twice as fast.\n"
CITED = "\\section{Results}\nAccuracy reaches \\result{exp1.acc}.\n"


def fake_host(sent: list[tuple[str, str, str]]) -> types.ModuleType:
    """The host's ``inference`` module, reduced to what the adapter calls."""
    module = types.ModuleType("inference")

    def query_model(*, model_str, prompt, system_prompt, openai_api_key, **_):
        sent.append((prompt, system_prompt, openai_api_key))
        return "[]" if "JSON array" in system_prompt else "1. Cite the accuracy with its token."

    module.query_model = query_model
    return module


def test_no_gate_writes_or_sends_the_key(tmp_path, monkeypatch):
    sent: list[tuple[str, str, str]] = []
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    monkeypatch.setitem(sys.modules, "inference", fake_host(sent))
    research = str(tmp_path / "research")

    gate1 = make_context(
        research_dir=research, consult_model=make_gate_model("gpt-test", SENTINEL)
    )
    code = iter([EXPERIMENT])
    reviewed = review_loop(
        make_review_context(research_dir=research),
        lambda feedback: next(code, None),
        gate1=gate1,
    )
    drafts = iter([TYPED, CITED])
    written = report_loop(
        make_report_context(
            research_dir=research,
            consult_model=make_gate_model("gpt-test", SENTINEL),
            # This is about where the key goes, not about the manuscript. CITED
            # is two sections, so the host's declared list would reject it and
            # the run would never reach a pass.
            sections=(),
        ),
        lambda feedback: next(drafts, None),
        registry=reviewed.registry,
        declared=reviewed.declared,
    )

    # Not vacuous: the key travelled, the model was asked by both gates, and
    # the experiment really printed its environment.
    assert reviewed.outcome == "pass" and written.outcome == "pass"
    assert {key for _, _, key in sent} == {SENTINEL}
    assert len(sent) >= 3
    stdout = next(tmp_path.rglob("stdout.txt")).read_text()
    assert "PATH" in stdout

    assert not [prompt for prompt, _, _ in sent if SENTINEL in prompt]
    leaked = [
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
        if path.is_file() and SENTINEL.encode() in path.read_bytes()
    ]
    assert leaked == []
