"""Wires G.A.T.E.S. into Agent-Researcher / Agent Laboratory.

The insertion point is between code execution and the reward model. Upstream,
``Replace.parse_command`` executed the code and decided success by searching the
(truncated) stdout for a marker string; ``get_score`` then asked an LLM for a
float and treated "a float parsed" as "the run worked". Here, execution goes
through the sandboxed runner, Gate 1 issues the verdict, and ``get_score``
ranks only what already passed.

This file holds only what knows Agent Laboratory: its phase names and paths,
its writer's sections, its model call, its original execution path and its
retrieval formats. The loops are ``gates/pipeline.py`` and are imported here so
the host has one module to import from.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from collections.abc import Callable

from .. import pipeline
from ..errors import GateError
from ..gate2 import IMPLAUSIBLE_SPEEDUP, PlanField, Range, Relation, SourceClaim
from ..llm import ModelFn, ModelLayer
from ..schema import PaperRecord
from .. import Gate1Config, Gate2Config, Gate3Config, Ledger, run_experiment
from ..pipeline import (  # noqa: F401 - the host imports these from here
    MLE_GATE_INSTRUCTIONS,
    REPORT_GATE_INSTRUCTIONS,
    GateContext,
    GatedExecution,
    build_evidence_bundle,
    gate1_enabled,
    gate_level,
    record_divergence,
    report_loop,
    review_loop,
)
from .arxiv import arxiv_lookup  # noqa: F401 - the host imports it from here


def make_context(
    *,
    research_dir: str = "./research_dir",
    phase: str = "running experiments",
    max_attempts: int = 3,
    timeout_s: int = 900,
    expected_keys: tuple[str, ...] | None = None,
    require_metrics: bool = True,
    num_classes: int | None = None,
    reward_model: str | None = None,
    task_ref: str | None = None,
    consult_model: Any = None,
) -> GateContext:
    """Build the gate context for one solver phase.

    ``task_ref`` is the plan or task text the experiment implements. Passing it
    closes the first link of each value's provenance chain; omitting it leaves
    that link recorded as unresolved rather than assumed.
    """
    artifact_root = os.path.join(research_dir, "gate_artifacts")
    config = Gate1Config(
        max_attempts=max_attempts,
        timeout_s=timeout_s,
        expected_keys=expected_keys,
        require_metrics=require_metrics,
        num_classes=num_classes,
        artifact_root=artifact_root,
        cwd=os.getcwd(),  # figures must land where papersolver looks for them
        task_ref=task_ref,
        # The LLM layer. In this scaffold the caller passes a closure over
        # inference.query_model; absent one, the gate issues the same verdict and
        # falls back to the deterministic feedback template.
        consult_model=consult_model,
    )
    return GateContext(
        config=config,
        ledger=Ledger(os.path.join(artifact_root, "divergence.jsonl")),
        phase=phase,
        reward_model=reward_model,
    )


def make_review_context(
    *,
    research_dir: str = "./research_dir",
    phase: str = "results interpretation",
    max_attempts: int = 2,
    relations: tuple[Relation, ...] = (),
    ranges: dict[str, Range] | None = None,
    implausible_speedup: float = IMPLAUSIBLE_SPEEDUP,
    plan_fields: tuple[PlanField, ...] = (),
    reward_model: str | None = None,
    sources: tuple[SourceClaim, ...] = (),
    lit_review: list[dict[str, Any]] | None = None,
) -> GateContext:
    """Build the gate context for the review phase, the one Gate 2 runs in.

    ``make_context`` builds a ``Gate1Config``, and ``gated_review`` hands the
    context's config straight to ``run_gate2``. Driven the way the host is
    documented to drive it, that raised ``AttributeError: 'Gate1Config' object
    has no attribute 'ranges'``: Gate 2 had no reachable entry point, and only
    looked wired because every test built its config by hand. This is that entry
    point.

    Separate from ``make_context`` rather than folded into it, because the two
    gates run in different phases and carry different budgets. Gate 1 gets three
    revisions and Gate 2 gets two, per `PLAN.md` §4, and one context holding both
    would have to hold two ``consecutive_rejections`` counters to keep them
    apart.

    ``relations``, ``ranges`` and ``plan_fields`` are what the plan declared
    about this experiment, so they arrive from the host at wiring time rather
    than being inferred from anything `gates/` reads (D13). Agent Laboratory's
    plan is free text, so whoever wires the host declares the fields; with none
    declared, tier B does not run and the report carries no tier B check.

    ``sources`` are declared the same way, and bound to ``lit_review``, the
    host's list of papers it actually fetched (D21, D23): a claim citing any
    other ``source_id`` is refused here, so no band comes from a paper nobody
    read. The numbers are declared rather than parsed from ``full_text``,
    because only a model could bind a number in prose to a registry key, and no
    model reaches a Gate 2 verdict (D19).
    """
    if sources:
        if lit_review is None:
            raise GateError("sources were declared without a lit_review to bind them to")
        fetched = {entry.get("arxiv_id") for entry in lit_review}
        unfetched = sorted({c.source_id for c in sources} - fetched)
        if unfetched:
            raise GateError(
                f"source(s) not in lit_review, so never fetched: {', '.join(unfetched)}"
            )
    artifact_root = os.path.join(research_dir, "gate_artifacts")
    config = Gate2Config(
        max_attempts=max_attempts,
        relations=relations,
        ranges=dict(ranges or {}),
        implausible_speedup=implausible_speedup,
        plan_fields=tuple(plan_fields),
        sources=tuple(sources),
        artifact_root=artifact_root,
    )
    return GateContext(
        config=config,
        ledger=Ledger(os.path.join(artifact_root, "divergence.jsonl")),
        phase=phase,
        reward_model=reward_model,
    )


#: What an extracted field may be called: a setting, never a result, so a field
#: a model read from the plan cannot collide with a number the paper reports.
_PLAN_KEY = re.compile(r"config\.[a-z][a-z0-9_]*")

PLAN_EXTRACTION_PROMPT = """You read a machine-learning research plan and list the concrete
settings it commits to: numbers, choices and names the experiment must use,
such as epochs, learning rate, batch size, optimizer, dataset or model.

Answer with a JSON array and nothing else. One object per setting:
  {"key": "config.<snake_case_name>", "declared": <the value>, "quote": "<the plan's exact words>"}

"declared" is a number, a string or true/false. "quote" is copied word for word
from the plan. List only what the plan states outright; if it states nothing
concrete, answer []."""


def extract_plan_fields(plan: str, model: ModelFn | None) -> tuple[PlanField, ...]:
    """Tier B's input, read from this host's free-text plan by a model (F2, D55).

    Agent Laboratory's plan is prose, so nobody declared ``plan_fields`` and
    tier B never ran. ``model`` is the judge, which must not be the model under
    test (D54). Every field returned is ``model_authored``, so a divergence on
    it warns and cannot fail the run.

    What the model says is checked before it is kept: a key must be a
    ``config.*`` setting, a value must be a number, string or bool, and the
    quote must appear in the plan, or the field was invented. A model that
    fails or answers nonsense returns nothing, and tier B stays silent.
    """
    call = ModelLayer(model).ask(f"PLAN:\n{plan}", PLAN_EXTRACTION_PROMPT)
    if not call.ok:
        return ()
    try:
        items = json.loads(call.text[call.text.index("["):call.text.rindex("]") + 1])
    except ValueError:
        return ()
    words = " ".join(plan.split())
    fields: list[PlanField] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        key, declared, quote = item.get("key"), item.get("declared"), item.get("quote")
        if (
            not isinstance(key, str)
            or not _PLAN_KEY.fullmatch(key)
            or key in {f.key for f in fields}
            or not isinstance(declared, (bool, int, float, str))
            or not isinstance(quote, str)
            or not quote.strip()
            or " ".join(quote.split()) not in words
        ):
            continue
        fields.append(PlanField(key, declared, f'plan: "{" ".join(quote.split())}"', model_authored=True))
    return tuple(fields)


def plan_field_instructions(fields: tuple[PlanField, ...]) -> str:
    """The engineer's half of tier B: record each setting the plan committed to.

    A field the run never records is unverifiable, so the engineer is asked for
    every one. Not added to Gate 1's expected keys, because then a field a model
    authored could fail a run, which D55 forbids.
    """
    if not fields:
        return ""
    lines = [
        "============= PLAN SETTINGS (RECORD THESE) =============",
        "The plan commits to these settings. Record the value your code actually",
        "uses for each, passing the variable that holds it:",
        "",
    ]
    lines += [f'    record_result("{f.key}", <variable>)   # plan: {f.declared!r}' for f in fields]
    return "\n".join(lines) + "\n"


#: The sections this host's paper writer is told to produce
#: (``papersolver.py:352``), which is what ``style.sections_present`` holds it to
#: (D27). ``"scaffold"`` is dropped: it is the document skeleton the writer
#: builds first, not a section a reader looks for.
WRITER_SECTIONS = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "methods",
    "experimental setup",
    "results",
    "discussion",
)


def make_report_context(
    *,
    research_dir: str = "./research_dir",
    phase: str = "report writing",
    max_attempts: int = 3,
    figure_root: str | None = None,
    sections: tuple[str, ...] = WRITER_SECTIONS,
    lookup: Callable[[str], PaperRecord | None] | None = None,
    consult_model: Any = None,
) -> GateContext:
    """Build the gate context for the writing phase, the one Gate 3 runs in.

    Its own builder for the reason Gate 2 has one (D15): a third phase with its
    own budget. ``phase`` is the host's own name for it (``ai_lab_repo.py:294``).
    ``figure_root`` is where the run's figures live; ``None`` resolves them
    against the working directory and skips the check that they stayed inside
    the run. ``sections`` is what the manuscript must contain, defaulting to this
    host's own writer list (D27); ``()`` leaves sections unchecked. ``lookup``
    resolves a cited arXiv id to a real paper, usually :func:`arxiv_lookup`;
    ``None`` leaves identifiers unresolved and says so in the report.
    ``consult_model`` is Gate 3's model layer (D31), built the way
    Gate 1's is: ``make_gate_model(backend, key)`` returns a new function for
    this gate, so Gate 3's spend is counted apart from Gate 1's.
    """
    artifact_root = os.path.join(research_dir, "gate_artifacts")
    config = Gate3Config(
        max_attempts=max_attempts,
        artifact_root=artifact_root,
        figure_root=figure_root,
        sections=sections,
        lookup=lookup,
        consult_model=consult_model,
    )
    return GateContext(
        config=config,
        ledger=Ledger(os.path.join(artifact_root, "divergence.jsonl")),
        phase=phase,
    )


#: Gate output is a short JSON array or four numbered sentences. Nothing it
#: produces legitimately runs long, so the cap is generous rather than tight and
#: exists to bound the pathological case rather than to shape the answer.
GATE_MAX_TOKENS = 1024


#: Models that reason at length unless told not to. Both of the gate's jobs are
#: classification and short instruction-writing; extended reasoning buys nothing
#: and costs a great deal. Measured on qwen3:8b before this was applied: 630
#: output tokens on average and 21,858 on the worst call, which at 26 tokens a
#: second is fourteen minutes spent on a request whose answer was "[]".
_NO_THINK_PREFIXES = ("qwen3",)


def make_gate_model(
    model_str: str,
    api_key: str | None = None,
    temp: float = 0.0,
    max_tokens: int | None = GATE_MAX_TOKENS,
):
    """Adapt this scaffold's ``query_model`` to the layer's two-argument seam.

    The import is deliberately lazy. ``adapters/`` is where host knowledge is
    allowed to live, but the package still has to import cleanly outside the
    host — a module-level ``import inference`` would break ``pip install gates``
    for everyone who is not Agent-Researcher.

    ``temp=0.0``: this model writes a validity report, not prose. The upstream
    reward model runs at 0.6 and that is part of what Gate 1 exists to answer.

    Model-specific quirks belong here rather than in ``gates/``, which stays
    ignorant of who is answering it. ``/no_think`` is qwen3's own switch and is
    inert text to any other model.
    """
    from inference import query_model  # noqa: PLC0415 — see docstring

    quiet = model_str.split(":")[0].split("-")[0].lower().startswith(
        _NO_THINK_PREFIXES
    )

    def call(prompt: str, system_prompt: str) -> str:
        if quiet:
            system_prompt = f"{system_prompt}\n/no_think"
        return query_model(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            openai_api_key=api_key,
            temp=temp,
            print_cost=False,
            max_tokens=max_tokens,
        )

    return call


#: Upstream's ceiling on what the writing agent ever saw. Reproduced exactly in
#: the bypass path, because a baseline that quietly got a bigger channel would
#: understate the defect this layer exists to fix.
LEGACY_MAX_LEN = 1000


def _ungated_execute(code: str, context: GateContext) -> GatedExecution:
    """The host's original path: run it, truncate to 1,000 characters, accept.

    Deliberately not "Gate 1 with the checks skipped". Upstream appended its
    crash marker *after* the program's own output and then sliced the whole
    string, so on a run that prints past the ceiling the marker falls off and
    the failure becomes invisible. That ordering is the defect, so the bypass
    reproduces it rather than tidying it up.
    """
    context.attempt += 1
    workdir = Path(context.config.artifact_root) / f"ungated_{context.attempt:02d}"
    execution = run_experiment(code, workdir,
                               timeout_s=context.config.timeout_s)
    captured = execution.stdout_text()
    exc = execution.exception
    if exc is not None:
        # Appended AFTER the program's own output, then the whole thing sliced.
        captured += f"[CODE EXECUTION ERROR]: {exc.message}\n{exc.traceback}"
    bundle = captured[:LEGACY_MAX_LEN]
    accepted = "[CODE EXECUTION ERROR]" not in bundle

    lost = exc is not None and accepted
    print(f"$$$$ gate 1 BYPASSED (level 0) — attempt {context.attempt}, "
          f"{'accepted' if accepted else 'rejected'}, {len(captured):,} chars "
          f"captured, {len(bundle):,} handed on"
          + ("  ** CRASHED, and the marker fell outside the slice **" if lost else ""))

    return GatedExecution(
        report=None,
        feedback=bundle,
        evidence_bundle=bundle,
        code=code,
        ungated_passed=accepted,
    )


def gated_execute(code: str, context: GateContext) -> GatedExecution:
    """Gate 1 when it is on, and this host's original path when it is off."""
    if not gate1_enabled():
        return _ungated_execute(code, context)
    return pipeline.gated_execute(code, context)


#: One line per paper in the text the host's arXiv search returns
#: (``tools.py`` ``ArxivSearch.find_papers_by_str``).
_SEARCH_RESULT_ID = re.compile(r"arXiv paper ID:\s*(\S+)")


def retrieved_arxiv_ids(
    lit_review: list[dict[str, Any]] | None,
    section_related_work: dict[str, str | None] | None,
) -> set[str]:
    """Every arXiv id the host retrieved, in its own formats (D21, D25).

    ``lit_review`` is the PhD student's, entries keyed ``arxiv_id``.
    ``section_related_work`` is the paper writer's per-section search results
    (``papersolver.py:357-367``). A host passes
    ``lambda: retrieved_arxiv_ids(phd.lit_review, solver.section_related_work)``
    to :func:`report_loop`.
    """
    ids = {str(e["arxiv_id"]).strip() for e in lit_review or [] if e.get("arxiv_id")}
    for text in (section_related_work or {}).values():
        ids.update(_SEARCH_RESULT_ID.findall(text or ""))
    return ids
