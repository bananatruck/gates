"""Wires Gate 1 into Agent-Researcher / Agent Laboratory's MLE solver.

The insertion point is between code execution and the reward model. Upstream,
``Replace.parse_command`` executed the code and decided success by searching the
(truncated) stdout for a marker string; ``get_score`` then asked an LLM for a
float and treated "a float parsed" as "the run worked". Here, execution goes
through the sandboxed runner, Gate 1 issues the verdict, and ``get_score``
ranks only what already passed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
import re
import time
import urllib.request
from typing import Any, Callable, Iterable
from xml.etree import ElementTree

from pathlib import Path

from ..report import render_evidence
from ..errors import GateError
from ..gate2 import IMPLAUSIBLE_SPEEDUP, PlanField, Range, Relation, SourceClaim
from ..gate3 import RENDERED_FILENAME
from ..schema import PaperRecord
from .. import (
    REGISTRY_FILENAME,
    Gate1Config,
    Gate2Config,
    Gate3Config,
    GateFailure,
    GateReport,
    Ledger,
    build_registry,
    render_feedback,
    render_summary,
    run_experiment,
    run_gate1,
    run_gate2,
    run_gate3,
    unresolved_discrepancies,
)

#: How much raw stdout to inline for the writing agent. Every citable number is
#: in the registry regardless, so this budget bounds context cost without
#: putting any value out of reach. Upstream's equivalent was 1000 characters,
#: and it held the whole experimental record.
STDOUT_BUDGET_CHARS = 40_000


MLE_GATE_INSTRUCTIONS = """
============= RESULTS RECORDING (REQUIRED) =============
Your code runs in a fresh Python process with an empty namespace. Nothing from a
previous version of your code exists. Every name you read must be bound by the
code you submit.

Every number the paper is allowed to report must be recorded with:

    record_result("<key>", <value>, unit="<unit>")

`record_result` is already available - do not import or define it. Use dotted
keys that name the experiment and the quantity, for example:

    record_result("exp1.K2.test_acc", test_acc, unit="ratio")
    record_result("exp2.sgc.wallclock_s", total_sgc_time, unit="seconds")

Pass the VARIABLE holding the measured value. A number typed directly into the
call is rejected: record_result("exp1.K2.test_acc", 0.816) fails the gate,
because a typed number is not a measurement.

Record the seed with record_metadata("seed", seed), along with any other
provenance that is not itself a result. A run with no declared seed cannot be
re-executed to confirm its own numbers, and the report has to say so.

If you record the same key more than once - once per epoch, for instance - the
registry keeps the LAST call, and every call is retained so the difference is
visible. Record the value you intend the paper to report, at the point it is
final. Do not record a running best and describe it as a final result.

Your code is checked before and after it runs. Unbound names, uncaught
exceptions, non-zero exit codes, missing declared keys and hardcoded values all
reject the submission and send it back to you with a report. Printing results is
still useful, but printed values are not citable - only recorded ones are.
"""


#: Gate 3's counterpart to MLE_GATE_INSTRUCTIONS, for the paper writer (D31).
#: A test checks every token shown here is one the gate's own patterns read.
REPORT_GATE_INSTRUCTIONS = r"""
============= RESULT CITATION (REQUIRED) =============
Every number that reports a result of this study must be written as a token:

    \result{<key>}

using a key from the verified results, for example \result{exp1.K2.test_acc}.
The renderer replaces each token with the value exactly as it was measured.

Do not type a result number yourself, and do not round one. A number typed
into a findings section (abstract, results, discussion, conclusion) is
rejected, because a number with no key was never measured.

Every Results section must cite at least one recorded value. Describing
results without numbers is rejected too.

If the verification layer declared limitations, write \limitations{} where the
paper discusses its limitations. The renderer inserts them word for word. Do
not paraphrase them or leave them out.

Your manuscript is checked before it is accepted. A rejected manuscript comes
back to you with a report naming what to change.
"""


@dataclass
class GateContext:
    """Per-phase gate state: budget, ledger, and what has passed so far."""

    config: Gate1Config | Gate2Config | Gate3Config
    ledger: Ledger | None = None
    phase: str = "running experiments"
    reward_model: str | None = None
    #: Score gate-rejected attempts anyway, purely to record the disagreement.
    #: This is the paper's central evidence table; it costs one model call per
    #: rejection.
    shadow_reward_on_reject: bool = True

    #: Executions performed. Names artifact directories; not the budget.
    attempt: int = 0
    #: Agent-level rewrites rejected in a row. This is what the budget counts —
    #: a single rewrite may spend several executions on automated repair, and
    #: those must not eat the ML engineer's turns.
    consecutive_rejections: int = 0
    has_passing_attempt: bool = False
    last_report: GateReport | None = None
    history: list[GateReport] = field(default_factory=list)

    @property
    def budget_exhausted(self) -> bool:
        return self.consecutive_rejections >= self.config.max_attempts

    def check_can_continue(self) -> None:
        """Raise once the budget is spent and nothing ever passed.

        A run that never produced a valid experiment must not produce a paper.
        If something did pass, the caller falls back to it instead.
        """
        if self.budget_exhausted and not self.has_passing_attempt:
            raise GateFailure(
                gate=self.last_report.gate if self.last_report else "GATE",
                attempts=self.consecutive_rejections,
                report=self.last_report,
            )

    def note(self, report: GateReport) -> None:
        """Record one execution. Does not move the budget."""
        self.last_report = report
        self.history.append(report)
        if report.passed:
            self.has_passing_attempt = True

    def close_turn(self, passed: bool) -> None:
        """Close one ML engineer turn, however many executions it took."""
        self.consecutive_rejections = 0 if passed else self.consecutive_rejections + 1


@dataclass
class GatedExecution:
    """What the solver gets back in place of a raw stdout string."""

    report: GateReport | None
    feedback: str
    evidence_bundle: str
    code: str = ""
    #: Set only on the bypass path, where there is no report to ask.
    ungated_passed: bool | None = None

    @property
    def passed(self) -> bool:
        if self.report is None:
            # Bypass: upstream's rule, which is a substring search over the
            # already-truncated view. Not a stand-in for it — the same test.
            return bool(self.ungated_passed)
        return self.report.passed

    @property
    def summary(self) -> str:
        if self.report is None:
            return f"gate bypassed — {'accepted' if self.passed else 'rejected'}"
        return render_summary(self.report)

    def legacy_view(self, max_len: int = 1000) -> str:
        """Reconstruct exactly what upstream's ``execute_code`` would have returned.

        Upstream appended its error marker to the *end* of the capture buffer and
        then sliced ``[:MAX_LEN]``, so on any run that printed more than
        ``max_len`` characters the marker fell off and the crash became
        invisible. Reproducing that view lets the ledger answer the
        counterfactual — what would the reward model have scored, given the
        channel the scaffold actually had? — instead of guessing at it.
        """
        if self.report is None:
            # The gate-off control already holds exactly this view.
            return self.evidence_bundle
        execution = self.report.execution
        if execution is None:
            return ""
        buffer = execution.stdout_text()
        exc = execution.exception
        if exc is not None:
            buffer += f"[CODE EXECUTION ERROR]: {exc.message}\n{exc.traceback}"
        return buffer[:max_len]


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


#: arXiv's Atom API. The one endpoint anything in this project contacts.
_ARXIV_API = "http://export.arxiv.org/api/query?id_list={}&max_results=1"

#: arXiv asks for roughly one request every three seconds. Enforced between
#: requests rather than as a per-call sleep, so a cache hit costs nothing.
_ARXIV_MIN_INTERVAL_S = 3.0

_ATOM = "{http://www.w3.org/2005/Atom}"

#: Trailing version, stripped before the request (D26). Asking arXiv for a
#: specific version would make a v4 citation of a paper the run read as v2 look
#: like a different paper, which is the registry check's question, not this one's.
_ARXIV_VERSION = re.compile(r"v\d+$")


def _fetch_url(url: str) -> str:
    """One GET, stdlib only. Raises on anything that is not a 200 with a body.

    Separated so the resolver can be tested without a socket: every test above
    passes its own ``fetch``, and this is what the real one does.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "gates-validity-layer (citation resolution)"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def arxiv_lookup(
    *,
    cache_dir: str = ".cache/arxiv",
    fetch: Callable[[str], str] = _fetch_url,
) -> Callable[[str], PaperRecord | None]:
    """A resolver for ``Gate3Config.lookup``, backed by arXiv and a disk cache.

    Lives in the adapter by D41, because ``gates/`` never opens a socket and
    ``rig/`` is the model-free scenario loop. The gate receives only the returned
    function, so it cannot tell arXiv from the dict-backed fake the suite uses.

    Three behaviours the gate depends on:

    * ``None`` means arXiv has no such paper. That is a verdict.
    * **Raising** means the question could not be asked. Gate 3 turns that into
      an INFO row saying citations went unchecked, never a rejection: an outage
      is not a defect in the manuscript.
    * A resolved *and* an absent answer are both cached; a failure is not. A
      cached outage would keep citations unchecked after the network returned,
      and an uncached absence would cost one request per fabricated citation on
      every turn of the retry loop, which is the case the loop exists to retry.
    """
    path = Path(cache_dir) / "records.json"
    cache: dict[str, dict[str, Any] | None] = {}
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # The cache is a convenience. A truncated write from an earlier run must
        # not take the resolver down, and through it the whole writing phase.
        cache = {}
    last_request: list[float | None] = [None]

    def lookup(identifier: str) -> PaperRecord | None:
        key = _ARXIV_VERSION.sub("", identifier.strip())
        if key in cache:
            row = cache[key]
            if row is None:
                return None
            record = _as_record(row)
            if record is not None:
                return record
            # A row this cannot read is a miss, not an error. Raising would
            # reach Gate 3 as "citations went unchecked", which would hide a
            # corrupt cache behind an outage message.

        if last_request[0] is not None:
            wait = _ARXIV_MIN_INTERVAL_S - (time.monotonic() - last_request[0])
            if wait > 0:
                time.sleep(wait)
        body = fetch(_ARXIV_API.format(key))
        last_request[0] = time.monotonic()

        entry = ElementTree.fromstring(body).find(f"{_ATOM}entry")
        cache[key] = None if entry is None else _parse_entry(entry, key, body)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            # An unwritable cache costs requests, not correctness.
            pass
        row = cache[key]
        return None if row is None else _as_record(row)

    return lookup


def _parse_entry(entry: Any, key: str, body: str) -> dict[str, Any]:
    """One Atom entry as a cache row. Missing fields stay empty, never guessed."""
    published = (entry.findtext(f"{_ATOM}published") or "").strip()
    return {
        "identifier": key,
        "title": " ".join((entry.findtext(f"{_ATOM}title") or "").split()),
        "authors": [
            " ".join((a.findtext(f"{_ATOM}name") or "").split())
            for a in entry.findall(f"{_ATOM}author")
        ],
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "locator": (entry.findtext(f"{_ATOM}id") or "").strip(),
        # What was retrieved, so "the same paper" is checkable later. Hashing the
        # response rather than the parsed fields catches a metadata correction.
        "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def _as_record(row: Any) -> PaperRecord | None:
    """A cache row as a record, or ``None`` if the row cannot be read as one.

    Returns rather than raises so a corrupt cache is a miss and gets refetched.
    A raise here would surface in Gate 3 as an unreachable resolver, which would
    report citations as unchecked when the network was fine.
    """
    if not isinstance(row, dict) or not isinstance(row.get("title"), str):
        return None
    year = row.get("year")
    return PaperRecord(
        identifier=str(row.get("identifier") or ""),
        title=row["title"],
        authors=tuple(row.get("authors") or ()),
        year=year if isinstance(year, int) else None,
        locator=str(row.get("locator") or ""),
        content_hash=str(row.get("content_hash") or ""),
    )


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


def gate1_enabled() -> bool:
    """Whether Gate 1 arbitrates, read from the environment.

    Set ``GATES_GATE1=off`` to run the host exactly as shipped. This exists so
    the two arms of a full-workflow comparison can differ in the gate and in
    nothing else. Checking out the pre-Gate-1 branch instead would also change
    the model plumbing, the rate-limit backoff and the prompts, and any
    difference in the papers could then be attributed to those.

    The name says Gate 1 because Gate 1 shipped first, but a host reads it as
    the whole layer's switch: Gate 2 reviews Gate 1's registry and Gate 3
    judges a manuscript against it, so neither has an input with Gate 1 off.
    Agent Laboratory skips both when this returns ``False`` (D53). Renaming it
    would strand the published Gate 1 evidence and the ablation runner that
    produced it, which is a worse trade than one paragraph.
    """
    return os.environ.get("GATES_GATE1", "on").strip().lower() not in {
        "off", "0", "false", "no"
    }


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
    print(f"$$$$ gate 1 BYPASSED (GATES_GATE1=off) — attempt {context.attempt}, "
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
    """Execute ``code`` under Gate 1 and return the verdict plus its artifacts."""
    if not gate1_enabled():
        return _ungated_execute(code, context)

    context.attempt += 1
    report = run_gate1(code, context.config, attempt=context.attempt)
    report.rewrite = context.consecutive_rejections + 1
    context.note(report)

    print(f"$$$$ {render_summary(report)}")

    return GatedExecution(
        report=report,
        feedback=render_feedback(report),
        evidence_bundle=build_evidence_bundle(report),
        code=code,
    )


def gated_review(registry: dict[str, Any], context: GateContext) -> GatedExecution:
    """Review a Gate 1 registry under Gate 2 and return the verdict.

    The mirror of :func:`gated_execute` one phase later. The subject is the
    registry Gate 1 wrote rather than source code, and the budget is spent the
    same way: ``rewrite`` is set from the context so the feedback header counts
    agent turns rather than reporting "attempt 1" forever.

    The caller does **not** call ``check_can_continue`` after this. Gate 2's
    policy on a spent budget is to proceed with the discrepancies declared, so
    the loop keeps going and the limitations travel forward in the bundle.
    """
    context.attempt += 1
    report = run_gate2(registry, context.config, attempt=context.attempt)
    report.rewrite = context.consecutive_rejections + 1
    context.note(report)

    print(f"$$$$ {render_summary(report)}")

    return GatedExecution(
        report=report,
        feedback=render_feedback(report),
        evidence_bundle=declared_limitations(report),
    )


def declared_limitations(report: GateReport) -> str:
    """What the manuscript has to disclose, whether or not Gate 2 passed.

    Gate 2 proceeds on exhaustion, so its findings must travel into the writing
    phase rather than stopping the run. An empty result means there is nothing
    to declare, not that nothing was checked - the report says which checks ran.
    """
    limitations = unresolved_discrepancies(report)
    if not limitations:
        return ""
    lines = ["DECLARED LIMITATIONS", ""]
    lines.extend(f"  - {item}" for item in limitations)
    return "\n".join(lines) + "\n"


@dataclass
class ReviewOutcome:
    """How Gate 2's feedback loop ended, and what the writer must disclose."""

    #: "pass" - a registry was admitted. "proceeded" - a budget was spent, Gate
    #: 2's or Gate 1's after a review, and the run continues with the last
    #: review's discrepancies declared. "no_pass" - ``revise`` stopped before
    #: either budget did.
    outcome: str = "no_pass"
    #: From the last review, on every exit. Empty means nothing to declare, not
    #: that nothing was checked.
    declared: str = ""
    #: The registry of the run Gate 2 last reviewed: what the writer cites and
    #: Gate 3 checks against. ``None`` when nothing was reviewed.
    registry: dict[str, Any] | None = None
    #: The Gate 1 run that registry came from, ``first`` included. Its ``code``
    #: and ``evidence_bundle`` are what the writer describes; the code the phase
    #: started with is stale once a revision is reviewed.
    run: GatedExecution | None = None
    reviews: list[GatedExecution] = field(default_factory=list)
    #: Every Gate 1 run this loop made, in order, passed or not. ``first`` is
    #: not among them: the loop reviewed it but did not run it.
    executions: list[GatedExecution] = field(default_factory=list)


def review_loop(
    context: GateContext,
    revise: Callable[[str | None], str | None],
    *,
    gate1: GateContext,
    first: GatedExecution | None = None,
    extra: dict[str, Any] | None = None,
) -> ReviewOutcome:
    """Gate 2's feedback loop, tier C: the one call site a host needs.

    ``revise(feedback)`` returns the next version of the experiment's code, or
    ``None`` to stop. Each version runs under Gate 1 on ``gate1`` first, and
    Gate 2 reviews only the registry Gate 1 built from that run, so a fix cannot
    reach Gate 2 without running (F12). A Gate 1 rejection sends Gate 1's report
    back and costs a Gate 1 turn, not a Gate 2 one.

    ``first`` is the Gate 1 pass the host already holds from its experiment
    phase. It is reviewed without running again, and ``revise`` is first called
    with that review's feedback. Without it, ``revise`` is first called with
    ``None`` and its code is the first submission.

    Bounded by both budgets. A spent Gate 2 budget does not raise (`CLAUDE.md`
    §4). A spent Gate 1 budget raises ``GateFailure`` if no revision ever ran
    clean, and otherwise ends the loop with the last review declared.
    """
    if not gate1_enabled():
        raise GateError("Gate 2 reviews Gate 1's registry, and GATES_GATE1 is off")
    if first is not None and (first.report is None or not first.passed):
        raise GateError("first must be a run Gate 1 passed; Gate 1's rejection stands")
    result = ReviewOutcome()
    feedback: str | None = None
    executed = first
    while executed is not None or (code := revise(feedback)) is not None:
        if executed is None:
            executed = gated_execute(code, gate1)
            result.executions.append(executed)
            # review_turn, not turn: loop_summary reads turn and counts Gate 2 only.
            record_divergence(
                gate1,
                executed.report,
                reward_score=None,
                extra={"review_turn": len(result.reviews), **(extra or {})},
            )
            gate1.close_turn(executed.passed)
            if not executed.passed:
                # Nothing ever passed: Gate 1 raises, as it does outside this loop.
                gate1.check_can_continue()
                if gate1.budget_exhausted:
                    # Something passed and Gate 2 reviewed it, so that review's
                    # discrepancies stand declared, as on a spent Gate 2 budget.
                    result.outcome = "proceeded" if result.reviews else "no_pass"
                    break
                feedback = executed.feedback
                executed = None
                continue
        registry = build_registry(executed.report, task_ref=gate1.config.task_ref)
        run, executed = executed, None
        reviewed = gated_review(registry, context)
        record_divergence(
            context,
            reviewed.report,
            reward_score=None,
            extra={
                "turn": len(result.reviews),
                "max_attempts": context.config.max_attempts,
                **(extra or {}),
            },
        )
        context.close_turn(reviewed.passed)
        result.reviews.append(reviewed)
        result.registry, result.run = registry, run
        # Set on every turn, so a revise that gives up still hands the writer
        # the discrepancies it was sent (F4).
        result.declared = reviewed.evidence_bundle
        if reviewed.passed:
            result.outcome = "pass"
            break
        if context.budget_exhausted:
            result.outcome = "proceeded"
            break
        feedback = reviewed.feedback
    return result


def gated_report(
    source: str,
    registry: dict[str, Any],
    context: GateContext,
    *,
    declared: str = "",
    retrieved: Iterable[str] | None = None,
) -> GatedExecution:
    """Judge one manuscript under Gate 3 and return the verdict.

    The mirror of :func:`gated_execute` one phase later (D30). ``source`` is the
    writer's output with its ``\\result{}`` and ``\\limitations{}`` tokens
    intact. ``declared`` is ``ReviewOutcome.declared``, and ``retrieved`` the
    paper ids the host's searches returned (:func:`retrieved_arxiv_ids`). The
    evidence bundle is
    the render Gate 3 wrote to disk, so what a host publishes is byte for byte
    what the gate judged.
    """
    context.attempt += 1
    report = run_gate3(
        source,
        registry,
        context.config,
        attempt=context.attempt,
        declared=declared,
        retrieved=retrieved,
    )
    report.rewrite = context.consecutive_rejections + 1
    context.note(report)

    print(f"$$$$ {render_summary(report)}")

    return GatedExecution(
        report=report,
        feedback=render_feedback(report),
        evidence_bundle=(Path(report.artifact_dir) / RENDERED_FILENAME).read_text(
            encoding="utf-8"
        ),
        code=source,
    )


@dataclass
class ReportOutcome:
    """How Gate 3's feedback loop ended. No "proceeded": a spent budget raises."""

    #: "pass" - a manuscript was admitted. "no_pass" - ``write`` stopped first.
    outcome: str = "no_pass"
    #: The admitted manuscript, rendered. The only text a host should publish,
    #: and ``None`` unless Gate 3 passed it.
    manuscript: str | None = None
    reports: list[GatedExecution] = field(default_factory=list)


def report_loop(
    context: GateContext,
    write: Callable[[str | None], str | None],
    *,
    registry: dict[str, Any],
    declared: str = "",
    retrieved: Callable[[], Iterable[str]] | None = None,
    extra: dict[str, Any] | None = None,
) -> ReportOutcome:
    """Gate 3's feedback loop: the one call site a host's writing phase needs.

    ``write(feedback)`` returns the next manuscript with its ``\\result{}``
    tokens intact, or ``None`` to stop, and is first called with ``None``.
    ``registry`` is the one the writer cites: ``ReviewOutcome.registry`` when
    Gate 2 ran, Gate 1's otherwise. ``declared`` is ``ReviewOutcome.declared``,
    the limitations the manuscript must state; empty means none. ``retrieved``
    returns the paper ids the host has retrieved so far and is called after each
    ``write``, because the reference host searches while it writes (D32);
    ``None`` leaves citations unchecked. Taking both
    rather than a ``ReviewOutcome`` keeps Gate 3 usable by a host that does not
    run Gate 2.

    A spent budget raises ``GateFailure`` (`CLAUDE.md` §4): an unverifiable
    manuscript is not emitted.
    """
    if not registry.get("citable"):
        # run_gate3 refuses it too, but only after the writer spent a turn.
        raise GateError("Gate 1 did not pass, so there is nothing a manuscript may cite")
    result = ReportOutcome()
    feedback: str | None = None
    while (source := write(feedback)) is not None:
        written = gated_report(
            source,
            registry,
            context,
            declared=declared,
            retrieved=None if retrieved is None else retrieved(),
        )
        record_divergence(
            context,
            written.report,
            reward_score=None,
            extra={
                "turn": len(result.reports),
                "max_attempts": context.config.max_attempts,
                **(extra or {}),
            },
        )
        context.close_turn(written.passed)
        result.reports.append(written)
        if written.passed:
            result.outcome = "pass"
            result.manuscript = written.evidence_bundle
            break
        # Nothing earlier passed, or the loop would have ended, so this raises
        # whenever the budget is spent.
        context.check_can_continue()
        feedback = written.feedback
    return result


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


def build_evidence_bundle(report: GateReport, budget: int = STDOUT_BUDGET_CHARS) -> str:
    """The evidence handed downstream in place of a 1000-character prefix.

    The verified registry comes first and is complete. Raw stdout follows, and
    is the only part subject to a budget — no citable value lives there that is
    not already in the registry above it.
    """
    if not report.passed:
        return f"[GATE 1 REJECTED THIS RUN]\n\n{render_feedback(report)}"

    metrics = report.metrics()
    execution_ = report.execution
    lines = [
        f"VERIFIED RESULTS — Gate 1 PASS (attempt {report.attempt})",
        "",
        "Every value below was recorded by record_result() during the run that",
        "produced the code above, and traced back to the line that computed it.",
        "These are the only numbers that may be reported.",
        "",
    ]
    if execution_ and execution_.run_id:
        lines += [
            f"  run {execution_.run_id}   code {(report.code_sha256 or '')[:12]}",
            "",
        ]
    if metrics:
        width = max(len(k) for k in metrics)
        for key in sorted(metrics):
            m = metrics[key]
            unit = f"  [{m.unit}]" if m.unit else ""
            where = f"  (experiment.py:{m.lineno})" if m.lineno else ""
            lines.append(f"  {key.ljust(width)} = {m.value}{unit}{where}")
    else:
        lines.append("  (none recorded)")

    execution = report.execution
    if execution and execution.metadata:
        lines += ["", "RUN METADATA"]
        lines += [f"  {k} = {v}" for k, v in sorted(execution.metadata.items())]

    warnings = report.warnings()
    if warnings:
        lines += ["", "WARNINGS — these must be stated in the report, not omitted"]
        for check in warnings:
            lines.append(f"  [{check.id}] {check.message}")
            # The evidence, not only the count. Telling the writer that "1 line
            # reports trouble the run continued past" and then withholding the
            # line asks it to disclose something it cannot see — which is this
            # layer's own failure mode, reproduced at its exit. render_feedback
            # already renders these for the engineer; the writer needs them at
            # least as much, because it is the one making the claim.
            lines += [f"    {row}" for row in render_evidence(check)]

    if report.artifact_dir:
        lines += [
            "",
            "PROVENANCE",
            f"  registry: {os.path.join(report.artifact_dir, REGISTRY_FILENAME)}",
        ]

    if execution:
        stdout = execution.stdout_text()
        lines += [
            "",
            f"FULL EXPERIMENT OUTPUT ({execution.stdout_bytes:,} bytes, untruncated "
            f"at {execution.stdout_path})",
            "",
            _fit(stdout, budget),
        ]
    return "\n".join(lines)


def record_divergence(
    context: GateContext,
    report: GateReport,
    reward_score: float | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log what the gate decided next to what the reward model said."""
    if context.ledger is None:
        return
    context.ledger.record_attempt(
        report,
        phase=context.phase,
        reward_score=reward_score,
        reward_model=context.reward_model,
        extra=extra,
    )


def _fit(text: str, budget: int) -> str:
    """Keep the head and the tail; say exactly how much was elided."""
    if len(text) <= budget:
        return text
    half = budget // 2
    elided = len(text) - budget
    return (
        f"{text[:half]}\n\n"
        f"    [... {elided:,} characters elided from the middle of stdout. "
        f"All recorded results appear in the registry above; the complete "
        f"output is on disk ...]\n\n"
        f"{text[-half:]}"
    )
