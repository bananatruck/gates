# Gate 2 / Gate 3 literature readout

Status: draft for the Thursday readout. Not merged.
Scope: what the cited literature actually says, and the six design questions it bears on.

Every number below was read off a PDF in the `Sources/` folder or off the arXiv source, and
the page or table it came from is named. Where I could not verify a number, it says so.
Nothing here is a measurement of our own system.

The two frozen constraints held: `reports/finalized-report-and-results/` was not touched,
and nothing here treats `gates/retrieval.py` as a paper registry (it is BM25 over the
log-line exemplar bank in `gates/exemplars.py`).

---

## 0. Two corrections to the brief, before anything is built on them

**AutoResearchClaw does not report "42% on audit."** I searched the full text
(`2605.20025v2`). The string `42` appears once in prose, as `42.9%`, and it is the
acceptance rate of the *Thorough* human-in-the-loop intervention mode (Table 3), not an
audit-failure rate. The 42% we are thinking of is almost certainly SAGE's, which is a
different paper measuring a different thing.

What AutoResearchClaw actually reports (Table 5, component ablation, 10 ARC-Bench topics,
best-of-3 over three reruns):

| Configuration | Completion | Quality | Accept | Fabrication found |
|---|---|---|---|---|
| Full AutoResearchClaw | 10/10 | 5.62 | 3/10 | no |
| w/o Verification | 10/10 | 5.48 | 5/10 | **yes** |
| w/o Debate | 10/10 | 4.25 | 1/10 | no |
| w/o Self-Healing | 6/10 | 4.83 | 1/6 | no |
| w/o Evolution | 9/10 | 5.14 | 2/10 | no |
| w/o Debate & Healing | 4/10 | 3.47 | 0/4 | no |

The real sentence is better for us than the one in the brief: *"Removing the verified
registry raises apparent acceptance from 3/10 to 5/10, but manual inspection reveals that
3 of those 5 papers contain values absent from any measurement record."* A registry costs
you 2/10 apparent acceptance and buys back 3 fabricated papers.

**The opening is still there, and it is sharper.** It is not a 42% audit number. It is
AutoResearchClaw's own admission, §4 case study: *"verification is necessary but not
sufficient. Full-Auto passes the numeric gate because the zero values are real logged
measurements, not fabricated numbers. However, the gate cannot tell whether those
measurements answer the research question."* A registry gate that a degenerate all-zero
run walks straight through is the gap. Same gap SAGE names as **method-provenance
grounding** — verifying that methodological claims are backed by executed artifacts —
and explicitly declares open in both itself and its baselines.

That is our opening, stated in the words of the two systems closest to us.

---

## 1. Five SourceClaim entries, hand-pulled (the Friday unblock)

These construct and run through the real `gates.gate2.band_for()`. Output verified below.

```python
from gates.gate2 import SourceClaim

CLAIMS = (
    # CORE-Bench Table A3. 95% CI over mean accuracy, n=3 trials, test set.
    SourceClaim(
        key="core_bench_hard_accuracy",
        source_id="arXiv:2409.11363",
        value=21.48,
        interval=(18.88, 24.08),          # 21.48 +/- 2.60
        setting="CORE-Agent + GPT-4o, CORE-Bench-Hard test split, n=3, 95% CI (Table A3)",
        describes="Agent reproduces a published result from its own code and data capsule.",
    ),
    SourceClaim(
        key="core_bench_easy_accuracy",
        source_id="arXiv:2409.11363",
        value=60.60,
        interval=(56.09, 65.11),          # 60.60 +/- 4.51
        setting="CORE-Agent + GPT-4o, CORE-Bench-Easy test split, n=3, 95% CI (Table A3)",
        describes="Same task with environment pre-built and results pre-extracted.",
    ),
    # Same table, cost column. The only published per-task cost band we have.
    SourceClaim(
        key="core_bench_hard_cost_usd",
        source_id="arXiv:2409.11363",
        value=2.9643,
        interval=(2.8755, 3.0531),        # 2.9643 +/- 0.0888
        setting="CORE-Agent + GPT-4o, CORE-Bench-Hard, mean USD/task, n=3, 95% CI",
        describes="Dollar cost of one agent attempt at one reproduction task.",
    ),
    # PaperBench Table 4. NOTE: one SEM, not a 95% CI. See section 2.
    SourceClaim(
        key="paperbench_replication_score",
        source_id="arXiv:2504.01848",
        value=21.0,
        interval=(20.2, 21.8),            # 21.0 +/- 0.8, ONE SEM over 3 seeds
        setting="Claude 3.5 Sonnet + BasicAgent, PaperBench, 3 seeds, +/-1 SEM (Table 4)",
        describes="Agent replicates a paper from scratch, graded against an author rubric.",
    ),
    # SAGE. Point estimate over 12 topics; the paper reports no interval.
    SourceClaim(
        key="sage_metrics_bearing_rate",
        source_id="arXiv:2606.31478",
        value=91.7,
        interval=None,                    # 11/12 topics. None reported.
        rel_tol=None,                     # undeclared -> falls to default_rel_tol
        setting="SAGE full system, 12-topic 5-domain benchmark, 11/12 metrics-bearing",
        describes="Run whose experiment stage emits >=1 task-relevant measured metric.",
    ),
)
```

Run through `band_for(claim, default_rel_tol=0.05)`:

| key | value | band | origin | width / value |
|---|---|---|---|---|
| core_bench_hard_accuracy | 21.4800 | [18.8800, 24.0800] | `reported_interval` | 24.21% |
| core_bench_easy_accuracy | 60.6000 | [56.0900, 65.1100] | `reported_interval` | 14.88% |
| core_bench_hard_cost_usd | 2.9643 | [2.8755, 3.0531] | `reported_interval` | 5.99% |
| paperbench_replication_score | 21.0000 | [20.2000, 21.8000] | `reported_interval` | 7.62% |
| sage_metrics_bearing_rate | 91.7000 | [87.1150, 96.2850] | `default_relative` | 10.00% |

Two things fall out of that table immediately, and they are section 2.

---

## 2. Headline finding: `SourceClaim.interval` has no kind, and it needs one

`interval` is `tuple[float, float] | None`, and `Band.origin` records
`reported_interval` when it is present. Look at rows 1 and 4 above. Both say
`reported_interval`. They are not the same object:

- CORE-Bench row: a **95% confidence interval** over a mean, n=3.
- PaperBench row: **one standard error of the mean**, n=3. Roughly 68% coverage.

We currently widen a claim by a 68%-coverage band and a 95%-coverage band through the
same code path, label both `reported_interval`, and publish that as the principled option.
That is the overclaim our own duty 3 forbids, committed by us, inside the field whose
entire purpose is to record where a band came from.

Worse, rows differ in kind again: CORE-Bench uses a **95% prediction interval** for
grading task answers and a **95% confidence interval** for reporting agent accuracy. Those
answer different questions (section 8). Both would land in `interval` today.

**Proposed change.** `interval` becomes a small frozen record, not a tuple:

```python
@dataclass(frozen=True)
class ReportedInterval:
    low: float
    high: float
    kind: str       # "ci" | "pi" | "sem" | "sd" | "iqr"
    coverage: float | None   # 0.95 for a 95% CI; None for +/-1 SEM
    n: int | None            # runs the interval was computed from
```

and `Band.origin` becomes `reported_ci_95`, `reported_sem`, ... rather than one flat
`reported_interval`. This is a strictly deterministic change, stays in tier A/B, costs no
model call, and it is the difference between "within tolerance" and "within *whose*
tolerance".

Cheap and worth doing before Friday, because all five entries above are already typed
wrong.

### 2b. The fabricated interval

`Jr.AI Scientist.pdf` (TMLR 02/2026) prints a generated paper's statistics section:

> "All reported results represent averages over 3 random seeds with different data splits.
> The improvements of NPT over LoCoOp are statistically significant (p < 0.05) across all
> datasets using paired t-tests. We also report 95% confidence intervals for AUROC
> improvements: iNaturalist [2.7%, 3.1%], SUN [2.1%, 2.6%], Places365 [1.6%, 2.0%],
> and Texture [1.2%, 1.6%]."

with the authors' margin annotation: **"This is a hallucination, as the experiment was in
fact executed only once."**

An agent will invent a confidence interval, a seed count, and a p-value in one paragraph.
This gives Gate 3 a new check that is *deterministic and provable*, so it sits on the right
side of the honesty boundary:

> **`report.dispersion_supported`** — if the manuscript states a CI, a `±`, an SD, a SEM,
> or a p-value for a key, the registry must record n >= 2 runs for that key. A single-run
> registry cannot license a dispersion claim.

No model needed, no literature needed, FAIL severity, and it catches the exact artifact
above. I think this is the strongest single check in this document.

---

## 3. Where does tolerance come from when a paper gives only a point estimate?

Ranked, most defensible first. The first option is the one we do not currently have.

**(a) The precision the source itself wrote down.** A paper that prints `60.60` has
asserted two decimal places; the value it measured lies in `[60.595, 60.605)`. That
half-unit-in-the-last-place band is *derivable from the point estimate alone*, requires no
assumption, and is the only band the source actually licenses. It is deterministic, so it
belongs in tier B beside `reported_interval`. Proposed `origin="reported_precision"`.

Caveat, and it is a real one: last-place precision measures *how the number was written*,
not how much it would move on a rerun. For `21.48` it gives a band of ±0.005 against a
true 95% CI of ±2.60 — 520x too narrow. So it is a **floor, not a substitute**: it is
sound as a bound on transcription and rounding error, and unsound as a bound on run-to-run
variance. It should never be used to claim agreement with a rerun.

**(b) A declared relative tolerance** (`rel_tol`), recorded as `declared_relative`. What
we have. Honest because it is labelled.

**(c) The default** (`default_rel_tol = 0.05`), recorded as `default_relative`. Also
honest because it is labelled, but see below — the number is wrong.

**(d) Not available: deriving an interval ourselves.** For a proportion like SAGE's 11/12
we *could* compute a Wilson interval. We should not put it in `interval`, because the
source did not report it and `reported_interval` would then be a lie. If we compute one it
must carry its own origin (`derived_wilson`) and say so in the report.

**What the literature does.** SAGE's sanitizer allows **1% relative tolerance** for
rounding when matching a drafted table cell against its measured registry. That is the
closest published analogue to our `default_rel_tol`, and it is 5x tighter than ours —
but note it is doing a *different job*: matching a number to its own measurement, not to a
rerun. Our 5% is in the wrong place for both jobs.

**Our default is falsely confident.** SAGE's 11/12 gets a `default_relative` band of
[87.1, 96.3], width 10.0% of value. The Wilson 95% interval for 11/12 is
[64.6%, 98.5%], width 33.9%. Our default is ~3.4x too narrow for exactly the kind of
small-n proportion the autonomous-research literature is full of. Recommend: do not apply
a relative tolerance to a proportion at all; route proportions to a Wilson interval with
its own origin, and reserve `rel_tol` for continuous metrics.

---

## 4. `PaperRecord` sketch, and the four APIs against it

CLAUDE.md §6 is right that this does not exist and that both Gate 2 tier B and Gate 3
citation binding need it. `SourceClaim.source_id` already promises "Gate 3 requires this
to be in the retrieval registry, so a band can never come from a paper nobody fetched" —
today nothing enforces that promise.

```python
@dataclass(frozen=True)
class PaperRecord:
    """One paper the scaffold actually fetched. Not a bibliography entry."""

    # -- identity: at least one must be present -----------------------------
    arxiv_id: str | None          # "2409.11363", no version
    doi: str | None
    openalex_id: str | None

    # -- what a citation renders as -----------------------------------------
    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str | None

    # -- the part that makes it evidence rather than metadata ---------------
    #: Where the scaffold got it. A record with no source_url is a claim that
    #: a fetch happened, not proof of one.
    source_url: str
    #: UTC ISO-8601. Metadata is not stable; a band cited in March is not
    #: necessarily the band the same call returns in September.
    retrieved_at: str
    #: sha256 of the retrieved bytes (PDF or abstract payload).
    content_sha256: str
    #: Which API answered, so a disputed record can be re-fetched the same way.
    resolver: str                 # "arxiv" | "openalex" | "crossref" | "s2"
    #: Version actually fetched, where the source has versions. "v2" != "v1";
    #: our own Sources folder holds v2 and v3 PDFs whose numbers may differ
    #: from v1.
    version: str | None = None
```

**Where it must live.** Not in `gates/`, and this is not a style preference. A gate that
makes a network call can block, time out, and rate-limit, which breaks *"deterministic
checks decide the verdict"* exactly as a model call would. `PaperRecord` is data; the
*fetching* belongs in the adapter, and records arrive injected, the same way `ModelFn`
does. That also keeps the `rig/` scenario loop runnable with no network and no key, which
is the existing half-two requirement. The dataclass itself is stdlib-only and can sit in
`gates/`; `urllib` calls cannot.

**The four APIs.**

| | arXiv API | OpenAlex | Semantic Scholar | Crossref |
|---|---|---|---|---|
| Auth | none | none (mailto = polite pool) | key advised; harsh 429s without | none (mailto polite) |
| Stdlib-friendly | Atom XML → `xml.etree` | JSON → `json` | JSON | JSON |
| arXiv id → record | native | yes | yes | patchy for preprints |
| DOI → record | no | yes | yes | native, authoritative |
| Version (`v2`/`v3`) | **yes** | no | no | no |
| Coverage beyond arXiv | none | very broad | broad | DOI-bearing only |
| Rate limit risk | low | low | **high** | low |

Recommendation: **OpenAlex primary** (no key, broad, carries both DOI and arXiv id),
**Crossref fallback** for DOI-only published work, **arXiv API** whenever we need the
version — and we do need it, since `Sources/` holds `v2` and `v3` PDFs and a band pulled
from v2 is not a band from v1. **Skip Semantic Scholar**: its key requirement and 429
behaviour is the one dependency here that could make an adapter flaky, and it adds no
field the other three lack.

---

## 5. `gate2.py` bounds speedup below but not above — hard block or literature question?

**Picking: literature question.** Leave `UNIT_RANGES["speedup"]` unbounded above.

The reason is not caution, it is the tier boundary. Tier A is
elimination-by-construction: facts about the numbers alone. "A speedup above N is
implausible" is not a fact about the numbers, it is an empirical prior about what
speedups occur — which is precisely what tier B's reference intervals are *for*. Putting a
prior in tier A would let us claim a provable elimination we cannot prove.

There is also a counterexample in our own Sources folder. SAGE's B07 case study reports an
FVA runtime of 4.74 s, *"about 4,700x FBA."* A hard ceiling anywhere near a plausible
value would have falsely rejected a real, measured, published ratio. `UNIT_RANGES`'
existing comment already commits us to this posture: a false rejection "costs the engineer
a rewrite for nothing."

The right home for the concern is the relation check, which is exact and already exists:
`OPS["ratio"]` verifies a declared speedup against the two recorded times with no prior at
all. Suggested follow-up, deterministic and cheap: **warn when a key with unit `speedup`
has no declared `Relation`** — not because the value is too big, but because an
undeclared speedup is an unverifiable one.

### 5b. But there is a real bug next door, and it is the reason the question came up

```
speedup admits(nan)  = True
speedup admits(inf)  = True
accuracy admits(nan) = False
```

`Range.admits` returns `True` for NaN and `+inf` on every unbounded-above unit —
`speedup`, `loss`, `count`, `s`, `ms`, `sec`, `secs`, `seconds`, `wallclock_s`. The
comparison `value <= high` is what rejects NaN, so bounded units are accidentally safe and
unbounded ones are not.

This matters because `OPS["ratio"]` returns `math.nan` when the denominator is zero, and a
zero denominator is exactly the unmeasured-wallclock case the `low_open` flag was written
to catch: *"a wallclock of exactly 0.0 is not a fast run, it is an unmeasured one."* We
catch the zero time and then let the NaN speedup derived from it pass tier A.
`_check_internal_consistency` does guard `math.isnan(expected)`, but only for a relation
someone declared; with no relation declared, nothing catches it.

Fix is two lines in `admits`, is a fact about the numbers alone, and belongs in tier A:
reject non-finite values for every unit. Per CLAUDE.md §5 this starts with a failing test
that reproduces it.

Note this also answers the framing: an upper bound on speedup *would* have caught the NaN,
but for the wrong reason, and at the cost of false rejections. Fix the finiteness hole;
leave the ceiling alone.

---

## 6. The scanner: what else slips through

`NUMBER = re.compile(r"(\d+\.\d+|\d{2,})")` plus `is_claim()` (needs a decimal point, or
4+ digits and not a year). Probed against the real `extract_claims()`:

| Probe (inside a `## Results` section) | Claims found | Class |
|---|---|---|
| `improved by 9 points` | none | miss |
| `accuracy of 87 percent` | none | miss |
| `a 3x speedup over the baseline` | none | miss |
| `metrics-bearing on 11 of 12 topics` | none | miss |
| `recovery improved 5/12 to 11/12` | none | miss |
| `throughput of 12,345 tokens per second` | none | miss |
| `we observe a two-fold improvement` | none | miss |
| `2000 iterations were run` | none | miss |
| `the effect was -0.42 on average` | **0.42** | **wrong value** |
| `loss fell to 1.2e-3` | **1.2** | **wrong value** |
| `F1 of 42.0 +/- 1.5 across seeds` | **42, 1.5** | **structure lost** |
| `accuracy 21.48 \cite{siegel2024}` | none | **silent line drop** |
| `we report 25.8 micro F1` | 25.8 | control, correct |

Three classes, and they are not equally bad:

1. **Misses.** "9 points" is the one in the brief, but `87 percent` is the bigger hole —
   any 2- or 3-digit integer result is invisible, and integer percentages are everywhere.
   `5/12` and `11 of 12` matter specifically because that is how SAGE and MLR-Bench state
   their headline results, so we cannot currently scan the papers we are comparing to.
2. **Wrong values**, which are worse than misses because they produce a confident
   mismatch rather than a gap. `-0.42` is recorded as `+0.42` (sign dropped — a sign flip
   is a coherence failure, not a rounding one). `1.2e-3` is recorded as `1.2`, off by
   1000x. Either could make a *correct* manuscript fail Gate 3, or let a wrong one pass.
3. **Silent line drop.** `SKIP_LINE` matches `\cite`, so a results sentence carrying an
   inline citation loses *every* number on it, reports nothing, and looks clean. This is
   the same failure shape as the `\section{}` heading bug already documented in
   `prose.py`'s docstring — a scanner reporting zero findings because it could not read
   the input. `claim_sections()` was added to make that visible for headings; nothing
   makes it visible for skipped lines.

Note the docstring's constraint: the published Gate 1 traceability number came from this
scanner reading `.tex`, so the LaTeX path cannot be altered without restating a measured
result. Any fix needs to be additive and measured on both paths.

**AxCell is the reason to be humble here.** Extracting `(task, dataset, metric)` from ML
papers scores 61.9 micro-F1. Adding the *score* to the tuple drops it to **25.8 micro-F1 /
19.7 macro-F1** (Table 1, NLP-TDMS Exp; prior SOTA was 7.5). Getting the number right is
where the task falls apart, and that is exactly our task. A regex will not close that gap;
what saves us is that we are matching against our *own* registry, not extracting
open-domain — which is an argument for making the registry rich (rounded variants,
percentage variants, as both SAGE and AutoResearchClaw do) rather than making the scanner
clever.

---

## 7. MiniCheck is PyTorch; `gates/` is stdlib-only. Where does it live?

**It cannot live in `gates/`, and it does not need to.** MiniCheck-FT5 is 770M parameters
and claims GPT-4-level fact-checking at ~400x lower cost on LLM-AggreFact. Attractive, and
still structurally confined.

The rule that decides this is already written: a model call can never block, fail, or
change a verdict, and `model_warning()` hardcodes `Severity.WARN` with no severity
argument. MiniCheck is a model. However good it is, it can only ever produce a WARN. So
the question "where does it live" has a cheap answer:

- **Not in `gates/`** — that would break the zero-dependency invariant for a component
  that cannot change a verdict anyway. Worst trade in the repo.
- **Behind the existing `ModelFn` injection point**, wrapped by an adapter, if anyone
  wants it in tier C. No new seam, no new invariant. `gates/` never learns it exists.
- **In `rig/`, as an instrument rather than a checker** — this is the use I would argue
  for. `rig/` is where we measure how well tier C does. MiniCheck is a cheap
  sentence-level entailment scorer, which makes it a reasonable *yardstick* for tier C's
  rate-with-an-interval claim, and `rig/` is already outside the packaged surface
  (`pyproject.toml` packages `gates*` only), so a torch dependency there costs the
  published claim nothing.

**Recommendation: punt for Gate 2/3 shipping; keep it as a `rig/` evaluation option.**
The supporting numbers say the tier-C ceiling is low enough that a better tier-C model is
not where the win is:

- **BadScientist**: fabricated papers reach acceptance rates up to **82.0%**; mitigation
  detectors barely beat chance (DetOnly best accuracy **56%** vs 50% random; ReD **67.0%**
  with a 44.9% FPR, and 0.0% TPR on two of three reviewer models). They also name
  *concern-acceptance conflict*: reviewers flag integrity problems and assign
  acceptance-level scores anyway.
- **SPOT**: on 83 published papers with 91 errata/retraction-grade errors, no model
  exceeds **21.1% recall or 6.1% precision**, and across eight runs models rarely
  rediscover the same error.

That last clause is the argument. A verifier that finds different errors on each run
cannot be a gate; it can only be a warning. Which is what our architecture already says.
The Gate 2 docstring's existing note — that tier C "carries BadScientist's near-chance
detection rate, which is why C must never decide anything" — is well supported, and SPOT
strengthens it.

---

## 8. Statistics, one line each

**Prediction vs confidence interval.** A confidence interval bounds where the *mean* lies;
a prediction interval bounds where the *next single observation* lies, so it is always
wider — and CORE-Bench uses both, a 95% PI from three manual runs to grade whether an
agent's reported answer is acceptable, and a 95% CI over three benchmark runs to report
agent accuracy. Gate 2 comparing one of our runs against a literature number wants a
**prediction** interval; we currently take whatever the paper printed, which is usually a
CI.

**TOST (two one-sided tests).** Equivalence testing: instead of failing to reject "they
differ", you declare a tolerance band and reject "they differ by more than the band" from
both sides — which is the correct frame for Gate 2, because *we want to affirm agreement*,
and an ordinary non-significant t-test never licenses that.

**Wilson score interval.** The small-n binomial interval that stays inside [0,1] and does
not collapse at the extremes the way the normal approximation does — mandatory here
because the autonomous-research literature reports 11/12, 8/10, 3/10, and at those n the
normal approximation is unusable.

**Cohen's kappa.** Inter-rater agreement corrected for agreement by chance; relevant to
MLR-Judge's human-agreement validation and AutoResearchClaw's two-reviewer adjudication,
and worth flagging that kappa is depressed by skewed prevalence, so a low kappa on a
corpus that is 80% fabricated does not by itself mean the raters disagree.

### Wilson intervals on the proportions we are citing

| Claim | k/n | point | Wilson 95% | width |
|---|---|---|---|---|
| SAGE metrics-bearing, baseline | 5/12 | 41.7% | [19.3%, 68.0%] | 48.7% |
| SAGE metrics-bearing, full | 11/12 | 91.7% | [64.6%, 98.5%] | 33.9% |
| MLR-Bench, Claude Code fabricated | 8/10 | 80.0% | [49.0%, 94.3%] | 45.3% |
| MLR-Bench, nonexistent citations | 5/10 | 50.0% | [23.7%, 76.3%] | 52.7% |
| ARClaw accept, full | 3/10 | 30.0% | [10.8%, 60.3%] | 49.5% |
| ARClaw accept, w/o verification | 5/10 | 50.0% | [23.7%, 76.3%] | 52.7% |
| CORE-Bench stochastic questions | 17/181 | 9.4% | [5.9%, 14.5%] | 8.6% |

**SAGE's headline 42% → 92% has overlapping 95% intervals** (11/12 lower bound 64.6%
vs 5/12 upper bound 68.0%). The effect is probably real — it is a paired within-topic
comparison, which a two-proportion interval ignores, and McNemar on the paired table would
be the right test — but the *unpaired* reading we would naturally quote in a related-work
sentence is not supported at n=12. If we cite "42 to 92%" we should cite it as 5/12 to
11/12 and say the comparison is paired.

This is also a warning shot for our own evaluation: at n=12 topics, nothing we measure
will separate from a baseline on an unpaired test. Design the comparison paired, per
topic, from the start.

---

## 9. Two more things worth 60 seconds on Thursday

**CORE-Bench's prediction interval covers less than it sounds like.** The 95% PI from
three manual runs applies to **17 of 181 task questions** — the ones with stochastic
answers. For the other 164 the grading is effectively exact match. So "tolerance from 3
runs into a 95% PI" is the right *method* and our best band, but in CORE-Bench it is
load-bearing for under 10% of questions (Wilson [5.9%, 14.5%]). We should not cite it as
though the whole benchmark rests on it.

**MLR-Bench's taxonomy, which we borrow, has four fact-based types**, chosen because they
are objectively verifiable: *Faked Experimental Results*, *Hallucinated Methodology*,
*Incorrect Citations*, *Mathematical Errors*. Mapping to us: type 1 is Gate 1 + Gate 3
registry binding; type 3 is Gate 3 citation binding (blocked on `PaperRecord`, §4); type 4
is Gate 2's `internal_consistency`; **type 2 — hallucinated methodology — is the one no
gate of ours currently touches**, and it is the same thing SAGE calls the
method-provenance gap and declares open. That convergence is worth a slide.

Reported frequencies: faked results and hallucinated methodology each appear in more than
half of 10 tasks, with *"almost all papers generated by AI Scientist V2"* containing both;
nonexistent citations appear in 50% of MLR-Agent tasks. The exact per-type bar values are
in Figure 6, which is a figure — I did not read numbers off it, so those three statements
are the text's, not mine.

---

## Provenance of this document

| Paper | Read from | Verified |
|---|---|---|
| CORE-Bench 2409.11363v2 | `Sources/Benchmarks/` | Table A3, Fig 4 caption |
| PaperBench 2504.01848v3 | `Sources/Benchmarks/` | Tables 4, 5, 6, 9 |
| RE-Bench 2411.15114v2 | `Sources/Benchmarks/` | title only, not used |
| BadScientist 2510.18003 | `Sources/Bad Scientist.pdf` | abstract, Table 4 |
| MLR-Bench 2505.19955 | `Sources/MLR Bench.pdf` | abstract, §5, Appendix B |
| SAGE 2606.31478v1 | `Sources/Relative Results/` | abstract, §3.4, §4.3, App D, App F |
| AutoResearchClaw 2605.20025v2 | `Sources/Relative Results/` | abstract, Table 5, §4, App H/I |
| Jr. AI Scientist (TMLR 02/2026) | `Sources/Jr.AI Scientist.pdf` | App B.3 annotation |
| ScientistOne 2605.26340v1 | `Sources/Relative Results/` | not used |
| MLReplicate | `Sources/MLReplicate.pdf` | not used |
| AxCell 2004.14356 | arXiv (not in `Sources/`) | Table 1 |
| MiniCheck 2404.10774 | arXiv abstract only | 770M, 400x, LLM-AggreFact |
| SPOT 2505.11855 | arXiv abstract only | 83 papers, 91 errors, 21.1%/6.1% |

MiniCheck's specific balanced-accuracy figures are **not** verified — the abstract states
GPT-4-level performance at 400x lower cost without giving the percentage, and I did not
read the full paper. Do not quote a MiniCheck accuracy number on Thursday.

Text was extracted from the PDFs with a throwaway stdlib script under `.cache/`
(gitignored); it is not part of the package and nothing in `gates/` gained a dependency.

---

## Proposed order of work

1. `Range.admits` rejects non-finite values. Failing test first. Smallest, is a real bug. (§5b)
2. `ReportedInterval` kind + coverage on `SourceClaim`. All five entries above are typed wrong today. (§2)
3. `report.dispersion_supported` — a dispersion claim requires n>=2 in the registry. (§2b)
4. `PaperRecord` + an OpenAlex/Crossref/arXiv resolver **in the adapter**, which unblocks Gate 2 tier B provenance and Gate 3 citation binding. (§4)
5. Scanner: negatives and scientific notation first (wrong values), then integer percentages, then make skipped lines visible the way `claim_sections()` made headings visible. (§6)
6. Proportions route to Wilson rather than a relative tolerance. (§3)
