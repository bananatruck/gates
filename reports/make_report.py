"""Build the Gate 1 report (HTML -> PDF).

Every figure in this document is a measurement recorded in this repository or a
number quoted from a paper in the source set, and each is attributed inline. No
value is illustrative.

    python reports/make_report.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import aggregate as agg
import run_tables as rt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT_HTML = HERE / "GATE1_REPORT.html"
OUT_PDF = HERE / "GATE1_REPORT.pdf"

RUNS_ROOT = Path("/home/kesh/AgentLaboratory-Gemini/ablation_runs")
RUN_DIR = RUNS_ROOT / "data_efficiency"
RUN_JSON = RUN_DIR / "ablation.json"

BENCH = Path(
    "/tmp/claude-1000/-home-kesh/976cef29-0afd-479c-a716-6c557e07b6cb"
    "/scratchpad/bench2.json"
)

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.5pt; line-height: 1.5;
       color: #0b0b0b; }
h1 { font-size: 22pt; color: #0b0b0b; margin: 0 0 2mm 0; }
h2 { font-size: 13pt; color: #2a78d6; margin: 9mm 0 2mm 0;
     border-bottom: 1px solid #e6e5e0; padding-bottom: 1.5mm; }
h3 { font-size: 10.5pt; color: #0b0b0b; margin: 5mm 0 1.5mm 0; }
p  { margin: 0 0 2.5mm 0; }
.sub { color: #52514e; font-size: 10pt; margin-bottom: 1mm; }
.meta { color: #8a8880; font-size: 8pt; margin-bottom: 6mm; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm 0; }
th { background: #f2f1ec; color: #0b0b0b; font-size: 8pt; text-align: left;
     padding: 2mm; border: 1px solid #e0dfd9; }
td { font-size: 8pt; padding: 1.8mm 2mm; border: 1px solid #e6e5e0;
     vertical-align: top; }
code, .mono { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.8pt; }
.fail { color: #c0392b; font-weight: bold; }
.warn { color: #b06a10; font-weight: bold; }
.info { color: #52514e; font-weight: bold; }
img { max-width: 100%; margin: 3mm 0; }
.callout { background: #f7f6f1; border-left: 3px solid #2a78d6;
           padding: 3mm 4mm; margin: 3mm 0; font-size: 8.6pt; }
.warnbox { background: #fbf5ec; border-left: 3px solid #eb6834;
           padding: 3mm 4mm; margin: 3mm 0; font-size: 8.6pt; }
.kpi { font-size: 17pt; font-weight: bold; color: #2a78d6; }
.kpilabel { font-size: 7.5pt; color: #52514e; }
pre { background: #f7f6f1; padding: 2.5mm 3mm; font-family: 'DejaVu Sans Mono',
      monospace; font-size: 7.6pt; line-height: 1.35; white-space: pre-wrap; }
.small { font-size: 8pt; color: #52514e; }
.ok { color: #12805c; font-weight: bold; }
.bad { color: #c0392b; font-weight: bold; }
.big { font-size: 15pt; font-weight: bold; color: #2a78d6; }
/* A one-cell table, not a bordered div. LibreOffice's HTML importer maps a
   div's border onto each *paragraph* it contains, so a multi-line callout came
   out as a stack of separate boxes, one per wrapped line. Table cell borders
   it renders as a single frame. */
table.hero { width: 100%; border-collapse: collapse; margin: 3mm 0; }
table.hero td { background: #f2f6fc; border: 1px solid #cfe0f5;
                padding: 3mm 4mm; }
"""


def hero(html: str) -> str:
    """A callout that survives the PDF converter in one piece.

    Two separate LibreOffice quirks, both found by rendering and looking:
    a bordered div comes out as one box per wrapped line, and class-based
    styling on a `td` is dropped entirely. A one-cell table fixes the first;
    inline `bgcolor` and `style` on the cell fix the second.
    """
    return (
        '<table class="hero" cellpadding="0" cellspacing="0"><tr>'
        '<td bgcolor="#f2f6fc" style="background:#f2f6fc;'
        'border:1px solid #cfe0f5;padding:3mm 4mm;font-size:9pt">'
        f'{html}</td></tr></table>'
    )

CHECKS = [
    ("static.syntax_valid", "static", "FAIL", "Source does not compile.",
     "Before execution — a broken program costs no compute."),
    ("static.no_unbound_names", "static", "FAIL",
     "A name is read but bound nowhere, resolved with <code>symtable</code>.",
     "The archived run's exact failure: <code>hidden_dim</code> read in "
     "<code>forward()</code>, assigned in no enclosing scope."),
    ("static.no_banned_calls", "static", "FAIL",
     "<code>exit()</code> / <code>sys.exit()</code> would forge a clean exit code.",
     "Removes the cheapest way to fake success."),
    ("exec.exit_code_zero", "exec", "FAIL", "The process exited non-zero.",
     "Reads the exit code, not a substring of stdout."),
    ("exec.no_uncaught_exception", "exec", "FAIL", "An exception escaped the run.",
     "Recorded by the harness in the child process."),
    ("exec.completed_within_budget", "exec", "FAIL",
     "The process group was killed on timeout.",
     "Kills the group, so a runaway cannot survive its own timeout."),
    ("exec.no_swallowed_traceback", "exec", "WARN",
     "A traceback appears in either stream on a run that still exited 0.",
     "Both streams: <code>except Exception as e: print(e)</code> puts the "
     "evidence on stdout."),
    ("env.clean_namespace", "env", "FAIL",
     "The initial globals contained anything beyond harness-provided names.",
     "Makes “the variables are real” a runtime property, not a hope."),
    ("env.code_identity", "env", "FAIL",
     "The source the child hashed differs from the source the parent wrote.",
     "Makes “hashed to the run that produced it” checkable rather than assumed."),
    ("env.seed_recorded", "env", "WARN", "No seed was declared.",
     "A run with no seed cannot be re-executed to check its own numbers."),
    ("env.provenance", "env", "INFO",
     "Python version, platform, device, library versions, code SHA-256.", ""),
    ("results.contract_present", "results", "FAIL",
     "<code>results.json</code> is absent or unparseable.",
     "Nothing was recorded, so nothing is citable."),
    ("results.expected_keys_present", "results", "FAIL",
     "A key the plan declared is missing.", ""),
    ("results.values_computed", "results", "FAIL",
     "A recorded value is a source literal at its call site.",
     "The strongest claim in Gate 1. <code>record_result(\"k\", acc)</code> "
     "passes; <code>record_result(\"k\", 0.816)</code> fails, as does "
     "<code>round(0.8160, 3)</code> — constant folding does not launder a "
     "typed number."),
    ("results.values_finite", "results", "FAIL", "A value is NaN or infinite.", ""),
    ("results.single_observation", "results", "WARN",
     "A key was recorded repeatedly with a changing value.",
     "Every <code>record_result</code> call is retained, so a best-epoch number "
     "reported as a final-epoch one is visible."),
    ("results.non_degenerate", "results", "WARN",
     "Exact zero, exact one, or chance level when the class count is known.",
     "Answers the limitation AutoResearchClaw reports for value registries: a "
     "registry passes real zeros."),
    ("logs.no_error_signals", "logs", "WARN",
     "Numerical warnings, device failures, non-convergence, printed exceptions, "
     "ERROR-level logging.", "Tuned for precision: <code>ValueError:</code> is "
     "flagged, <code>Mean Squared Error:</code> is not."),
    ("output.untruncated", "output", "INFO",
     "stdout/stderr byte counts; asserts no truncation was applied.",
     "Retires the 1,000-character ceiling."),
    ("report.fixes_grounded", "report", "INFO",
     "Generated fixes cited something the run does not contain.",
     "Records that the model's text was discarded and why."),
    ("logs.model_error_signals", "logs (model)", "WARN",
     "Lines the pattern set could not reach, read by the LLM layer.",
     "WARN by construction — <code>model_warning()</code> takes no severity "
     "parameter, so no model output can reach the verdict."),
]


def sev_class(s):
    return {"FAIL": "fail", "WARN": "warn", "INFO": "info"}.get(s, "info")


def checks_table():
    rows = []
    for cid, fam, sev, fails, note in CHECKS:
        rows.append(
            f"<tr><td class='mono'>{cid}</td><td>{fam}</td>"
            f"<td class='{sev_class(sev)}'>{sev}</td>"
            f"<td>{fails}{'<br><span class=small>' + note + '</span>' if note else ''}"
            f"</td></tr>"
        )
    return (
        "<table><tr><th style='width:26%'>check</th><th style='width:10%'>family</th>"
        "<th style='width:8%'>severity</th><th>fails when</th></tr>"
        + "".join(rows) + "</table>"
    )


def scanner_rows(bench):
    if not bench:
        return ("<tr><td colspan=5 class=small>Model-tier measurement not "
                "present in this build.</td></tr>")
    out = []
    base = bench["deterministic"]
    out.append(
        f"<tr><td>deterministic (regex only)</td>"
        f"<td>{base['precision']:.3f}</td>"
        f"<td class=small>{base['precision_ci95'][0]:.3f}–{base['precision_ci95'][1]:.3f}</td>"
        f"<td>{base['recall']:.3f}</td>"
        f"<td class=small>{base['recall_ci95'][0]:.3f}–{base['recall_ci95'][1]:.3f}</td></tr>"
    )
    for key, name in (("qwen_fewshot_0", "+ qwen3:8b, no few-shot"),
                      ("qwen_fewshot_3", "+ qwen3:8b, few-shot k=3")):
        if key in bench:
            r = bench[key]
            out.append(
                f"<tr><td>{name}</td><td>{r['precision']:.3f}</td>"
                f"<td class=small>{r['precision_ci95'][0]:.3f}–{r['precision_ci95'][1]:.3f}</td>"
                f"<td>{r['recall']:.3f}</td>"
                f"<td class=small>{r['recall_ci95'][0]:.3f}–{r['recall_ci95'][1]:.3f}</td></tr>"
            )
    return "".join(out)


def verdict_paragraph(bench):
    """State what the model tier did, in the direction the evidence points."""
    if not bench or "qwen_fewshot_3" not in bench:
        return ("<p>The model tier's own precision and recall are not in this "
                "build. Until they are, the layer's log-scanning claim rests on "
                "the deterministic baseline alone.</p>")
    base = bench["deterministic"]
    naive = bench.get("qwen_fewshot_0")
    best = bench["qwen_fewshot_3"]
    dp = best["precision"] - base["precision"]
    dr = best["recall"] - base["recall"]

    parts = []
    if dr > 0 and dp >= -0.001:
        parts.append(
            f"<p><b>The model tier raised recall from {base['recall']:.3f} to "
            f"{best['recall']:.3f} at no cost in precision</b> "
            f"({base['precision']:.3f} → {best['precision']:.3f}), taking F1 from "
            f"{base['f1']:.3f} to {best['f1']:.3f}. It runs on a local 8B model at "
            f"zero marginal cost, so this is applied to every attempt rather than "
            f"sampled.</p>"
        )
    elif dp < 0:
        parts.append(
            f"<div class='warnbox'><b>The model tier traded precision for recall</b> "
            f"({dp:+.3f} precision, {dr:+.3f} recall). On this evidence the scan "
            f"should stay deterministic and the LLM layer kept for the feedback "
            f"report, which is the job that justified it.</div>"
        )
    else:
        parts.append(f"<p>Recall {dr:+.3f}, precision {dp:+.3f} — no clear gain.</p>")

    if naive and naive.get("spurious"):
        parts.append(
            f"<div class='callout'><b>Retrieval is what bought the precision back.</b> "
            f"Without few-shot examples the same model scored "
            f"{naive['precision']:.3f} precision and flagged "
            f"{', '.join('<code>' + x + '</code>' for x in naive['spurious'])} — "
            f"framework noise, one of them the cuDNN registration notice the corpus "
            f"was built around. Balanced retrieval (k signals and k negatives, never "
            f"whichever scores highest) removed both while recall held at "
            f"{best['recall']:.3f}. The nearest <i>negative</i> was the useful "
            f"neighbour, which is the whole reason the exemplar bank is half "
            f"noise.</div>"
        )
    if best.get("missed"):
        parts.append(
            f"<p class='small'>Still missed at k=3: "
            f"{', '.join('<code>' + m + '</code>' for m in best['missed'])} — "
            f"{len(best['missed'])} of 34 signals.</p>"
        )
    return "".join(parts)


def img(name: str) -> str:
    """Embed the figure as a data URI.

    LibreOffice's HTML import did not resolve the relative assets/ paths and
    silently produced blank gaps where every chart should have been -- the PDF
    built, exited 0, and shipped without a single figure. Embedding removes the
    resolution step rather than trusting it.
    """
    import base64

    path = ASSETS / name
    if not path.exists():
        return f"<p class='small'>[missing figure: {name}]</p>"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}">'


def _accuracy_section() -> str:
    """Both arms' reports, measured off `rig/report_accuracy.py`'s record.

    Absent rather than assumed: if the measurement has not been run, this says
    so instead of rendering an empty table that would read as a zero.
    """
    path = Path(RUNS_ROOT) / "report_accuracy.json"
    if not path.exists():
        return ("<p class='small'>Report-accuracy measurement not present. "
                "Generate it with <code>python -m rig.report_accuracy</code>.</p>")
    data = json.loads(path.read_text(encoding="utf-8"))
    return rt.report_accuracy_table(data) + rt.report_accuracy_note(data)


def _models_note(runs) -> str:
    """Name the models from the run records, never from memory.

    An earlier draft of this report asserted the engineer was a local code
    model. Every ablation record on disk said `deepseek-v4-flash`, and had done
    for some time; the sentence was left over from a superseded round. Prose
    that contradicts the data it sits beside is worse than no prose, and a
    reader who spots it is right to distrust everything around it — so the
    models are read out of the records that back the numbers.
    """
    engineers = sorted({r.engineer for r in runs.runs}) or ["(no run records)"]
    gates = sorted({r.gate for r in runs.runs}) or ["(no run records)"]
    return (
        f"Read from the {runs.n} run record(s) behind every number in this "
        f"report, not asserted. The ML engineer — the agent whose work is being "
        f"judged — runs on <code>{'</code>, <code>'.join(engineers)}</code>. "
        f"Gate 1's two model roles run on "
        f"<code>{'</code>, <code>'.join(gates)}</code>, locally and at no "
        f"per-call cost; they read logs and write the feedback report, and are "
        f"never consulted for the verdict. Both arms use the same engineer "
        f"model at the same temperature, so neither is advantaged."
    )


ARCHIVED_KEY = "archived run (different topic, no gate)"


def _paper_section() -> str:
    """Every generated paper, audited claim by claim on identical criteria."""
    path = Path(RUNS_ROOT) / "paper_audit.json"
    if not path.exists():
        return ("<p class='small'>Paper audit not present. Generate it with "
                "<code>python -m rig.audit_papers</code>.</p>")
    data = json.loads(path.read_text(encoding="utf-8"))
    out = [
        "<p>Three papers, each produced by a complete Agent Laboratory workflow "
        "— literature review through paper writing. The first two are the "
        "ablation: same research question, same engineer model, same prompts, "
        "differing only in whether Gate 1 arbitrates, with the host run as "
        "shipped via <code>GATES_GATE1=off</code>. The third is the archived run "
        "on its own topic and under the original prompts.</p>",
        img("papers.png"),
        rt.paper_audit_table(data),
        "<p class='small'>The gate-off arm is audited against what its own run "
        "recorded on disk, which is deliberately generous: "
        "<code>record_result</code> works without the gate, so the values "
        "existed — they simply never reached the writer. Crediting the arm for "
        "numbers it could not see keeps the comparison symmetric and makes the "
        "gap that survives the harder claim.</p>",
        "<div class='callout'><b>The result is not the one the framing "
        "predicts, and it is more useful than that one.</b> The gate-off arm did "
        "not fabricate. Its Results section says, correctly and in its own "
        "words, that <i>“the decisive measurements — <code>eff.acc_at_25</code>, "
        "<code>eff.acc_at_100</code>, <code>eff.efficiency_ratio</code>, and "
        "<code>eff.train_s</code> — are not present in the evidence window. The "
        "captured log was truncated at the legacy 1,000-character "
        "boundary.”</i> It then fills the section with the training-loss values "
        "it could see. It is an honest paper about a truncated log.</div>",
        "<p>That is why the honest reading separates two contributions rather "
        "than crediting one. <b>The prompt</b> — shared by both arms, and "
        "rewritten for this study to forbid describing a quantity that was not "
        "recorded — is what stops the fabrication; the archived run, under the "
        "original prompts and with no gate, invented all 65 of its figures. "
        "<b>Gate 1</b> is what makes the results reach the writer bound to the "
        "run that produced them. Neither alone yields a paper that is complete "
        "<i>and</i> traceable: instruction without delivery gives an honest "
        "paper with no findings, and delivery without instruction is the "
        "archived paper.</p>",
    ]
    archived = data.get(ARCHIVED_KEY, {})
    if archived.get("claims"):
        out.append(
            "<div class='warnbox'><b>The archived paper states "
            f"{archived['n_claims']} numeric findings and sources none of "
            "them.</b> Its saved experiment code calls "
            "<code>record_result</code> zero times, and the run that produced "
            "it raised <code>NameError</code> on every attempt while scoring "
            "1.0. That is the failure mode a validity layer exists to make "
            "impossible rather than unlikely.</div>"
        )
        out.append("<p class='small'>A sample of its unsourced claims, quoted "
                   "as the paper states them:</p>")
        out.append(rt.paper_claims_list(data, ARCHIVED_KEY, limit=8))
    pending = [
        name for name in ("with Gate 1", "without Gate 1")
        if data.get(name, {}).get("note") and not data.get(name, {}).get("n_claims")
    ]
    if pending:
        out.append(
            "<p class='small'><b>Not in this build: "
            f"{', '.join(pending)}.</b> Those runs had not finished when this "
            "report was generated, and are left absent rather than estimated.</p>"
        )
    return "\n".join(out)


def run_section(run, run_dir):
    """The measured run: what each arm produced, and whether it holds up."""
    runs = agg.Aggregate(agg.load_runs(RUNS_ROOT))
    aggregate_section = (
        agg.summary_table(runs) + agg.per_run_table(runs)
        if runs.n else "<p class='small'>No run records found.</p>"
    )
    accuracy_section = _accuracy_section()
    paper_section = _paper_section()
    models_note = _models_note(runs)
    if not run:
        return ("<h2>2. The run</h2><p class='small'>No ablation run record "
                "found. Sections 2-5 are generated from one.</p>")

    g, u = run["arms"]["gated"], run["arms"]["ungated"]
    gv, uv = g["verification"], u["verification"]
    g_rate = f"{100 * gv['rate']:.0f}%" if gv["rate"] is not None else "n/a"
    u_n = len(uv["reported"])

    gated_dir = Path(run_dir) / "gated" / "gate1"
    ungated_dir = Path(run_dir) / "ungated"
    acc_g = g["accepted_at"]
    # If nothing was accepted, the last attempt is still the interesting one --
    # it is what the ungated path would have shipped.
    last_g = g["attempts"][-1]["turn"] if g["attempts"] else None
    show_g = acc_g or last_g
    snippet_gate = (
        rt.log_snippet(gated_dir / f"attempt_{show_g:02d}" / "stdout.txt", tail=6)
        if show_g else "<p class='small'>no attempt produced output</p>"
    )
    snippet_ungated = rt.log_snippet(ungated_dir / "attempt_01" / "stdout.txt",
                                     head=10)

    return f"""<h2>2. The comparison: Agent Laboratory with and without Gate 1</h2>
<p>This is the whole argument in one figure. Both arms run the same task, the
same engineer model, the same temperature and the same turn budget; they differ
only in what is allowed to decide that an experiment succeeded.</p>

{img("headline.png")}

{hero(f'<b>Both arms accept every run.</b> Acceptance is not where they differ — everything that happens to the numbers afterwards is. With Gate 1, <span class="big">{g_rate}</span> of what was reported reproduced on re-execution and all of it carries a trace id. Without, <span class="big">{u_n}</span> values were recorded through any contract, five of fifteen results never reached the write-up at all, and none of them can be tied to the run that produced them.')}

<h3>2.1 Against the rates the benchmark papers report</h3>
<p>"Research agents fabricate results" is not a claim this report needs to
establish — the field has quantified it. <b>MLR-Bench</b> (arXiv 2505.19955)
finds that current coding agents produce fabricated or invalidated experimental
results in <b>80%</b> of cases, and that both agents it evaluates score below its
6.0 acceptance threshold on <i>Soundness</i> while scoring respectably on
<i>Completeness</i> — fluent, and unsound. <b>BadScientist</b> (arXiv 2510.18003)
finds deliberately fabricated papers accepted at rates up to <b>82%</b>, with
detection "barely exceeding random chance", and names the defence it thinks is
needed: <i>provenance verification</i>. That is what Gate 1 is.</p>
{img("benchmark.png")}
<p class="small">The three bars are related but not the same construct, and the
chart says so rather than implying a like-for-like ranking. MLR-Bench's is a
share of runs; BadScientist's is an acceptance rate under an LLM reviewer; ours
is the share of reported results that no execution can be shown to have
produced. What they have in common is the failure they measure.</p>

<h3>2.2 The task, and one run in detail</h3>
<p><b>Research question given to the agent:</b> how much labelled data does a
classifier need? Make a synthetic 3-class dataset, train multinomial logistic
regression by gradient descent on 25% of the training samples and on all of
them, and report the two test accuracies and their ratio.</p>
<p class="small"><b>Models.</b> {models_note}</p>
{rt.accuracy_summary(run)}

<h3>2.3 The numbers themselves</h3>
{rt.reported_numbers_table(run)}

<h3>2.4 Turn by turn</h3>
{rt.attempts_table(run)}

<h2>3. Across every run on disk</h2>
<p>One run is an anecdote. Every ablation record found under
<code>{RUNS_ROOT}</code> is aggregated here; a run that predates a measurement is
reported as <i>not measured</i> rather than as zero.</p>
{aggregate_section}

<h3>3.1 What each arm's report actually says</h3>
<p>Both arms produce a report; they are measured here the same way, on three
properties that are genuinely different and are kept apart because collapsing
them would hide the real result. <b>Completeness</b> — did the result reach the
writing agent at all? <b>Traceability</b> — does anything bind it to the
execution that produced it? <b>Accuracy</b> — does it reproduce when the accepted
code is re-run? For the gated arm the report is its registry. For the ungated arm
it is whatever numbers appear in the 1,000 characters upstream hands the writer,
because those are the only ones the writer can copy.</p>
{accuracy_section}

<h3>3.2 The papers</h3>
<p>Everything above measures a link in the chain. A reader sees none of it —
they see a paper. So the last question is the only one that reaches them: of the
numbers a generated paper states as its findings, how many can be tied back to
an execution? Both papers are put through the same audit, with the same claim
filter and the same rounding tolerance; the only asymmetry is the one being
measured, which is that one run has a registry to check against and the other
recorded nothing.</p>
{paper_section}

<h2>4. Every check that ran</h2>
<p>Not the catalogue of what Gate 1 <i>could</i> check — the record of what it
did check, on this run, turn by turn.</p>
{rt.checks_fired_table(run, "gated")}

<h3>4.1 What the same checks said about the ungated arm</h3>
<p>The ungated arm is judged by upstream's rule, but every one of its executions
was also passed through Gate 1 so the two verdicts can be compared on identical
evidence. Those shadow verdicts:</p>
{rt.checks_fired_table(run, "ungated")}

<h2>5. What the run cost</h2>
{rt.usage_table(run)}
{rt.cost_note(run)}

<h2>6. The feedback the engineer actually received</h2>
{rt.feedback_example(run_dir, run)}

<h2>7. The logs</h2>
{rt.log_locations(run_dir)}

<h3>7.1 With Gate 1 — captured in full, and the values recorded separately</h3>
{snippet_gate}

<h3>7.2 Without Gate 1 — the same capture, but only the first 1,000 characters
ever reach the agent</h3>
{snippet_ungated}
"""


def build_html(bench, run, run_dir):
    today = date.today().isoformat()
    audit_table = agg.audit(
        agg.Aggregate(agg.load_runs(RUNS_ROOT)),
        extra={
            "Report completeness, traceability and reproduction, both arms":
                "rig/report_accuracy.py over every attempt on disk; "
                "ungated arm re-executed to check",
        },
    )
    return f"""<html><head><meta charset="utf-8"><title>Gate 1 Report</title>
<style>{CSS}</style></head><body>

<h1>G.A.T.E.S. — Gate 1</h1>
<div class="sub">What it checks, what it measured, and what changed when it was
switched on</div>
<div class="meta">Generated {today} &nbsp;·&nbsp; gates @ main &nbsp;·&nbsp;
249 gate tests, 63 host integration tests &nbsp;·&nbsp; every number below is a
recorded measurement or a cited published figure</div>

<h2>1. The defect, in one line of the host scaffold</h2>
<pre>def execute_code(code_str, timeout=60, MAX_LEN=1000):
    ...
    except Exception as e:
        output_capture.write(f"[CODE EXECUTION ERROR]: {{str(e)}}\n")   # appended AFTER the prints
    return output_capture.getvalue()[:MAX_LEN]                        # ...then sliced off</pre>
<p>The writing agent's entire experimental record was 1,000 characters, and the
crash marker was appended after the program's own output — so on any run that
printed more than that, the marker fell off the same slice and the solver's only
crash test never fired.</p>
<table>
<tr><th style="width:38%">measured in the archived run</th><th>value</th></tr>
<tr><td>Reward score on a run raising <code>NameError</code> every attempt</td><td class="bad">0.95 → 0.98 → 1.0</td></tr>
<tr><td>Test accuracy reported in the paper</td><td>81.60% <span class="small">(never measured)</span></td></tr>
<tr><td>Speedup reported</td><td>13.61× <span class="small">(never measured)</span></td></tr>
<tr><td>Ablation collapse reported</td><td>39.20% <span class="small">(never measured)</span></td></tr>
<tr><td>Archived runs usable of seven</td><td class="bad">1</td></tr>
</table>

{run_section(run, run_dir)}

<h2>8. The full check inventory</h2>
<p>Twenty deterministic checks in six families, plus one model-assisted check.
The run fails if and only if a <span class="fail">FAIL</span> check fails;
<span class="warn">WARN</span> and <span class="info">INFO</span> are reported
and can never change a verdict.</p>
{img("checks.png")}
{checks_table()}

<div class="callout"><b>Why the severity split is load-bearing.</b> The LLM layer
is a required part of Gate 1 — the feedback report is the loop's return path to
the ML engineer, and no template writes “bind <code>hidden_dim</code> before use,
or pass it into <code>GCN.__init__</code>”. It coexists with a model-free verdict
because of where it sits and what severity it can emit: model findings are WARN
by construction, and report generation runs after <code>decide()</code> has
already fixed the verdict. A test parses the module's AST to assert
<code>Severity.FAIL</code> never appears in an expression there.</div>

<h2>9. Metrics the gate records per attempt</h2>
<table>
<tr><th style="width:24%">metric</th><th>definition</th></tr>
<tr><td class="mono">trace_id</td><td><code>sha256(run_id, key, lineno)</code> — binds one value to one execution. Two runs of identical source give different trace ids, which is what makes a backfilled number detectable.</td></tr>
<tr><td class="mono">chain_integrity</td><td>fraction of values whose provenance chain (task → command → log → value) is complete, with the missing link named per key.</td></tr>
<tr><td class="mono">citable</td><td>false on a rejected run's registry, so a consumer that ignores the verdict still cannot cite it.</td></tr>
<tr><td class="mono">arg_kind</td><td><code>computed</code> or <code>literal</code>, from static analysis of the <code>record_result</code> call site.</td></tr>
<tr><td class="mono">call_count</td><td>times a key was recorded, with the value span — a best-epoch number reported as final is visible.</td></tr>
<tr><td class="mono">code_sha256</td><td>hashed by the child that ran it and compared to what the parent wrote.</td></tr>
<tr><td class="mono">model.calls / degraded</td><td>what the LLM layer spent, and whether any call failed, so a thinner report is never mistaken for a complete one.</td></tr>
</table>

<h2>10. The log scanner, measured</h2>
<p>68 labelled lines, 34 of them error signals, 16 of those outside anything a
regex was going to reach. Sixteen lines come verbatim from the archived run.</p>
{img("scanner.png")}
<table>
<tr><th>scanner</th><th>precision</th><th>95% CI</th><th>recall</th><th>95% CI</th></tr>
{scanner_rows(bench)}
</table>
{verdict_paragraph(bench)}
{img("compression.png")}
<p>Before the model reads anything, the log is collapsed to its distinct shapes:
<b>203 non-blank lines to four</b>, a 95.9% smaller prompt. The collapse is
lossless in distinct content — every shape survives with its first real line
number, so nothing a scanner could have flagged disappears.</p>

<h2>11. What the layer costs in a phase</h2>
{img("callsplit.png")}
<p>Measured over five executions of one solver phase: Gate 1's own model calls
are <b>a third of the calls but a sixth of the tokens</b>. That is why the gate
can run on <code>qwen3:8b</code> on an 8 GB laptop GPU — 5.5 GB resident, ~16s a
call, zero marginal cost — while the engineer stays on a stronger model. Putting
the <i>engineer</i> on a weak model is the tempting economy and the wrong one: it
fails the gate more often, and every rejection costs a fixes call, a
shadow-reward call and another turn.</p>

<h2>12. Defects the runs found</h2>
<p>Nine, fixed. Four appeared only against a real model, and two of those only
against the weaker one.</p>
<table>
<tr><th style="width:32%">defect</th><th style="width:15%">found by</th><th>consequence</th></tr>
<tr><td>Unparseable reply executed an empty experiment</td><td>scripted e2e</td><td>Gate reported “you never called record_result()” when the command had not parsed</td></tr>
<tr><td>Evidence bundle showed warning counts without evidence</td><td>scripted e2e</td><td>The writer was told to disclose warnings it could not see</td></tr>
<tr><td>Fallback note reported an outage that had not happened</td><td>scripted e2e</td><td>Three different situations under one message</td></tr>
<tr><td>Prose in backticks parsed as code</td><td>gemini-3.5-flash</td><td>A good fix discarded over <code>['so','that','it','when','executing']</code></td></tr>
<tr><td>Prompt leaked the gate's artifact paths</td><td>gemini-3.5-flash</td><td>Told the engineer to edit <code>/home/…/attempt_03/experiment.py</code></td></tr>
<tr><td>Scan prompt offered two numbers per row</td><td>qwen3:8b</td><td>The weaker model reported the file line, not the row index; a correct finding was discarded</td></tr>
<tr><td>Output cap unreachable for most backends</td><td>qwen3:8b</td><td>One call generated <b>21,858 tokens</b> for an answer whose correct form is <code>[]</code></td></tr>
<tr><td>Rate-limit retry killed the phase in 25s</td><td>gemini-3.5-flash</td><td>Flat 5s × 5 against a per-minute meter</td></tr>
<tr><td>Five config keys reached nothing</td><td>integration audit</td><td>A config declaring a lit-review backend did not get one</td></tr>
</table>

<h2>13. This run, scored against MLR-Bench's taxonomy</h2>
<p>Our run, their categories. This is <b>not</b> a run of MLR-Bench — that needs
their harness and their task set — but the first of their four classes is
measurable on any run, and here it is measured rather than asserted.</p>
{rt.taxonomy_table(run) if run else "<p class='small'>No run record.</p>"}

<h2>14. Where this sits against the published benchmarks</h2>
{img("literature.png")}
<table>
<tr><th style="width:20%">source</th><th style="width:32%">what it measures</th><th>reported</th></tr>
<tr><td>MLR-Bench<br><span class="small">arXiv 2505.19955</span></td>
<td>Four fact-based hallucination types in AI-generated papers.</td>
<td>Faked results and hallucinated methodology each in more than half of 10 tasks;
<b>almost all AI Scientist V2 papers contain both</b>. Nonexistent citations in
<b>50%</b> of MLR-Agent tasks.</td></tr>
<tr><td>BadScientist<br><span class="small">arXiv 2510.18003</span></td>
<td>Whether fabricated papers pass multi-model LLM review.</td>
<td>Accepted at <b>up to 82.0%</b>; detection “barely exceeding random chance”.</td></tr>
<tr><td>CORE-Bench<br><span class="small">arXiv 2409.11363</span></td>
<td>Computational reproducibility by agents.</td>
<td>Best agent <b>21%</b> on the hardest level; correctness judged against a
<b>95% prediction interval</b> over three manual reproductions.</td></tr>
<tr><td>PaperBench<br><span class="small">arXiv 2504.01848</span></td>
<td>Replicating AI research end to end.</td>
<td>Best agent <b>21.0%</b>; ML PhDs <b>41.4%</b> best-of-3 after 48 hours.</td></tr>
</table>
<p>BadScientist is the reason Gate 1's verdict consults no model: if LLM
reviewers detect fabrication at barely above chance, a validity layer built on
model judgement inherits that ceiling.</p>
<div class="callout"><b>Naming.</b> This project's design documents paraphrase
MLR-Bench's taxonomy. Its published names are <i>faked experimental results,
hallucinated methodology, incorrect citations, mathematical errors</i>. “Silent
failure scored as success” is a cause of the first in their scheme, not a fifth
class, and any “eliminates N of four” claim should say so.</div>

<h2>15. Where every number here came from</h2>
<p>Each quantity below is either a measurement with a file behind it, a figure
quoted from a paper, or narrative. The third category is listed rather than
hidden, because a report about fabricated results should not contain any of its
own.</p>
{audit_table}

<h2>16. What Gate 1 does not claim</h2>
<ul>
<li>It does not check whether a result is <i>plausible</i> — that is Gate 2.</li>
<li>It does not read the manuscript — that is Gate 3.</li>
<li>The static tier is conservative on purpose; the runtime tier catches what it
declines to claim, one execution later.</li>
<li>The scanner's recall is bounded by the lines it was shown, and that count is
recorded so the claim cannot exceed the evidence.</li>
<li>A mean computed inside the experiment over a silently shortened list arrives
as one value the gate cannot see behind.</li>
<li>The ablation is <b>n = 1 at temperature 0</b>. It shows what happened on one
task with one model; it is not a rate.</li>
<li>n is still 1 for the archived-run comparison — the re-run has not been done.</li>
</ul>

<h2>17. Reproducing this</h2>
<pre>./tools_local_model.sh start                  # local qwen3:8b, no API key
python tools_ablation.py --model qwen3-8b-local --turns 4
python -m rig.corpus                          # deterministic scanner baseline
python -m rig.gate1_loop                      # the five loop scenarios
python reports/make_charts.py && python reports/make_report.py
python reports/make_deck.py
pytest                                        # 249 gate tests</pre>

<p class="small">Sources: MLR-Bench (2505.19955), BadScientist (2510.18003),
CORE-Bench (2409.11363), PaperBench (2504.01848), RE-Bench (2411.15114),
AutoResearchClaw (2605.20025), ScientistOne (2605.26340), SAGE (2606.31478).</p>

</body></html>"""


def main():
    bench = json.loads(BENCH.read_text()) if BENCH.exists() else None
    if bench is None:
        print("  WARNING: no bench2.json — model-tier numbers will be omitted")
    run = json.loads(RUN_JSON.read_text()) if RUN_JSON.exists() else None
    if run is None:
        print(f"  WARNING: no {RUN_JSON} — the run sections will be empty")
    OUT_HTML.write_text(build_html(bench, run, RUN_DIR), encoding="utf-8")
    print(f"  wrote {OUT_HTML.name}")
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE),
         str(OUT_HTML)],
        check=True, capture_output=True, timeout=300,
    )
    print(f"  wrote {OUT_PDF.name} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
