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

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT_HTML = HERE / "GATE1_REPORT.html"
OUT_PDF = HERE / "GATE1_REPORT.pdf"

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
"""

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


def build_html(bench):
    today = date.today().isoformat()
    return f"""<html><head><meta charset="utf-8"><title>Gate 1 Report</title>
<style>{CSS}</style></head><body>

<h1>G.A.T.E.S. — Gate 1</h1>
<div class="sub">Execution validity for autonomous research agents:
checks, metrics, runs, and measured impact on Agent Laboratory</div>
<div class="meta">Generated {today} &nbsp;·&nbsp; gates @ main &nbsp;·&nbsp;
247 gate tests, 63 host integration tests &nbsp;·&nbsp; every figure below is a
recorded measurement or a cited published number</div>

<h2>1. What Gate 1 is, and the defect it answers</h2>
<p>Gate 1 sits between code execution and the reward model in an autonomous
research scaffold, and answers one question with no model in the path:
<i>did this code actually run to completion, and were the numbers it reports
produced by this run rather than inherited, hardcoded, or invented?</i></p>

<p>In the audited scaffold (Agent Laboratory / Agent-Researcher) a single line
produced both a silent failure and a fabricated paper:</p>
<pre>def execute_code(code_str, timeout=60, MAX_LEN=1000):
    ...
    except Exception as e:
        output_capture.write(f"[CODE EXECUTION ERROR]: {{str(e)}}\\n")   # appended AFTER the prints
    return output_capture.getvalue()[:MAX_LEN]                        # ...then sliced off</pre>

<p>Two consequences follow from the same slice. The writing agent was starved —
<code>exp_results</code>, the entire experimental record handed to the paper
writer, was 1,000 characters. And the failure detector was blinded — the crash
marker is appended <i>after</i> the program's own output, so it fell off the end
of that same slice, and the solver's only crash test
(<code>if "[CODE EXECUTION ERROR]" in code_ret</code>) never fired.</p>

<div class="callout">Observed in <code>results/gemini_3_5_flash_run_1/</code>:
a run raising <code>NameError: name 'hidden_dim' is not defined</code> on every
attempt was scored <b>0.95 → 0.98 → 1.0</b> by the reward model and written up
with a results table to two decimals — 81.60% test accuracy, a 13.61× speedup,
a 39.20% ablation collapse. None of it was measured.</div>

<h2>2. The checks</h2>
<p>Twenty deterministic checks in six families, plus one model-assisted check.
Severity decides the verdict: the run fails if and only if a <span class="fail">FAIL</span>
check fails. <span class="warn">WARN</span> and <span class="info">INFO</span>
are reported and propagate into the evidence bundle, and can never block.</p>

{img("checks.png")}
{checks_table()}

<div class="callout"><b>Why the severity split is load-bearing.</b> The LLM layer
is a required component of Gate 1 — the feedback report is the loop's return path
to the ML engineer, and no template writes “bind <code>hidden_dim</code> before
use, or pass it into <code>GCN.__init__</code>”. It coexists with a model-free
verdict because of <i>where</i> it sits and <i>what severity</i> it can emit:
model findings are WARN by construction, and report generation runs after
<code>decide()</code> has already fixed the verdict. A test parses the module's
AST to assert <code>Severity.FAIL</code> never appears in an expression there.</div>

<h2>3. The metrics and artifacts Gate 1 produces</h2>
<p>Per attempt, under <code>&lt;research_dir&gt;/gate_artifacts/gate1/attempt_NN/</code>:</p>
<table>
<tr><th style="width:22%">artifact</th><th>contents</th></tr>
<tr><td class="mono">experiment.py</td><td>exactly what ran</td></tr>
<tr><td class="mono">stdout.txt / stderr.txt</td><td>complete, untruncated, separated</td></tr>
<tr><td class="mono">results.json</td><td>declared metrics with call-site provenance and the full observation history per key</td></tr>
<tr><td class="mono">registry.json</td><td>the citable value registry: typed values, trace ids, provenance chains, chain-integrity rate</td></tr>
<tr><td class="mono">gate1_report.json</td><td>verdict and every check, with evidence; model budget; generated fixes</td></tr>
<tr><td class="mono">divergence.jsonl</td><td>one line per attempt pairing the gate verdict with the reward score</td></tr>
</table>

<h3>Reported metrics</h3>
<table>
<tr><th style="width:26%">metric</th><th>definition</th></tr>
<tr><td class="mono">trace_id</td><td><code>sha256(run_id, key, lineno)</code> — binds one value to one execution. Two runs of identical source produce different trace ids, which is what makes a backfilled number detectable.</td></tr>
<tr><td class="mono">chain_integrity</td><td>fraction of recorded values whose provenance chain (task → command → log → value) is complete, with the missing link named per key.</td></tr>
<tr><td class="mono">citable</td><td>false on a rejected run's registry, so a consumer that ignores the verdict still cannot cite it.</td></tr>
<tr><td class="mono">arg_kind</td><td><code>computed</code> or <code>literal</code>, from static analysis of the <code>record_result</code> call site.</td></tr>
<tr><td class="mono">call_count</td><td>how many times a key was recorded, with the value span.</td></tr>
<tr><td class="mono">model.calls / degraded</td><td>what the LLM layer spent, and whether any call failed — so a thinner report is never mistaken for a complete one.</td></tr>
</table>

<h2>4. Logs: what the gate does with them</h2>
<p>The capture is written to disk in full and never truncated. Two things then
read it: a precision-tuned pattern set, and — for the lines those patterns cannot
reach — the model.</p>

{img("compression.png")}

<p>Before the model reads anything, the log is collapsed to its distinct shapes.
Experiment logs are overwhelmingly repetitive; measured on a realistic capture,
<b>203 non-blank lines reduced to four distinct shapes</b>. The collapse is
lossless in distinct content — every shape survives with its first real line
number, so nothing a scanner could have flagged disappears, and recall is not
traded for the saving. Two details do real work: <code>nan</code> and
<code>inf</code> are not digits, so <code>loss nan</code> survives as its own
shape rather than being absorbed into the epoch group; and when a group's values
drift, the last member is kept beside the first.</p>

<h3>Scanner accuracy on the labelled corpus</h3>
<p>68 labelled lines, 34 of them error signals, 16 of those outside anything a
regex was going to reach. Sixteen lines are taken verbatim from the archived run.
The corpus is self-checking: every entry naming a signal must be caught by that
signal, and every entry labelled a gap must not be caught at all.</p>

{img("scanner.png")}
<table>
<tr><th>scanner</th><th>precision</th><th>95% CI</th><th>recall</th><th>95% CI</th></tr>
{scanner_rows(bench)}
</table>
{verdict_paragraph(bench)}

<div class="callout">The four hard negatives that decide this are drawn from the
archived run itself: TensorFlow's <code>Unable to register cuFFT factory</code>
is logged at E level and mentions CUDA; oneDNN's banner contains the words
“numerical” and “errors”. Both appear on every healthy run that imports
TensorFlow.
<br><br><b>What a false positive here actually costs, stated precisely.</b> These
findings are WARN, and <code>decide()</code> fails only on blocking checks, so a
spurious one cannot force a rewrite. What it does is put a non-issue into the
“these must be stated in the report, not omitted” section the writer receives —
so the manuscript discloses a problem that never happened. That is a credibility
cost rather than a compute one, and this project's own earlier wording (“costs
the agent a rewrite”) overstated it. The correction has been made in the code
comments as well as here.</div>

<h2>5. The runs performed</h2>
<table>
<tr><th style="width:20%">run</th><th style="width:16%">model</th><th>result</th></tr>
<tr><td>Loop rig, 5 scenarios</td><td>none (scripted)</td>
<td>All five behave exactly as documented. Gate 1 rejects 6 engineer turns; the upstream 1,000-character detector would have accepted 3 of them.</td></tr>
<tr><td>End-to-end solver</td><td>scripted</td>
<td>Full gated path exercised — <code>process_command</code>, <code>Replace.parse_command</code>, <code>gated_execute</code>, the feedback loop, <code>get_score</code>, the ledger. Two defects found.</td></tr>
<tr><td>Live MLE-solver phase</td><td>gemini-3.5-flash</td>
<td>415s, 5 executions. Gate 1 caught a genuine <code>NameError</code> in model-written code; the engineer recovered on the next turn. Two defects found.</td></tr>
<tr><td>Live ablation</td><td>qwen3:8b (local)</td>
<td>658s. Gate on converged in 2 turns; gate off never converged in 3. Two defects found.</td></tr>
<tr><td>Corpus benchmark</td><td>qwen3:8b (local)</td>
<td>68 labelled lines &times; 2 prompt variants. Numbers in §4.</td></tr>
</table>

<h3>The divergence is conditional, and the live runs showed it</h3>
<div class="warnbox">In the live Gemini run the two rejected attempts were scored
<b>0.15 and 0.2</b> by the reward model — not 1.0. Both crashed <i>before printing
anything</i>, so the crash marker sat at index 0 of the 1,000-character view and
upstream's detector would have caught them too. The divergence the archived run
exhibits is real but requires an experiment that prints past the ceiling before it
dies. That is exactly what the channel-fidelity argument claims, and the honest
form of the claim is narrower than “the reward model scores crashed runs 1.0”.</div>

<h2>6. Impact on Agent Laboratory</h2>
<h3>6.1 What the writing agent receives</h3>
<table>
<tr><th style="width:50%">before</th><th>after</th></tr>
<tr><td>1,000 characters of stdout, prefix-sliced, with the crash marker
(if any) already fallen off the end.</td>
<td>A verified registry of typed values with trace ids and provenance chains,
run metadata, every WARN that must be stated rather than omitted, and the full
untruncated capture on disk with a generous inline budget.</td></tr>
<tr><td>No way to distinguish a measured number from an invented one.</td>
<td><code>results.values_computed</code> rejects any value that is a source
literal at its call site.</td></tr>
<tr><td>A crashed run reaching the writer scored 1.0.</td>
<td>A run that never passed cannot reach the writer at all —
<code>GateFailure</code> ends the phase.</td></tr>
</table>

<h3>6.2 The ablation: gate on versus gate off</h3>
<p>Same task, same model, same engineer; the arms differ in what decides that an
experiment succeeded, and in the width of the channel the engineer sees.</p>
{img("ablation.png")}
<pre>  ARM       TURNS  ACCEPTED  AT TURN  CRASH OK  NO RESULTS OK  FALSE SUCCESS
  gated         2      True        2         0              0              0
  ungated       3     False        —         0              0              0</pre>
<p><b>false_success was 0 in this run.</b> Every ungated failure crashed with zero
bytes of stdout, so the marker stayed inside the slice and upstream's rule caught
them all — the condition for divergence was not met. What the run does show is a
convergence difference: the gated engineer received a report naming
<code>static.syntax_valid</code> and fixed it; the ungated engineer received
1,000 characters of raw output and failed three times. The gated arm's first turn
also cost <i>zero compute</i> — the static tier rejected a syntax error before
execution.</p>
<div class="warnbox"><b>Read this at its true size.</b> n = 1, temperature 0, so
re-running reproduces it exactly rather than sampling. It is suggestive of a
convergence difference, not a rate. The ungated arm also does not receive the
results-contract instructions (faithful — upstream has none), which makes
“accepted without results” true of it by construction; the harness has a
<code>--same-instructions</code> mode that isolates the decision rule instead,
and any table should say which mode produced it.</div>

<h3>6.3 What the layer costs</h3>
{img("callsplit.png")}
<p>Measured over five executions of one solver phase: Gate 1's own model calls
are <b>a third of the calls but a sixth of the tokens</b>, because the digest
keeps its prompts small while the engineer's carry code, history and the evidence
bundle. That asymmetry is why the gate can run on a small local model —
<code>qwen3:8b</code> on an 8 GB laptop GPU, 5.5 GB resident, ~16s per call,
zero marginal cost — while the engineer stays on the stronger model.</p>
<p>Putting the <i>engineer</i> on a weak model is the tempting economy and the
wrong one: a weaker engineer fails Gate 1 more often, and every rejection costs a
fixes call, a shadow-reward call and another engineer turn.</p>

<h2>7. Validity metrics against the published benchmarks</h2>
{img("literature.png")}
<table>
<tr><th style="width:22%">source</th><th style="width:34%">what it measures</th><th>reported</th></tr>
<tr><td>MLR-Bench<br><span class="small">arXiv 2505.19955</span></td>
<td>Four fact-based hallucination types in AI-generated papers: faked
experimental results, hallucinated methodology, incorrect citations,
mathematical errors.</td>
<td>Faked results and hallucinated methodology each appear in more than half of
10 tasks; <b>almost all papers from AI Scientist V2 contain both</b>. Nonexistent
citations in <b>50%</b> of MLR-Agent tasks. Case study: the coding agent failed
to run the experiment and generated simulated results instead.</td></tr>
<tr><td>BadScientist<br><span class="small">arXiv 2510.18003</span></td>
<td>Whether fabricated papers requiring no real experiments can pass multi-model
LLM review.</td>
<td>Fabricated papers accepted at <b>up to 82.0%</b>. Detection accuracy
“barely exceeding random chance”. Reviewers flag integrity concerns yet assign
acceptance-level scores.</td></tr>
<tr><td>CORE-Bench<br><span class="small">arXiv 2409.11363</span></td>
<td>Computational reproducibility of published work by agents.</td>
<td>Best agent <b>21%</b> on the hardest level. Correctness judged against a
<b>95% prediction interval</b> over three manual reproductions — the tolerance
methodology Gate 2 adopts rather than choosing a band by hand.</td></tr>
<tr><td>PaperBench<br><span class="small">arXiv 2504.01848</span></td>
<td>Replicating AI research end to end.</td>
<td>Best agent <b>21.0%</b> average replication score; ML PhDs <b>41.4%</b> best-of-3
after 48 hours.</td></tr>
</table>

<h3>Where Gate 1 sits against these</h3>
<p>MLR-Bench's first class — faked experimental results, the most prevalent one —
is the class Gate 1 addresses, and it does so upstream of every reviewed
approach: before the reward model, before interpretation, before writing.
BadScientist's finding is the reason a checking <i>agent</i> was rejected as the
mechanism: if LLM reviewers detect fabrication at barely above chance, a validity
layer built on model judgement inherits that ceiling. Gate 1's verdict therefore
consults no model at all.</p>
<div class="callout"><b>The honest mapping.</b> This project's design documents
describe the MLR-Bench classes in paraphrase (“fabricated numeric results”,
“silent failure scored as success”). MLR-Bench's published names are <i>faked
experimental results, hallucinated methodology, incorrect citations, mathematical
errors</i>. Any claim of the form “eliminates N of the four MLR-Bench classes”
should use the published taxonomy or state the mapping explicitly — silent
failure scored as success is a <i>cause</i> of faked experimental results in
their scheme, not a fifth class.</div>

<h2>8. What running it actually found</h2>
<p>Nine defects were found and fixed during integration and live testing. Four
appeared only against a real model, and two of those only against the
<i>weaker</i> model — which is the argument for testing on both.</p>
<table>
<tr><th style="width:34%">defect</th><th style="width:16%">found by</th><th>why it mattered</th></tr>
<tr><td>Unparseable reply executed an empty experiment</td><td>scripted e2e</td>
<td>Gate 1 reported “you never called record_result()” when the real fault was that the command did not parse — the same misdiagnosis already fixed once on the timeout path.</td></tr>
<tr><td>Evidence bundle showed warning counts without the evidence</td><td>scripted e2e</td>
<td>The writer was told to disclose warnings it could not see — this layer's own failure mode, reproduced at its exit.</td></tr>
<tr><td>Fallback note reported an outage that had not happened</td><td>scripted e2e</td>
<td>Conflated “no model configured”, “model unreachable” and “model output rejected as ungrounded”.</td></tr>
<tr><td>Prose in backticks parsed as code</td><td>gemini-3.5-flash</td>
<td>A good fix discarded over <code>['so','that','it','when','executing']</code>.</td></tr>
<tr><td>Prompt leaked the gate's artifact paths</td><td>gemini-3.5-flash</td>
<td>Produced “define <code>n_classes</code> … in <code>/home/…/attempt_03/experiment.py</code>” — grounded, because the path is in the evidence, and useless.</td></tr>
<tr><td>Scan prompt offered two numbers per row</td><td>qwen3:8b</td>
<td>The weaker model reported the file line instead of the row index; grounding discarded it, losing a correct finding to prompt ambiguity.</td></tr>
<tr><td>Output cap was unreachable for most backends</td><td>qwen3:8b</td>
<td>Capped only when <code>api_key == "ollama"</code>. One call generated <b>21,858 tokens</b> — fourteen minutes for an answer whose correct form is <code>[]</code>.</td></tr>
<tr><td>Rate-limit retry killed the phase in 25s</td><td>gemini-3.5-flash</td>
<td>Flat 5s × 5 against a per-minute meter. Now exponential, bounded by a 15-minute per-process budget so an exhausted daily quota fails fast instead of stalling.</td></tr>
<tr><td>Five config keys reached nothing</td><td>integration audit</td>
<td>A config declaring a separate literature-review backend did not get one — the same defect class the layer exists to catch, one level up.</td></tr>
</table>

<h2>9. What Gate 1 does not claim</h2>
<ul>
<li><b>It does not check whether a result is plausible.</b> A correctly measured,
correctly recorded, causally traced number can still be scientifically wrong.
Comparing against the literature is Gate 2.</li>
<li><b>It does not read the manuscript.</b> Whether a prose claim is supported by
a registry value is Gate 3.</li>
<li><b>The static tier is conservative on purpose.</b> Module-level
use-before-assignment is not reported, because deciding it needs ordering
analysis and would produce false positives. The runtime tier catches what the
static tier declines to claim, one execution later.</li>
<li><b>The log scanner's recall is bounded by what it was shown</b>, and the
number of lines examined is recorded so the claim cannot exceed the evidence.</li>
<li><b>One requirement is partial.</b> Dropping a failed replicate from an
averaged metric is visible when each replicate is recorded, but a mean computed
inside the experiment over a silently shortened list arrives as one value that
Gate 1 cannot see behind. Closing it needs a replicate contract checked against
the plan's declared seed count, which is Gate 2's.</li>
<li><b>n is still 1 for the headline archive comparison.</b> Five of seven
archived runs died on a CLI defect (now fixed); the re-run has not been
performed, and no rate should be quoted from a single run.</li>
</ul>

<h2>10. Reproducing everything in this report</h2>
<pre>./tools_local_model.sh start                  # local qwen3:8b, no API key
python -m rig.gate1_loop                      # the five loop scenarios
python -m rig.corpus                          # deterministic scanner baseline
python tools_ablation.py                      # gate on vs gate off
python tools_live_gate1.py                    # one live solver phase
python reports/make_charts.py && python reports/make_report.py
pytest                                        # 247 gate tests</pre>

<p class="small">Sources: MLR-Bench (arXiv 2505.19955), BadScientist (arXiv
2510.18003), CORE-Bench (arXiv 2409.11363), PaperBench (arXiv 2504.01848),
RE-Bench (arXiv 2411.15114), AutoResearchClaw (arXiv 2605.20025), ScientistOne
(arXiv 2605.26340), SAGE (arXiv 2606.31478).</p>

</body></html>"""


def main():
    bench = json.loads(BENCH.read_text()) if BENCH.exists() else None
    if bench is None:
        print("  WARNING: no bench2.json — model-tier numbers will be omitted")
    OUT_HTML.write_text(build_html(bench), encoding="utf-8")
    print(f"  wrote {OUT_HTML.name}")
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE),
         str(OUT_HTML)],
        check=True, capture_output=True, timeout=300,
    )
    print(f"  wrote {OUT_PDF.name} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
