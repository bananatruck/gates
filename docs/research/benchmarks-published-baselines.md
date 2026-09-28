# Benchmarks with published baselines for the AAMAS 2027 draft

Status: research note, 28 Sep 2026, for the first draft due 8 Oct 2026 AoE.
Scope: which existing benchmarks let "Agent Lab + GATES" at levels 0-3 be compared, per level and overall, with published systems.
It extends `paper/PLAN.md` §3-§5, `docs/research/benchmark-fit.svg` and `docs/research/gate2-gate3-literature-readout.md` (§10); it does not repeat their content.

Conventions.
Every page number is the PDF page (the viewer's page index), not the printed folio.
Where the two differ it is said.
"Not found" means I looked in the named source and did not find it; it never means "zero".
Nothing here is a measurement of our system.
Numbers quoted from a figure's printed bar labels are marked "(figure label)".

---

## 0. Findings that change the plan

1. **ARC-Bench is the best second benchmark, and it is runnable on the laptop.**
   Its 25 ML topics are "CPU-executable in under 10 minutes on a single core" (AutoResearchClaw 2605.20025v2, App. D.1, p.17), every topic manifest declares hypotheses, conditions and metrics (Gate 2 input), the judge penalises ungrounded numbers (Gate 3 input), and the official repo already ships an Agent Laboratory adapter (`experiments/arc_bench/baseline/adapters/agent_lab_adapter.py`, https://github.com/aiming-lab/AutoResearchClaw).
   Agent Laboratory is the one system the paper excluded, because "it does not deliver end-to-end execution under fair-input conditions" (p.6), which the repo README attributes to runs that "die in lit-review SUMMARY-loop" (`experiments/arc_bench/baseline/README.md`).
   A gated Agent Lab that completes ARC-Bench topics is itself a result.
2. **MLReplicate (2605.16616) is the only benchmark with published numbers for Agent Laboratory and five of the other listed systems on the same tasks**, and its repo releases every system's generated papers, LaTeX and reviews, plus Agent Lab YAML task configs (https://github.com/gsasikiran/MLReplicate-benchmarking, Apache-2.0).
   It is the best re-scoring corpus we have for Gate 3 and for our judges.
3. **The local MLR-Bench kit does not contain AI Scientist v2's papers.**
   `run_kit_2026-09-26/mlrbench/baselines/ai_scientist_v2/<task>/` holds only four review JSONs per task (150 JSON files in `baselines/`, nothing else).
   The PDFs, `experiments/` code and `token_tracker.json` exist upstream at the same pinned commit `f728d57` under `ai_scientist_v2_papers/o4-mini/<task>/` (checked with the GitHub contents API).
   PLAN §7's "the same two judges re-score the ten released AI Scientist v2 papers" needs those files fetched first.
4. **MLR-Bench's published human-verified hallucination labels are per paper and public** (`human_eval/Hallucination Check.csv` at `f728d57`), and they show the Figure 6 procedure is not "flagged by both judges".
   See §1.6; PLAN §3's label definition is stricter than MLR-Bench's, so our rate and theirs are not the same quantity.
5. **CORE-Bench at 45 tasks x 4 levels does not fit in 10 days** (about 45 waves at 5.5-6 h each), it gives Gate 3 no input, and HAL has "declared CORE-Bench solved" (https://hal.cs.princeton.edu/corebench_hard).
   It should drop to a pilot or to future work (§4).
6. **No published number anywhere uses DeepSeek V4.1 Flash.**
   The closest is DeepSeek-V4-Pro at 17.1 on ResearchClawBench through a plain harness (§9).
   Every cross-system comparison in the draft is therefore a different-model comparison unless we re-score released artifacts with our own judges.

---

## 1. MLR-Bench (Chen et al., arXiv 2505.19955; local `Sources/MLR Bench.pdf`)

### 1.1 Task and output

| Item | Value | Source |
|---|---|---|
| Pool | 201 workshop tasks (NeurIPS, ICLR, ICML) | abstract / §1, p.2 |
| End-to-end subset | 10 ICLR 2025 workshop tasks | Table 8, p.21 |
| Given | a workshop task description (`tasks/<id>.md`) | kit `tasks/` |
| Produced | idea, proposal, experiment code and results, paper | §2, Table 2, p.5 |
| Their hardware | "Ubuntu 22.04 server with access to four NVIDIA RTX 3090 GPUs" | p.5 |

### 1.2 Metrics

Task score: MLR-Judge, 1-10 per dimension (Clarity, Novelty, Soundness, Significance, Overall), two judges averaged (Table 7 caption, p.7).
Integrity: four "fact-based hallucination types" (Faked Experimental Results, Hallucinated Methodology, Incorrect Citations, Mathematical Errors), found by two LLM judges and verified by human annotators (App. B, p.21-22).

### 1.3 Published numbers for research systems

| System, model | Metric | Value | Where | n |
|---|---|---|---|---|
| AI Scientist v2, o4-mini-high | Overall (two-judge mean) | 4.25 ± 1.25 | Table 7, p.7 | 10 tasks |
| AI Scientist v2 | Clarity / Novelty / Soundness / Significance | 6.55 / 6.70 / 3.70 / 4.85 | Table 7, p.7 | 10 |
| AI Scientist v2 | Overall, Gemini-2.5-Pro-Preview-05-06 judge | 5.00 ± 2.19 | Table 17, p.24 | 10 |
| AI Scientist v2 | Overall, Claude-3.7-Sonnet judge | 3.50 ± 1.20 | Table 18, p.24 | 10 |
| AI Scientist v2 | Faked experimental results | 100% (figure label) | Figure 6, p.22 | 10 |
| AI Scientist v2 | Hallucinated methodology | 90% (figure label) | Figure 6, p.22 | 10 |
| AI Scientist v2 | Incorrect citations | 30% (figure label) | Figure 6, p.22 | 10 |
| AI Scientist v2 | Mathematical errors | 10% (figure label) | Figure 6, p.22 | 10 |
| MLR-Agent, o4-mini-high + Codex | Overall | 3.10 ± 0.60 | Table 7, p.7 | 10 |
| MLR-Agent, Gemini-2.5-Pro-Preview + Gemini CLI | Overall | 4.60 ± 1.00 | Table 7, p.7 | 10 |
| MLR-Agent, Claude-3.7-Sonnet + Claude Code | Overall | 4.70 ± 1.22 | Table 7, p.7 | 10 |
| MLR-Agent | Faked results / methodology / citations / math | 80% / 60% / 50% / 0% (figure labels) | Figure 6, p.22 | 10 |
| Claude Code (experiment stage only) | Overall | 4.95 ± 0.82 | Table 5, p.6 | 10 |
| Claude Code | tasks whose results were "synthesized or placeholder data" | 8 of 10 | §5, p.8 | 10 |
| Agent Laboratory | any | not reported (only cited, ref. [26]) | whole paper | - |

Cost: MLR-Agent with Claude Code costs $1.15 (o4-mini-high), $1.24 (Gemini) and $2.40 (Claude) (Table 7 caption, p.7).
AI Scientist v2's own `token_tracker.json` files upstream sum to $18.84 over the 10 tasks, mean $1.88 per task with `o4-mini-2025-04-16` (my sum of the ten files at `f728d57`).

### 1.4 Runnability

Code and data: MIT, pinned in `run_kit_2026-09-26/mlrbench/` (tasks, judge, review JSONs; see §0 item 3 for what is missing).
Compute: the tasks are open-ended, so GPU need depends on the idea; MLR-Bench used four RTX 3090s (p.5).
Closed models: MLR-Judge used Gemini and Claude; PLAN D63 substitutes `deepseek-v4-pro` and Nemotron Ultra `:free`.
Harness: already built (`tools_levels.py`, `rig/judge.py`, PLAN §5 phase 1).
Our cost: one wave per task, 5.5-6 h and about $3-4, so 10 tasks x 1 seed is about 10 waves.

### 1.5 Gate input

All three gates have input: executed code (Gate 1), a plan whose ranges and relations Gate 2 checks, a paper (Gate 3).
Per-level effect: yes, this is the primary benchmark.

### 1.6 Comparability and re-scoring

Same tasks: yes (the 10 in Table 8).
Same judges: no; the two published judges disagree in rank order (Table 17 vs 18, p.24), which PLAN D63 already notes.
Same model: no (o4-mini-high vs DeepSeek V4.1 Flash).

The label procedure differs from PLAN §3.
The released `human_eval/Hallucination Check.csv` (19 rows: 10 AI Scientist v2, 9 MLR Agent; `iclr2025_scope` has no MLR Agent row) gives AI Scientist v2: faked results 10/10, methodology 9/10, citations 3/10 (question, scope, verifai), math 1/10 (mldpr), which reproduces the Figure 6 labels.
In the kit's judge JSONs for AI Scientist v2, the Claude judge flags nonexistent citations on 8/10 papers and the Gemini judge on 4/10, and only the Gemini judge flags a mathematical error (mldpr).
So Figure 6 counts a type a person confirmed from either judge's evidence (App. B, p.22: annotators "are provided with two review files" and verify the evidence), not the intersection of two judges.
My inference from those counts: a both-judges rule would have produced at most 4/10 citations and 0/10 math for AI Scientist v2.
The kit's review JSONs reproduce Tables 17-18 exactly (Claude overall mean 3.5, Gemini 5.0, my recomputation).

Re-scoring we can do: fetch the 10 AI Scientist v2 PDFs and `experiments/` from upstream and score them with our two judges and our human-confirmation rule; do the same for `agent_results/end2end_*` (MLR-Agent).
That puts AI Scientist v2, MLR-Agent and Agent Lab + GATES under one judge, on the same tasks, with only the backbone model differing.

---

## 2. ARC-Bench (AutoResearchClaw, arXiv 2605.20025v2; local `Sources/Relative Results/2605.20025v2.pdf`)

### 2.1 Task and output

| Item | Value | Source |
|---|---|---|
| Core ML topics | 25 (T01-T25, renamed ML01-ML25 in the repo) | §4.1, p.6; Table 9, p.18 |
| Science extension | 20 in the paper (10 HEP, 7 biology, 3 statistics); the repo README says 55 topics incl. 10 quantum | p.6; repo README |
| Given | a manifest: research question, synthesis, conditions, metrics, datasets, hypotheses | App. D.1, p.17; `config/ml/manifests/ML01.yaml` |
| Produced | code, machine-readable metrics, claims, writeup (experiment-stage mode); full paper (end-to-end mode) | §4.1, p.6 |
| Compute | "CPU-executable in under 10 minutes on a single core using standard numpy/scipy/sklearn primitives" | App. D.1, p.17 |

### 2.2 Metrics

Experiment-stage strict score in [0,1], weighted Code Development : Code Execution : Result Analysis = 25:25:50 (§4.1, p.6).
Integrity is built into the rubric: "Number grounding: every numerical claim in the writeup must trace to a captured artefact; fabricated numbers penalise the relevant leaf to 0.1-0.3", and "Verdict-data consistency" for inverted hypothesis verdicts (App. D.2, p.19).
End-to-end mode: paper quality 1-10, accept if ≥ 5 (Table 3 caption, p.7).
Table 5 (p.9) adds a manual "Fabrication" audit column.

### 2.3 Published numbers

| System, model | Metric | Value | Where | n |
|---|---|---|---|---|
| AI Scientist v2, GPT-5.3-codex | Overall strict | 0.419 (CD 0.712, CE 0.442, RA 0.261) | Table 2, p.7 | 25 topics |
| AIDE-ML, GPT-5.3-codex | Overall strict | 0.511 | Table 2, p.7 | 25 |
| AutoResearchClaw Full-Auto, GPT-5.3-codex | Overall strict | 0.596 | Table 2, p.7 | 25 |
| AutoResearchClaw CoPilot (HITL) | Overall strict | 0.648 | Table 2, p.7 | 25 |
| AI Scientist v2 | topics with no valid results | 6 of 25 | §4.2, p.7 | 25 |
| AI Scientist v2 | science-domain overall | 0.084 (statistics 0.418; biology and HEP failed) | Table 4, p.8 | 20 |
| AutoResearchClaw Full-Auto, best-of-3 | completion / quality / accept | 10/10, 5.62, 3/10 | Table 5, p.9 | 10 |
| AutoResearchClaw w/o verification | accept, fabrication | 5/10, fabrication ✓ | Table 5, p.9 | 10 |
| Agent Laboratory | any | excluded | §4.1, p.6 | - |

SAGE (arXiv 2606.31478v1) reruns a frozen 12-topic ARC-Bench subset under its own blind rubric (Claude Opus 4.8 judge, 2:2:3 weighting, 0-100): AI-Scientist-v2 48.2, SAGE 52.0, and 24.8 for a row labelled "AutoResearchClaw" that the text defines as the host reflection baseline "SAGE w/o MHFA" (Table 1 and §4.1, p.8).
That 24.8 is not AutoResearchClaw as released and should not be cited as such.
SAGE's backbone model is described only as "the same backbone model" for all systems (p.8); its name is not found.
SAGE reports "a complete end-to-end SAGE run costs approximately $10 to $20" (p.8).

### 2.4 Runnability

Inputs (manifests, rubrics, runners, judge prompts) are public: repo MIT, Hugging Face dataset `AIMING-Lab-UNC/ARC-Bench` with license `mit` (HF API).
Run outputs and scoring write-ups are not public ("Large run outputs (`results/`) and scoring write-ups (`analysis/`) are kept local-only", repo README).
Judge: the paper's strict judge was run by "a Claude Code subagent (Opus 4.7), a Codex CLI agent (GPT-5.4), and a human expert" (App. D.2, p.19); `scripts/judge.py` takes any OpenAI-compatible model via `ARC_JUDGE_MODEL`, and `scripts/judge_results_only.py` is a no-LLM programmatic check.
Harness: the Agent Lab adapter renders a per-topic YAML from the manifest and maps `research_dir_*` outputs to ARC-Bench's `submission/` layout; its defaults are `mlesolver_max_steps=3`, `papersolver_max_steps=1`, `num_papers_lit_review=1` and it adds "framework-fix" task notes for the lit-review loop.
We would swap the upstream Agent Lab clone for our gated host and keep the adapter; I estimate one to two days, not verified.
Per-topic cost and wallclock for us: not derivable from the paper; the experiment itself is capped at 10 CPU minutes, so a wave is likely shorter than an MLR-Bench wave, but this is unmeasured.

### 2.5 Gate input

Gate 1: yes, the metrics artefact must come from executed code.
Gate 2: the strongest of any benchmark here, since each manifest states measurable hypotheses (for example ML01 H2: accuracy differs by "<2 absolute percentage points" between variants), which map directly onto declared relations.
Gate 3: yes, the writeup's numbers and verdicts are what Result Analysis grades.
Per-level effect: yes.

### 2.6 Comparability and re-scoring

Same tasks: yes if we run the same topics; but only the 25-topic mean is published, and per-topic scores are not released, so a subset run cannot be set against a same-subset baseline.
Same judge: we can run the released strict-audit prompt (`scripts/prompts/manual_strict_audit_prompt.md`) with our judges, which is the same rubric but a different judge.
Same model: no (GPT-5.3-codex).
Re-scoring of published baselines: impossible, since their outputs are not released.
Honest framing: our L0-L3 deltas are measured; the published 0.419 / 0.511 / 0.596 are context, and the headline is whether gated Agent Lab completes topics that the ARC-Bench authors could not get Agent Lab to finish.

---

## 3. BadScientist (arXiv 2510.18003v2; local `Sources/Bad Scientist.pdf`)

### 3.1 Task and output

A paper agent (GPT-5, adapted from AI-Scientist with experiment execution removed) writes fabricated papers from 25 seed topics, 4 papers per seed per strategy, under strategies s1 TooGoodGains, s2 BaselineSelect, s3 StatTheater, s4 CoherencePolish, s5 ProofGap and All (§3.2, p.3; §4.1, p.6; App. B, p.15).
Review agents o3, o4-mini and GPT-4.1 score them against thresholds calibrated on ICLR 2025 (τrate = 7, τ0.5 = 6.667, p.6).

### 3.2 Metrics

Acceptance rate ACPT at τrate and τ0.5; Integrity Concern Rate ICR per reviewer and any-of-panel ICR@M, judged by GPT-5 (p.6).

### 3.3 Published numbers

| Setting | Metric | Value | Where |
|---|---|---|---|
| s1 TooGoodGains | ACPT τrate / τ0.5 | 67.0% / 82.0% | Table 1, p.7 |
| All | ACPT τrate / τ0.5 | 52.0% / 69.0% | Table 1, p.7 |
| All | ICR@M | 51.7% | Table 1, p.7 |
| o4-mini, s1 / s3 / All | concern-acceptance conflict | 100.0% each | Table 2, p.7 |
| ReD mitigation | ACPT τ0.5, baseline to ReD | 37.0% to 58.0% | Table 3, p.8 |

No numbers for any research system on our list: the only generator is their own GPT-5 paper agent.

### 3.4 Runnability

Code is released (https://github.com/Bad-Scientist/BadScientist, MIT, contains `perform_review.py` and a prompt registry), although the paper says "We will partially release our artifact due to ethical concerns" (p.10).
Reviewers are OpenAI models; we would substitute our non-model-under-test reviewer, as PLAN §5 phase 4 does.
Harness: PLAN phase 4 reuses the MLR-Bench runs and changes only the writer, so marginal cost is writer and reviewer calls only.

### 3.5 Gate input and comparability

Gate 3 carries it; Gates 1-2 see the real runs underneath (PLAN §4 figure 3).
The published 69.0% is for papers with no experiments at all, so it is context, not a baseline (PLAN already says this).
PLAN phase 4 uses s1, s2, s3 and All; s4 and s5 are omitted, which is a choice to state in the paper.

---

## 4. CORE-Bench (arXiv 2409.11363v2; local `Sources/2409.11363v2.pdf`)

### 4.1 Task, metric, numbers

270 tasks from 90 CodeOcean papers, 45 train / 45 test papers, three levels; Hard gives only the README (Table 3, p.6).
Metric: pass@1 accuracy on task questions; no integrity or fabrication metric is defined (not found).

| System, model | Hard, test | Where |
|---|---|---|
| CORE-Agent, GPT-4o | 21.48% (mean of 3 runs); 21.48 ± 2.60 (95% CI) | Table 5, p.10; Table A3, p.20 |
| CORE-Agent, GPT-4o-mini | 16.30% | Table 5, p.10 |
| AutoGPT, GPT-4o | 6.67% | Table 5, p.10 |
| HAL leaderboard, Claude Code + Claude Opus 4.5 | 77.78% (95.5% with manual validation), $87.16 | hal.cs.princeton.edu/corebench_hard |
| Any research system on our list | not found | - |

PLAN §4 and `results.csv` cite Table 5 as p.9; it is on PDF page 10 (printed folio 10); p.9 ends just before it.
Table 5 prints Easy as 60.00% while Table A3 prints 60.60% ± 4.51% for the same row (p.10 vs p.20); only the Hard value matters for us.

### 4.2 Runnability, gates, verdict

Repo MIT (`siegelz/core-bench`).
Their harness ran each task on an Azure VM, Standard_NC4as_T4_v3 for GPU capsules (p.19), with a 2-hour per-task limit (p.9) and a $4 cost cap (p.10).
Harness for us: PLAN phase 5 calls it "the largest piece of new work".
Gate input: Gate 1 strongly, Gate 2 weakly, Gate 3 none (PLAN §3).
Budget: 45 test tasks x 4 levels is about 45 waves, more than the whole 10-day window at about 4 waves a day.
Comparability: CORE-Agent used GPT-4o; HAL's top entries use frontier Claude; neither is a research system.

---

## 5. MLReplicate (arXiv 2605.16616v1; local `Sources/MLReplicate.pdf`)

### 5.1 Task and output

8 ICML 2025 outstanding papers turned into standardized specs (hypotheses, experimental design, metrics, key findings), 6 systems, 48 targeted submissions, 45 generated manuscripts, 37 reviewed (Table 2, p.3; §1, p.3; §3, p.5).
Two of the 8 are position papers with no experiments (P7, P8; Table 2, p.3).

### 5.2 Metrics

Automated review by the ICAIS 2025 pipeline (DeepReviewer, the ZGCA review system, SafeReviewer) with an accept decision; human review of the 10 accepted papers on six 4-point dimensions, a 6-point overall rating, and "a binary hallucination judgment" (§3, p.5-6).

### 5.3 Published numbers

| System, model (Table 3, p.4) | Metric | Value | Where |
|---|---|---|---|
| All systems | accepted by automated review | 10 of 37 | §3, p.5 |
| Agent Laboratory, o3-mini | accepted | 5 of the 10 | §3, p.5 |
| AI Scientist v2, gpt-4o-mini/gpt-4o/o3-mini | accepted | 0 | §3, p.5 |
| Agent Laboratory | human-flagged fabricated or unsupported claims | 59% of evaluations | §3, p.7 |
| AI-Researcher | same | 100% | §3, p.7 |
| TinyScientist | same | 67% | §3, p.7 |
| CycleResearcher-12B | same | 33% | §3, p.7 |
| All accepted papers | same | 59% | abstract, p.1 |
| Agent Laboratory | tokens in / out, cost, runtime | 626K / 42.3K, $0.92, 1,105 s | §3, p.7 |
| CycleResearcher | cost, runtime | $0.46, 243 s, 23 GB GPU on one H100 | §3, p.7 |
| AI-Researcher | median cost, runtime | $17.72, 2,252 | §3, p.7 |
| AI Scientist v1 | per-paper cost, runtime | about $15.00, about 28,800 s | §3, p.7 |
| Human vs automated overall rating | Pearson r | +0.29 (not significant) | §1, p.3; §3, p.7 |

Per-system counts in Figure 3 (p.5) are bar labels I could not read reliably from the extracted text; not quoted.
The denominators behind 59% for Agent Laboratory are not stated (5 papers x 3 reviewers = 15 evaluations would not give 59%); not resolved.

### 5.4 Runnability

Repo Apache-2.0 with, for Agent Laboratory: 8 task YAMLs in Agent Lab's own config format (`AgentLaboratory/dataset/*.yaml`), 7 generated PDFs, LaTeX and review PDFs (`icais-conference/agent-laboratory/`), and the 5 accepted PDFs (`icais-conference/accepted-papers/agent-laboratory/`).
The released YAML sets `copilot-mode: True`, `llm-backend: "o3-mini"`, `mlesolver-max-steps: 2`, `papersolver-max-steps: 2`; whether any human input was given in copilot mode is not stated in the file.
Per-paper human hallucination labels: not found in the repo.
Running it: the specs include masked-diffusion training (P3) and CollabLLM (P5), which are unlikely to fit an 8 GB GPU; P2 (conformal prediction) and P4 look CPU-scale from their text but I have not checked.
The ICAIS reviewers are not all reproducible (ZGCA is a hosted system; not verified further).

### 5.5 Gate input and comparability

A run would give all three gates input on the experimental specs; the specs' "key findings" are published reference values, which is Gate 2 tier B material.
Re-scoring: yes, this is the main value.
We can run Gate 3's `source.identifiers_resolve` and our MLR-Judge prompts over the released papers of Agent Laboratory, AI Scientist v2, AI-Researcher, CycleResearcher and TinyScientist, which gives a same-judge baseline for "Agent Laboratory as released" that no other benchmark provides.

---

## 6. ADRS as used by ScientistOne's CoE Integrity Audit (arXiv 2605.26340v1; local `Sources/Relative Results/2605.26340v1.pdf`)

### 6.1 Task and output

Five systems-optimization tasks (Prism, Cloudcast, EPLB, LLM-SQL, TXN), each with a fixed evaluator, starter code and scoring metric (§6, p.7).
Every system ran 3 seeds per task, giving 15 papers per system and 75 in all, with Gemini 3.1 Pro as backbone, up to 20 solver iterations and 2-hour code-generation windows (p.8).

### 6.2 Metrics

Solver score per task (Table 3, p.12), ScholarPeer review (Table 2, p.11), and the CoE Integrity Audit: I1 score verification, I2 specification violation, I3 reference verification, I4 method-code alignment (Table 1, p.8).
"All I1-I3 flagged results were manually verified by human reviewers. I4 judgments were validated on a sampled basis." (p.8).

### 6.3 Published numbers

Table 1 (p.8) is already transcribed in `gate2-gate3-literature-readout.md` §0 and is not repeated.
New here:

| System | ScholarPeer Overall (1-10), mean of 15 | #Accept | Where |
|---|---|---|---|
| Sakana AI-Scientist v2 | 2.5 | 0/15 | Table 2, p.11 |
| AutoResearchClaw | 1.9 | 0/15 | Table 2, p.11 |
| DeepScientist | 2.5 | 1/15 | Table 2, p.11 |
| AI-Researcher | 3.4 | 2/15 | Table 2, p.11 |
| ScientistOne | 4.5 | 6/15 | Table 2, p.11 |

Solver scores, best-of-3 (Table 3, p.12), for example TXN (higher is better): Human 2724.8, Sakana 4184, ARC 3247, AIR 4311, DS 4286, ScientistOne 3906.
The paper says "the BFTS-ADRS design mismatch confounds both I2 and I4 for Sakana, cross-system comparison on these two checks should exclude Sakana" (p.8).
Agent Laboratory: not evaluated.

### 6.4 Runnability

ADRS tasks are public (https://github.com/UCB-ADRS/ADRS, Apache-2.0; `openevolve/examples/ADRS/<task>/` with `initial_program.py` and `evaluator.py`).
They are simulators: TXN, LLM-SQL, Cloudcast and Prism need no GPU from their file lists; EPLB needs PyTorch and a workload file from Hugging Face (task README).
ScientistOne's baseline adapters are not found in its release; its artifact repo (https://github.com/scientist-one/generated-artifacts, Apache-2.0) holds only ScientistOne's own 21 papers and solver code (15 ADRS, 5 MLE-Bench, 1 Parameter Golf).
Harness for Agent Lab: point mle-solver at `initial_program.py` and the evaluator; ScientistOne needed 16-19 patched source files for Sakana and AIR (p.8), so expect days, not hours.

### 6.5 Gate input and comparability

Gate 1: I1 is literally "does the claimed score reproduce under the canonical evaluator", which is Gate 1's contract.
Gate 2: I2 specification violations are declared-constraint checks.
Gate 3: I3 and the paper's numbers.
Comparability: the audit's definitions are deterministic for I1-I3, so applying them to our runs gives a like-for-like integrity metric, but on Gemini 3.1 Pro vs DeepSeek V4.1 Flash.
Re-scoring: only ScientistOne's 21 papers are released, so Gate 3's citation audit can cover ScientistOne but not the four baselines.

---

## 7. Hidden Pitfalls of AI Scientist Systems (Luo, Kasirzadeh, Shah, arXiv 2509.08713v2)

### 7.1 Task, metrics, numbers

A synthetic Symbolic Pattern Reasoning (SPR) task used to probe four failure modes: benchmark selection, data leakage, metric misuse, post-hoc selection bias (abstract; §5).
Systems: Agent Laboratory with "default LLM API configurations" and AI Scientist v2 with its code model replaced by O3-mini (p.9); Agent Lab's model is not named (not found).

| Pitfall | Agent Laboratory | AI Scientist v2 | Where |
|---|---|---|---|
| Benchmark selection | first-4 of listed benchmarks: 82.4% (779/945) with SOTA refs, 79.6% (738/927) without | easy benchmarks: 47.1% with SOTA refs, 18.0% without | Tables 3-4, p.10 |
| Metric misuse | SWA-first: 100% report SWA only; CWA-first color-flip: 70% both | frequently neither metric, e.g. SWA-first shape-flip 20% SWA only, 50% both | Tables 7-8, p.13 |
| Post-hoc selection | best-label pick 78.5% control vs 43.5% manipulated | 82.0% vs 31.5%; worst label 49.0% manipulated | Tables 9-10, p.14 |
| Data leakage | undisclosed synthetic data (run 11) and subsampling (run 16) | frequent subsampling or synthesis | Tables 5-6, p.11-12 |

Detection by an LLM auditor (gemini-2.5-flash-preview-05-20): overall accuracy 55.0% from the paper alone vs 82.0% with paper, logs and code (Table 11, p.18).

### 7.2 Runnability, gates, comparability

Repo https://github.com/niharshah/AIScientistPitfalls ships the SPR task, datasets and both systems' generated research; the GitHub API reports no top-level license (subfolders carry upstream licenses), so reuse terms are unclear.
CPU-only synthetic data; the Agent Lab configuration used is in the repo, so harness work is small.
Gate input: metric misuse is a Gate 2 declared-field check (the plan names the metric, the result must report it); undisclosed data synthesis is a Gate 1 provenance question; post-hoc selection is methodology Gate 2 does not currently check.
Comparability: same host (Agent Laboratory), different and unnamed model.
Table 11 is the published case for our design choice that gates read logs and code, not the paper alone.

---

## 8. AstaBench E2E-Bench (arXiv 2510.21652, ICLR 2026)

40 test and 10 validation problems for each of E2E-Bench and E2E-Bench-Hard (table on p.6); the input is an AI/NLP research question with a detailed step description, and the output is a report, trace, code and artifacts, scored by an LLM judge against a per-problem rubric (App. E, p.33-34).
Published numbers (Table 10, p.20): Asta Panda (claude-sonnet-4) 70.5 ± 6.2, Asta CodeScientist (claude-3-7-sonnet) 65.3 ± 7.1, ReAct (claude-sonnet-4) 52.5 ± 6.8.
**Faker (gpt-4.1), "a baseline agent used to validate the scoring metrics", which "simply prompts a LM to make up the report, code, and artifacts", scores 39.2 ± 6.9 on E2E-Bench and 25.4 ± 4.5 on E2E-Bench-Hard** (Table 10, p.20; definition p.45).
Full-completion rates are near zero for every agent, Faker 0.00 (Table 20, p.41).
AI Scientist and Agent Laboratory are named as target agents (p.33) but no numbers for them are found.
Repo Apache-2.0 (`allenai/asta-bench`); running Agent Lab needs an adapter to its inspect-based harness, not estimated.
All three gates would have input; published baselines are not systems on our list, so this is future work, and the Faker row is a citation for "rubric judges reward fabrication".

---

## 9. ResearchClawBench (arXiv 2606.07591v5)

40 tasks from 10 scientific domains, each grounded in a hidden published paper with raw data; expert rubrics; GPT-5.1 scores the final report (§4.1, p.10).
Published (Table 5, p.10): Claude Code (Claude-Opus-4.6) 21.5, EvoScientist v0.1.1 (GPT-5.4) 18.8, ResearchClaw (GPT-5.4) 16.3, and **DeepSeek-V4-Pro via ResearchHarness 17.1**.
"ResearchClaw" here is `ymx10086/ResearchClaw` (reference list, p.18), not AutoResearchClaw.
No integrity metric separate from the rubric is found; error analysis names "experimental protocol mismatch, evidence mismatch, and missing scientific core" (abstract).
Repo and data MIT; tasks span astronomy to energy-system modelling with domain data, so not runnable in the window.
Published-only context; it is the only published number on a DeepSeek V4 model in an end-to-end research setting that I found.

---

## 10. Other benchmarks checked, with Agent Lab or AI Scientist numbers where they exist

| Benchmark | What it measures | Published number for a listed system | Gate input | Runnable by 8 Oct |
|---|---|---|---|---|
| Agent Lab's own MLE-Bench subset (arXiv 2501.04227v2) | 10 low-complexity text/tabular MLE-Bench tasks | mle-solver: 4 medals (2 gold, 1 silver, 1 bronze), above median on 6/10; AIDE (o1-preview) 5/10, OpenHands (gpt-4o) 2/10, MLAB 0/10 (§4.4 and Figure 9, p.18-19); mle-solver's backbone is not named in that passage | Gate 1, weak Gate 2, no Gate 3 | No |
| Agent Lab's own human review (2501.04227v2) | NeurIPS-style ratings of its papers | overall 3.5/10 (gpt-4o), 3.8/10 (o1-mini), 4.0/10 (o1-preview) (§4.1.1, p.12; Figure 6) | - | published-only |
| MLE-bench (openai/mle-bench) | 75 Kaggle competitions; Lite 22, 158 GB; 36 vCPU, 440 GB RAM, one 24 GB A10, 24 h (repo README) | ScientistOne and DeepScientist on 5 tasks (ScientistOne Table 4, p.13) | Gate 1 | No (data size, 24 GB GPU) |
| AutoReproduce's REPRODUCE-Bench (arXiv 2505.20662v4) | 13 papers, code reproduction, o1 judge | Agent Laboratory (GPT-4o): Mixed-Level Align 48.64, Exec Rate 23.08%, Perf Gap 82.31 (Table 2, p.6) | Gate 1; Gate 2 via the reference-performance gap | No (repo-reproduction harness) |
| DeepReviewer-14B scores (Jr. AI Scientist, TMLR 02/2026) | reviewer rating of public papers | AI Scientist-v1 3.30 (n=10), AI Researcher 3.25 (n=7), AI Scientist-v2 2.75 (n=3), CycleResearcher-12B 3.92 (n=6), Zochi 4.50 (n=2), Jr. AI Scientist 5.75 (n=3) (Table 2a, p.12) | re-scoring instrument only | Unverified: they used "a single A100 80G GPU" (p.10) |
| PaperBench (2504.01848v3) | 20 ICML 2024 papers, 8,316 leaf nodes, A10 GPU, 12 h runs | none for listed systems; best BasicAgent Claude 3.5 Sonnet 21.0% (Table 4, p.7) | Gate 1 | No ($66 per paper to grade with o3-mini, p.6) |
| RE-Bench (2411.15114v2) | 7 environments, up to 6 H100s (p.8) | none for listed systems | Gate 1 | No |
| EXP-Bench (2505.24785v2) | 461 tasks from 51 papers; complete executable experiments 0.5% (abstract) | none for listed systems (Agent Lab only cited) | Gates 1-2 | No (not assessed further) |
| ScienceAgentBench (2410.05080v3) | 102 program-output tasks (abstract) | none for listed systems | Gate 1 | Not assessed |
| LMR-Bench (2506.17335v1) | 28 masked-function reproduction tasks (abstract) | none for listed systems | Gate 1 | Not assessed |
| ResearchBench (2503.21248v3) | inspiration retrieval, hypothesis composition and ranking (abstract) | none; no experiments | none | Not applicable |
| SPOT | already in the readout §7 | - | WARN tier only | - |

AI-Researcher's Scientist-Bench, CycleResearcher's and DeepScientist's own evaluations were not read; their numbers are not in this note.

---

## 11. Ranked recommendation

Budget assumption: about 7 compute days before writing, at about 4 waves a day, so about 28 four-level waves and roughly $85-115 of DeepSeek API at $3-4 a wave (from the brief's per-wave figures, not measured here).

| Rank | Benchmark | Per-level effect | Published baselines to compare | Effort to run by 8 Oct | Verdict |
|---|---|---|---|---|---|
| 1 | MLR-Bench, 10 tasks | Yes, all three gates | AI Scientist v2 and MLR-Agent: scores (Tables 7, 17, 18) and hallucination rates (Figure 6 plus released per-paper CSV) | Low: harness built; about 10 waves per seed; fetch AI Scientist v2 PDFs upstream | **Run now**, plus re-judge AI Scientist v2 and MLR-Agent papers with our judges |
| 2 | ARC-Bench, ML subset (e.g. ML01-ML10) | Yes, all three; best Gate 2 input | AI Scientist v2 0.419, AIDE-ML 0.511, AutoResearchClaw 0.596 / 0.648 (Table 2), 25-topic means only; SAGE's AI-Scientist-v2 48.2 on 12 topics | Medium: adapter exists; swap in gated host; about 10 waves; judge with released prompt | **Run subset** |
| 3 | BadScientist | Yes, Gate 3 carries it | reviewer acceptance of fabricated papers (Table 1), no system baselines | Low: reuses MLR-Bench runs | **Run now** (as PLAN phase 4) |
| 4 | Post-hoc audit of released papers (PLAN figure 7) | Gate 3's run-independent check only | MLR-Bench AI Scientist v2 30% citations (Figure 6); ScientistOne I3 (Table 1); MLReplicate's six systems; ScientistOne's own 21 papers; Pitfalls outputs | Low: `rig/posthoc_audit.py` built; collect PDFs | **Run now** |
| 5 | MLReplicate | Yes if run; re-scoring otherwise | Agent Laboratory (o3-mini) 5/10 accepted, 59% fabricated or unsupported; AIR, TS, CR, AIS v1/v2 | Re-scoring: low. Running: unknown GPU fit per spec | **Published-only comparison plus re-scoring**; run 1-2 CPU-scale specs only if waves remain |
| 6 | ADRS with CoE audit definitions | Yes, all three | Sakana v2, ARC, DS, AIR, ScientistOne (Tables 1-3), Gemini 3.1 Pro | Medium-high: new harness | **Future work** (stretch if ARC-Bench harness finishes early) |
| 7 | CORE-Bench Hard | Gates 1-2 only | CORE-Agent GPT-4o 21.48%; HAL frontier entries | High: new harness; 45 waves for the full split | **Future work**, or a 5-10 task pilot only if the harness exists by 2 Oct |
| 8 | Hidden Pitfalls SPR | Gates 1-2 on specific failure modes | Agent Laboratory and AI Scientist v2 (Tables 3-10) | Low-medium, CPU; license unclear | **Future work** (cite Tables 7 and 11 now) |
| 9 | AstaBench E2E-Bench | Yes, all three | Asta agents, ReAct, Faker; none on our list | High | **Future work**; cite the Faker 39.2 |
| 10 | ResearchClawBench | Yes, in principle | DeepSeek-V4-Pro 17.1; ResearchClaw, EvoScientist | High: domain data and tools | **Published-only context** |
| 11 | MLE-bench, PaperBench, RE-Bench, EXP-Bench, AutoReproduce, LMR-Bench, ScienceAgentBench | Gate 1 mostly; no manuscript | Agent Lab on MLE-Bench subset and REPRODUCE-Bench only | High or hardware-blocked | **Future work** |

The minimum credible first draft is ranks 1-4: two benchmarks with all three gates and per-level runs (MLR-Bench, ARC-Bench), one adversarial Gate 3 benchmark (BadScientist), and one audit of other systems' released papers.
Replacing CORE-Bench with ARC-Bench is the main change this note proposes to PLAN §3 and §5.

---

## 12. Comparison matrix sketch

Cells: a published number with its source, "ours" (to be measured by us), "re-score" (published artifacts we can score with our judges or Gate 3), or "n.r." (not reported).
Models differ across every published cell; the model is named in the section cited.

| System | MLR-Bench (10 tasks) | ARC-Bench | BadScientist | CORE-Bench Hard | ADRS / CoE audit | MLReplicate | Pitfalls SPR |
|---|---|---|---|---|---|---|---|
| Agent Lab + GATES L0 | ours | ours | ours | future | future | future / optional | future |
| Agent Lab + GATES L1 | ours | ours | ours | future | future | future / optional | future |
| Agent Lab + GATES L2 | ours | ours | ours | future | future | future / optional | future |
| Agent Lab + GATES L3 | ours | ours | ours | future | future | future / optional | future |
| Agent Laboratory as published | n.r. | excluded (§4.1, p.6) | n.r. | n.r. | n.r. | 5/10 accepted, 59% fabricated or unsupported (p.5, p.7); re-score | Tables 3, 5, 7, 9 (p.10-14) |
| AI Scientist v2 | Overall 4.25; faked results 100% (Table 7 p.7; Fig. 6 p.22); re-score | 0.419 (Table 2 p.7); 48.2 in SAGE (Table 1 p.8) | n.r. | n.r. | I1 5/12, I2 10/15, I3 0/159, I4 5/15 (Table 1 p.8); ScholarPeer 2.5 (Table 2 p.11) | 0 accepted (p.5); re-score | Tables 4, 6, 8, 10 (p.10-14) |
| AI Scientist v1 | n.r. | n.r. | n.r. | n.r. | n.r. | about $15/paper (p.7); re-score | n.r. |
| MLR-Agent | Overall 3.10-4.70 (Table 7 p.7); Fig. 6 p.22; re-score | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. |
| ScientistOne | n.r. | n.r. | n.r. | n.r. | I1 12/12, I3 0/337, I4 14/15 (Table 1 p.8); ScholarPeer 4.5 (Table 2 p.11); own papers re-score | n.r. | n.r. |
| AutoResearchClaw | n.r. | 0.596 Full-Auto (Table 2 p.7) | n.r. | n.r. | I1 5/12, I4 3/15 (Table 1 p.8) | n.r. | n.r. |
| AI-Researcher | n.r. | n.r. | n.r. | n.r. | I1 9/12, I3 21/222 (Table 1 p.8) | 100% fabricated or unsupported (p.7); re-score | n.r. |
| DeepScientist | n.r. | n.r. | n.r. | n.r. | I1 11/12, I3 42/201 (Table 1 p.8) | n.r. | n.r. |
| CycleResearcher-12B | n.r. | n.r. | n.r. | n.r. | n.r. | 33% (p.7); re-score | n.r. |
| TinyScientist | n.r. | n.r. | n.r. | n.r. | n.r. | 67% (p.7); re-score | n.r. |
| Jr. AI Scientist | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. |
| SAGE | n.r. | 52.0 on 12 topics, own judge (Table 1 p.8) | n.r. | n.r. | n.r. | n.r. | n.r. |
| AIDE-ML | n.r. | 0.511 (Table 2 p.7) | n.r. | n.r. | n.r. | n.r. | n.r. |
| CORE-Agent (reference, not a research system) | n.r. | n.r. | n.r. | 21.48% GPT-4o (Table 5 p.10) | n.r. | n.r. | n.r. |

Jr. AI Scientist appears only in its own DeepReviewer table (§10), which is not one of these benchmarks.

---

## 13. Open questions and claims I could not verify

1. Whether our Agent Lab host at default settings completes ARC-Bench topics, and how long a wave takes there; ARC-Bench's authors could not make Agent Lab finish, and our wallclock on CPU-only topics is unmeasured.
2. Which coding agent produced MLR-Bench Tables 15-16.
   Their o4-mini-high rows equal the MLR-Agent rows of Tables 17-18, but their two-judge mean (3.95) does not equal Table 7's "o4-mini-high + Codex" 3.10, and the Gemini rows do not average to Table 7's 4.60, while the Claude rows do average to 4.70.
   My guess, not stated in the paper, is that Tables 15-16 use Claude Code as the coding agent for all three models.
3. Which MLR-Agent configuration Figure 6's MLR-Agent bars and the released CSV's "MLR Agent" rows describe; not stated.
4. Why `iclr2025_scope` has no MLR Agent row in the released hallucination CSV while Figure 6 reports percentages over 10 tasks.
5. The denominator behind MLReplicate's 59% for Agent Laboratory, and whether `copilot-mode: True` in its released configs meant a human intervened in Agent Lab's runs.
6. SAGE's backbone model (not named) and ARC-Bench's per-topic scores (not released), so neither can be matched to a subset.
7. Whether DeepReviewer-14B runs on an RTX 4060 8 GB with quantization; not tested.
8. MLE-bench's license: the GitHub API reports `NOASSERTION`; I did not read the LICENSE file.
9. Whether the MLReplicate specs P2, P4 or P6 run on CPU or 8 GB GPU; judged from titles only.
10. Agent Lab's MLE-Bench backbone model and the 10 task names (Agent Lab 2501.04227v2 §4.4 does not list them in the text I read).
11. The AIScientistPitfalls repo has no top-level license file per the GitHub API, so reuse of its generated research may need permission.
12. Whether BadScientist's released code includes the manipulated-paper generator for every strategy, given "partially release" (p.10); I listed files, not their contents.
13. The benchmark-fit rubric in `benchmark-fit.svg` scores ARC-Bench's Gate 3 coverage as 1 of 2; given the App. D.2 number-grounding and verdict-data criteria, 2 may be fairer, which is a judgment for the author.

---

## Provenance

| Source | Read from | Parts used |
|---|---|---|
| MLR-Bench 2505.19955 | `Sources/MLR Bench.pdf`; kit at `f728d57`; upstream repo at `f728d57` via GitHub API | Tables 5-8, 15-18; Fig. 6; App. B; `human_eval/Hallucination Check.csv`; `token_tracker.json` x10; kit review JSONs |
| AutoResearchClaw 2605.20025v2 | `Sources/Relative Results/`; repo `aiming-lab/AutoResearchClaw` | Tables 1-5, 9; App. D; `experiments/arc_bench/` README, baseline README, Agent Lab adapter, ML01 manifest and rubric, judge scripts |
| ScientistOne 2605.26340v1 | `Sources/Relative Results/`; project page; `scientist-one/generated-artifacts` | Tables 1-5; §6; App. G |
| SAGE 2606.31478v1 | `Sources/Relative Results/` | Table 1; §4.1 |
| BadScientist 2510.18003v2 | `Sources/Bad Scientist.pdf`; repo `Bad-Scientist/BadScientist` | Tables 1-3; §3-4; App. B |
| CORE-Bench 2409.11363v2 | `Sources/2409.11363v2.pdf`; HAL leaderboard | Tables 3, 5, A3; App. B |
| MLReplicate 2605.16616v1 | `Sources/MLReplicate.pdf`; repo `gsasikiran/MLReplicate-benchmarking` | Tables 2-3; §3; repo tree, one YAML, `human_evaluation_map.json` |
| Jr. AI Scientist (TMLR 02/2026) | `Sources/Jr.AI Scientist.pdf` | Tables 1-2 |
| PaperBench 2504.01848v3, RE-Bench 2411.15114v2 | `Sources/Benchmarks/` | compute and headline numbers only |
| Agent Laboratory 2501.04227v2 | arXiv PDF | §4.1.1, §4.4, Figure 9 caption |
| Hidden Pitfalls 2509.08713v2 | arXiv PDF; repo README | Tables 3-11 |
| AutoReproduce 2505.20662v4 | arXiv PDF | Table 2 |
| AstaBench 2510.21652 | arXiv PDF | Tables 10, 20; E2E and Faker definitions |
| ResearchClawBench 2606.07591v5 | arXiv PDF; HF dataset API | Table 5; §4.1 |
| ADRS | repo `UCB-ADRS/ADRS` | README, task folder listings |
| MLE-bench | repo README | split sizes, hardware |
| EXP-Bench, ScienceAgentBench, LMR-Bench, ResearchBench | arXiv abstracts only | as stated |

Text was extracted with `pdftotext -layout` into the session scratchpad; no file in the repository other than this one was written.
