# Benchmarks with published results for AI Scientist v2, ScientistOne or Agent Laboratory, with task lists

Status: research note, 27 Sep 2026, for picking comparison tasks before returning to MLR-Bench.
It extends `docs/research/benchmarks-published-baselines.md` (called "the baselines note" below) and does not repeat its analysis.
Where this note touches a claim of the baselines note, it says whether the claim held.

Conventions, same as the baselines note.
Every page number is the PDF page (viewer index), not the printed folio.
"Not verified" means I tried and could not confirm; what I tried is stated.
"Not found" means I looked in the named source and did not find it; it never means zero.
Nothing here is a measurement of our system.
Repo commits are the ones I read on 27 Sep 2026 via the GitHub API or raw.githubusercontent.com.
Systems are abbreviated AIS2 (The AI Scientist v2, Sakana), S1 (ScientistOne, Google Cloud AI Research) and AL (Agent Laboratory, Schmidgall et al.).

---

## 1. Summary table

Marks: a number means a published result for that system on that benchmark; "-" means not evaluated there; "excl." means named and deliberately excluded.
"Scores existing outputs" means the benchmark's scorer can be applied to already-produced papers, code or logs without re-running the agent.

| # | Benchmark (source) | AIS2 | S1 | AL | Tasks | CPU / GPU | Scores existing outputs |
|---|---|---|---|---|---|---|---|
| 1 | MLR-Bench end-to-end (2505.19955v3) | Overall 4.25 ± 1.25 (Table 7, p.7) | - | - | 10 of 201 | GPU used (4x RTX 3090, p.21); per task not stated | yes; AIS2 papers, code, reviews released |
| 2 | ARC-Bench experiment stage (2605.20025v2) | 0.419 on 25 ML topics (Table 2, p.7); 0.084 on 20 science topics (Table 4, p.8) | - | excl. (p.6) | 25 ML + 20 science in paper; 55 in repo | CPU only (all 55 manifests `gpu_required: false`) | yes (judge reads a submission dir); baseline outputs not released |
| 3 | SAGE's frozen ARC-Bench subset (2606.31478v1) | 48.2 (Table 1, p.8); human 4.72/10 on 6 ML topics (Table 4, p.12) | - | - | 12 | CPU (same manifests) | yes (artifact-level rubric); outputs not verified |
| 4 | MLReplicate (2605.16616v1) | 0 accepted (p.5) | - | 5 of 10 accepted; 59% fabricated or unsupported (p.5, p.7) | 8 papers | per spec; AIS2 ran on 2x L40S (p.17) | yes; PDFs, LaTeX, reviews released |
| 5 | ADRS with CoE Integrity Audit (S1 paper, 2605.26340v1) | I1 5/12, I2 10/15, I3 0/159, I4 5/15 (Table 1, p.8); ScholarPeer 2.5, 0/15 (Table 2, p.11); solver scores (Table 3, p.12) | I1 12/12, I2 0/15, I3 0/337, I4 14/15; ScholarPeer 4.5, 6/15; solver scores | - | 5 | CPU simulators; EPLB needs PyTorch | yes for I1 and I3 (evaluator re-run, reference check); only S1 outputs released |
| 6 | MLE-bench, S1's 5 Medium/High tasks (S1 paper) | - | per task (Table 4, p.13) | - | 5 | S1 used 8x H100 (p.33) | yes (mle-bench grader on a CSV); S1 code and papers released |
| 7 | Parameter Golf (S1 paper) | - | 1.0600 BPB (Table 4, p.13) | - | 1 | 8x H100, 10 min (p.33) | no (live competition rules) |
| 8 | MLE-bench Lite, AL's 10 tasks (2501.04227v2) | - | - | mle-solver per task (Figure 9, p.18) | 10 | small data (0.002-5.7 GB); AL's GPU not stated | yes (grader); AL submissions not released |
| 9 | AgentRxiv MATH-500 (2503.18102v1) | - | - | 70.2% to 78.2% sequential, 79.8% parallel (p.7-8, p.11) | 1 (+3 transfer sets) | API calls only | partly (re-run the discovered prompt) |
| 10 | Agent Lab's own human study (2501.04227v2) | - | - | ratings per backend (p.10-11) | 5 questions | not stated | human study, not re-runnable |
| 11 | AIS2's ICBINB workshop test (2504.08066v1) | 1 of 3 accepted, 6.33 (p.9, Table 4 p.31) | - | - | 3 papers | NVIDIA GPU required (repo README) | human review, not re-runnable |
| 12 | Hidden Pitfalls SPR (2509.08713v2) | Tables 4, 6, 8, 10 (p.10-14) | - | Tables 3, 5, 7, 9 (p.10-14) | 20 SPR datasets x 4 pitfall tests | small synthetic data; GPU not stated | yes (auditor reads paper, code, logs); outputs released |
| 13 | REPRODUCE-Bench (AutoReproduce 2505.20662v4) | - | - | Align 48.64, Exec 23.08%, Gap 82.31 (Table 2, p.6) | 13 papers | A100 used (p.12) | partly (needs generated code) |
| 14 | ARAC-Bench (2608.12788v1) | 61.68 (Table 3, p.9) | - | 55.65 (Table 3, p.9) | 200 papers | not stated | no (stage-isolated inputs; no scripts released) |
| 15 | FML-bench v2 (2605.17373v2) | 0.193 mean normalized improvement, strategy only (Table 2, p.7) | - | - | 18 | A100-80GB (p.6) | no (needs its loop) |
| 16 | Re-scoring of public papers: DeepReviewer-14B (2506.01372v2; Jr. AI Scientist 2511.04583) | 2.33 (Table 2, p.7) and 2.75 (Table 2a, p.12), same 3 papers | - | - | 3 AIS2 papers | one 14B model | yes |
| 17 | Re-scoring of public papers: ScholarPeer and Stanford Agentic Reviewer (ScientistTwo 2609.19644v1) | 2.0 / 0.0% and 2.5 / 0.0% (Table 2, p.10) | 3.8 / 14.3% and 4.1 / 0.0% (Table 2, p.10) | - | 3 AIS2, 21 S1 papers | hosted reviewers | yes |
| 18 | BadScientist (2510.18003v2) | - | - | - | 25 seed topics | API only | yes (reviewer panel) |

No benchmark I found reports all three systems.
ARAC-Bench (row 14) is the only one that reports AIS2 and AL on the same tasks with the same backbone (Kimi-K2.6).
MLReplicate (row 4) and Hidden Pitfalls (row 12) report AIS2 and AL on the same tasks with different backbones.
The ADRS audit (row 5) and ScientistTwo's re-scoring (row 17) are the only places S1 and AIS2 appear together.

Searched and found no numbers for any of the three systems: AstaBench E2E (names AIS and AL as targets, p.33, per the baselines note), MLRC-Bench (2504.09702v3, cites both only in references, p.15-16), InnovatorBench (2510.27598, no mention), AI-Researcher's Scientist-Bench (2505.18705v1, AL cited in related work, p.19), DeepScientist (2509.26603v1, AL cited, p.3), ResearchArena (2605.19156v1, related work only, p.2), AARRI-Bench (2606.07462v1, references only), AutoLab (2606.05080v1), InquiTree (2606.09550v1), ASI-Bench (2608.17271v1), SciIntegrity-Bench (2605.10246v2, reference only), and FML-bench v1 (2510.10472v2, which runs AI Scientist v1, "Lu et al., 2024", p.3).

Surveys checked, 2025-2026: "AI for Auto-Research: Roadmap & User Guide" (2605.18661v2) Table 11, p.52, restates each system's self-reported evaluation (AL "$2-13/paper; 3.5-4.0 NeurIPS scale", AIS2 "ICLR 2025 ICBINB workshop; score 6.33", S1 "Chain-of-evidence verifiability against fabrication") and adds no benchmark numbers.
"Autonomous Research Agents: A Survey of AI Scientists and the Verification Gap" (2608.05179v1) Table 3, p.11, and "Workflow Closure Is Not Scientific Closure" (2605.26200v1) Table on p.5 are qualitative feature and autonomy tables only.

---

## 2. Benchmarks, one section each

### 2.1 MLR-Bench

| Item | Value |
|---|---|
| Paper | Chen et al., arXiv 2505.19955v3, NeurIPS 2025 Datasets and Benchmarks track (arXiv comment) |
| Repo | https://github.com/chchenhui/mlrbench at `f728d57` (MIT); local kit `run_kit_2026-09-26/mlrbench/` copies tasks, judge and review JSONs from that commit |
| Systems reported | AIS2 only (with o4-mini-high); AL and S1 not evaluated |

Published AIS2 numbers are in the baselines note §1.3 and I re-checked Table 7 (p.7): Clarity 6.55, Novelty 6.70, Soundness 3.70, Significance 4.85, Overall 4.25 ± 1.25.
Per-task AIS2 overall scores are published only as the released review JSONs; I extracted them from the kit and they reproduce Tables 17-18 (Claude mean 3.5, Gemini mean 5.0):

| Task ID (repo `tasks/<id>.md`) | Topic (Table 8, p.21) | Category | AIS2 Overall, Claude judge | AIS2 Overall, Gemini judge | Human-verified hallucination types (`human_eval/Hallucination Check.csv`) |
|---|---|---|---|---|---|
| iclr2025_bi_align | Bidirectional Human-AI Alignment | Trustworthy AI | 4 | 7 | results, methodology |
| iclr2025_buildingtrust | Building Trust in Language Models and Applications | Trustworthy AI | 2 | 3 | results, methodology |
| iclr2025_data_problems | Navigating and Addressing Data Problems for Foundation Models | Trustworthy AI | 2 | 3 | results, methodology |
| iclr2025_dl4c | Emergent Possibilities and Challenges in Deep Learning for Code | LLM/VLM | 4 | 2 | results, methodology |
| iclr2025_mldpr | The Future of Machine Learning Data Practices and Repositories | Trustworthy AI | 3 | 5 | results, methodology, math |
| iclr2025_question | Quantify Uncertainty and Hallucination in Foundation Models | LLM/VLM | 6 | 8 | results, methodology, citations |
| iclr2025_scope | Scalable Optimization for Efficient and Adaptive Foundation Models | Trustworthy AI | 4 | 7 | results, methodology, citations |
| iclr2025_scsl | Spurious Correlation and Shortcut Learning | Trustworthy AI | 4 | 7 | results, methodology |
| iclr2025_verifai | VerifAI: AI Verification in the Wild | Trustworthy AI | 2 | 2 | results, methodology, citations |
| iclr2025_wsl | Neural Network Weights as a New Data Modality | ML Theory | 4 | 6 | results |

Full pool: 201 tasks in `tasks/` at `f728d57`, one Markdown file per workshop.
By venue-year from the file names: iclr2023 14, iclr2024 17, iclr2025 37, icml2023 18, icml2024 24, neurips2023 41, neurips2024 50.
The paper groups them into 9 topics (LLMs/VLMs, AI for Science, ML Theory, Trustworthy AI, Computer Vision, ML Systems, Multimodality, RL, Others; p.3); the per-topic counts are bar labels in Figure 2 (p.4) that I could not map reliably from the extracted text.

Compute: "four NVIDIA RTX 3090 GPUs" for all experiments (p.21); per-task need is not stated, and tasks are open-ended so the need depends on the idea.
Scoring: MLR-Judge, 1-10 per dimension, Gemini-2.5-Pro-Preview and Claude-3.7-Sonnet averaged; hallucination types found by the two judges and human-verified (App. B, p.21-22).
Scores existing outputs: yes; the judge takes a paper plus experiment logs.
Released outputs: AIS2 PDF, `experiments/`, four review JSONs and `token_tracker.json` per task under `ai_scientist_v2_papers/o4-mini/<task>/`; MLR-Agent outputs under `agent_results/end2end_*` (checked at `f728d57`).
The baselines note's claims in §1 that I touched (Table 7 values, the per-task hallucination CSV, the upstream folder layout, Tables 17-18 reproduction) all held.

### 2.2 ARC-Bench

| Item | Value |
|---|---|
| Paper | AutoResearchClaw, arXiv 2605.20025v2 (venue not stated on arXiv) |
| Repo | https://github.com/aiming-lab/AutoResearchClaw at `be4ba47` (2026-08-19), `experiments/arc_bench/`; HF dataset `AIMING-Lab-UNC/ARC-Bench` at `5a5923b`, licence `mit` |
| Systems reported | AIS2 (GPT-5.3-codex); AL excluded (p.6); S1 not evaluated |

AIS2 numbers (re-checked): experiment-stage overall strict 0.419 with CD 0.712, CE 0.442, RA 0.261 (Table 2, p.7); failed to produce valid results on 6 of 25 topics (p.7); science domains 0.084 overall, statistics 0.418, biology and HEP failed (Table 4, p.8).
Which 6 topics AIS2 failed is not listed; the text says they are "concentrated in topics requiring iterative experiment refinement (dynamical systems, causal discovery)" (p.7).
Per-topic scores are not published.

**Correction to the baselines note.**
The baselines note says the 25 topics are "T01-T25, renamed ML01-ML25 in the repo".
For 19 of 25 IDs the paper topic and the repo manifest agree; for 6 they are different topics.
The paper's T-list is Table 9 (p.18); the repo list is `experiments/arc_bench/config/ml/manifests/ML*.yaml` and `config/ml/topics.yaml`.
The paper also says (App. E, p.20) that T01-T10 "span tabular ML, RL, MoE, NLP, physics-informed ML, and finance", which matches neither Table 9 nor the repo; the ARC-Bench core in the repo is all sklearn-scale ML.
SAGE (§2.3) uses the repo numbering (its ML20 is seasonal forecasting, its ML16 is bandits).
Which list produced Table 2's 0.419 is not stated; not verified.

ML topics, repo IDs (all `gpu_required: false`; ML01 declares `estimated_wall_clock_sec: 240`):

| Repo ID | Repo title (manifest `title:`) | Paper Table 9 topic for the same number |
|---|---|---|
| ML01 | Dropout regularization strategies on shallow tabular MLPs | same (T01 Dropout) |
| ML02 | Bagging vs boosting vs stacking for noisy non-linear regression | same (T02 Ensemble; paper metric Accuracy, repo metric RMSE) |
| ML03 | Nelder-Mead, Powell, CMA-ES on non-convex functions | same |
| ML04 | Standard, min-max, robust scaling on KNN classification | same |
| ML05 | Dimensionality reduction preserving cluster structure | same |
| ML06 | Adaptive learning-rate schedules for logistic regression | same |
| ML07 | TF-IDF vs count vs hashing with NB/SVM | same |
| ML08 | Imbalance-handling strategies on sklearn binary classifiers | same |
| ML09 | Bayesian optimization vs grid vs random search for RF tuning | same |
| ML10 | Cross-validation strategy reliability for small-sample model selection | same |
| ML11 | Classical unsupervised outlier detectors under anomaly injection | same |
| ML12 | Clustering algorithms on non-convex and anisotropic shapes | same |
| ML13 | GP regression kernel choice on 1-D and 5-D functions | same |
| ML14 | Split-conformal, Mondrian conformal, CQR intervals | **differs**: T14 Sparse linear models (Lasso, ElasticNet) |
| ML15 | Filter-based vs embedded L1 feature selection with noise features | same |
| ML16 | Bandit algorithm robustness under stationary and drifting rewards | **differs**: T16 Time-series forecasting |
| ML17 | LDA vs NMF vs LSA on 20newsgroups subsets | same |
| ML18 | Post-hoc calibration methods for sklearn classifiers | **differs**: T18 Transfer learning |
| ML19 | Graph- vs pseudo-label semi-supervised learning | same |
| ML20 | Classical forecaster robustness on seasonal time series | **differs**: T20 Active learning |
| ML21 | PC vs GES vs NOTEARS-linear on Gaussian linear-SEM DAGs | same |
| ML22 | Active learning query strategies for logistic regression | **differs**: T22 Multi-label classification |
| ML23 | Pointwise vs pairwise vs listwise learning-to-rank | same |
| ML24 | Online classification under concept drift: SGD, PA, NB, FTRL | **differs**: T24 GP regression |
| ML25 | Reservoir computing vs MLP vs GP for Lorenz-63 forecasting | same |

Science topics.
The paper evaluates 20: B01-B07 biology (COBRApy, BiGG), S01-S03 statistics, P01-P10 HEP-ph (MadGraph5_aMC@NLO, Pythia8, Delphes, FeynRules, MadAnalysis5) (p.7; Table 4 caption, p.8).
The repo adds Q01-Q10 quantum (Qiskit), giving 55 manifests: `config/{biology,physics,quantum,statistics,ml}/manifests/`.
Examples: B07 "Reproducible FBA protocol benchmark: FBA vs pFBA vs loopless/FVA on E. coli"; S01 "Bootstrap confidence intervals under non-Gaussian data"; S02 "Double machine learning for average treatment effect estimation"; S03 "Reliability of LLM-assisted statistical model selection"; P03 "8 TeV LHC dilepton exclusion contours for a B-L Z' boson with kinetic mixing"; Q01 "Comparing quantum data encoding strategies for variational classifiers"; Q03 "Classical optimizer comparison for VQE on H2 under finite shot noise".

Compute: topics are "CPU-executable in under 10 minutes on a single core" (App. D.1, p.17); sandbox limits 8 GB memory, wall-clock timeout "default 300-600 s" (App. C, p.17); every one of the 55 manifests declares `gpu_required: false` (checked).
Scoring: strict rubric per topic (`config/<domain>/rubrics/*.json`), CD:CE:RA = 25:25:50, number-grounding penalty (App. D.2, p.19); the paper's judges were a Claude Code subagent (Opus 4.7), a Codex CLI agent (GPT-5.4) and a human (App. D.2, p.19).
Scores existing outputs: yes, if the outputs are mapped to ARC-Bench's `submission/` layout, which the released Agent Lab adapter does (`experiments/arc_bench/baseline/adapters/agent_lab_adapter.py`, present at `be4ba47`).
The repo README's note that Agent Lab "Fair-input runs die in lit-review SUMMARY-loop" is at `experiments/arc_bench/baseline/README.md` line 82 (held).
Released baseline outputs: none (baselines note §2.4, held).

### 2.3 SAGE's frozen 12-topic ARC-Bench subset

| Item | Value |
|---|---|
| Paper | Ma et al., "One Reflection Is Not Enough", arXiv 2606.31478v1 |
| Repo | https://github.com/JieMaMagic/SAGE (MIT, pushed 2026-07-02); contents listing shows the system (`researchclaw/`, prompts, tests); evaluation outputs not found in the listing |
| Systems reported | AIS2, re-run on the same topics with the same backbone, whose name is not given (p.8) |

Numbers: AIS2 Code Dev 58.3, Code Exec 51.7, Result Analysis 39.2, Overall 48.2 (2:2:3 weighting, 0-100, blind Claude Opus 4.8 judge), 2 per-topic wins vs SAGE's 7 (Table 1, p.8); blind human rating on the 6 ML topics, AIS2 4.72/10 (Table 4, p.12).
Per-topic AIS2 scores are not published (Figure 3, p.12, gives per-topic scores for SAGE and its ablation only).

Task list (Table 6, p.19; frozen before execution, App. A, p.19):

| ID | Domain | SAGE's one-line description |
|---|---|---|
| ML01 | ML | Dropout variants for tabular MLPs |
| ML02 | ML | Ensemble regressors under noise |
| ML12 | ML | Clustering: algorithm vs. selector effect |
| ML16 | ML | Bandit regret decomposition |
| ML18 | ML | Minority-aware post-hoc calibration |
| ML20 | ML | Classical seasonal forecasters |
| P03 | Physics | B-L Z' dilepton exclusion recast |
| Q01 | Quantum | VQC data-encoding audit |
| Q03 | Quantum | VQE optimizers under shot noise |
| B07 | Biology | FBA variants on E. coli |
| S01 | Statistics | Bootstrap coverage audit |
| S02 | Statistics | (row not extracted from the PDF text; S02 is named in App. A, p.19) |

Compute: all 12 manifests declare `gpu_required: false`; P03 needs the MadGraph toolchain, Q01 and Q03 need Qiskit, B07 needs COBRApy (manifest text).
Cost: "a complete end-to-end SAGE run costs approximately $10 to $20" (p.8).

### 2.4 MLReplicate

| Item | Value |
|---|---|
| Paper | Gaddipati et al., arXiv 2605.16616v1 |
| Repo | https://github.com/gsasikiran/MLReplicate-benchmarking at `c0bdd77` (Apache-2.0) |
| Systems reported | AL (o3-mini), AIS2 (gpt-4o-mini, gpt-4o, o3-mini), plus AI Scientist v1, AI-Researcher, CycleResearcher, TinyScientist (Table 3, p.4); S1 not evaluated |

Numbers are in the baselines note §5.3 and held where I re-checked them (5 of 10 acceptances for AL, 0 for AIS2, p.5; 59% for AL, p.7).
New: Figure 2 (p.5) labels the 10 accepted papers "AL-P5 AL-P4 AL-P2 AL-P6 AL-P3 AIR-P5 CR-P3 CR-P7 TS-P8 TS-P1", and the repo's `icais-conference/accepted-papers/agent-laboratory/` holds exactly the AL PDFs for 11119 (P4), 11432 (P2), 12176 (P6), 14095 (P3) and 1940 (P5).
New: the AL task notes were "generated by gpt-5 by inputting the corresponding paper from the dataset" (App. B.3, p.19), which is a form of input the other systems did not get in the same shape.

Task list (Table 2, p.3), with the repo's file stem used for every system's inputs and outputs:

| ID | Paper | Topic | Repo stem | AL accepted | AIS2 PDF released |
|---|---|---|---|---|---|
| P1 | Roll the dice & look before you leap (Nagarajan et al., 2025) | Language Models | `12175_Roll_the_dice_look_befor` | no | yes |
| P2 | Conformal Prediction as Bayesian Quadrature (Snell and Griffiths, 2025) | Probabilistic Methods | `11432_Conformal_Prediction_as_` | yes | yes |
| P3 | Train for the Worst, Plan for the Best (Kim et al., 2025b) | Generative Models | `14095_Train_for_the_Worst_Plan` | yes | yes |
| P4 | The Value of Prediction in Identifying the Worst-Off (Fischer-Abaigar et al., 2025) | Fairness | `11119_The_Value_of_Prediction_` | yes | yes |
| P5 | CollabLLM (Wu et al., 2025) | Human-AI Collaboration | `1940_CollabLLM_From_Passive_Re` | yes | yes |
| P6 | Score Matching with Missing Data (Givens et al., 2025) | Unsupervised Learning | `12176_Score_Matching_with_Miss` | yes | yes |
| P7 | Position: The AI Conference Peer Review Crisis (Kim et al., 2025a) | Meta-Science | `53_Position_The_AI_Conference_` | no AL PDF released | yes |
| P8 | Position: AI Safety should prioritize the Future of Work (Hazra et al., 2025) | AI Safety | `463_Position_AI_Safety_should_` | no | yes |

Compute per task: not stated by the paper.
From the released AL YAMLs (my reading, not run): P2 asks for a pretrained multilabel image classifier over MS-COCO; P4 allows a synthetic fallback dataset with Gaussian and tree models; P6 uses Gaussian, ICA-like and GGM models with a 2-layer MLP of width 200.
So P4 and P6 look CPU-scale, P2 needs image inference, P3 and P5 need model training (masked diffusion, CollabLLM).
AIS2's runs used "two NVIDIA L40S GPUs (48GB GPU memory each)" (App. B.2, p.17); AL's hardware is not stated (App. B.3, p.19).
Scoring: ICAIS 2025 automated reviewers (DeepReviewer, ZGCA, SafeReviewer) then three human reviews with a binary hallucination judgment (p.5-6).
Scores existing outputs: yes; the released PDFs and LaTeX are the scorer's input.
Released: `dataset/*.pdf` (the 8 source papers), per-system inputs, `icais-conference/<system>/generated_pdfs|latex|reviews`, 8 AIS2 PDFs, 7 AL PDFs.

### 2.5 ADRS with the CoE Integrity Audit (ScientistOne paper)

| Item | Value |
|---|---|
| Paper | Meng et al., ScientistOne, arXiv 2605.26340v1; project page https://scientist-one.github.io/ |
| Task repo | https://github.com/UCB-ADRS/ADRS at `2139fd8` (Apache-2.0), `openevolve/examples/ADRS/<task>/` |
| Output repo | https://github.com/scientist-one/generated-artifacts at `721f1fb` (Apache-2.0) |
| Systems reported | S1 and AIS2 ("Sakana AI-Scientist v2"), plus AutoResearchClaw, DeepScientist, AI-Researcher; all on Gemini 3.1 Pro, 3 seeds per task, up to 20 solver iterations, 2-hour code-generation windows (App. G, p.33); AL not evaluated |

Task list and best-of-3 solver scores (§6, p.7; Table 3, p.12; repo folder name in brackets):

| Task [ADRS folder] | What it optimizes (p.7) | Dir. | Human | AIS2 | S1 | Files in the ADRS folder |
|---|---|---|---|---|---|---|
| Prism [`prism`] | LLM-serving model placement across GPUs | up | 21.89 | 26.26 | 26.26 | evaluator, initial program, requirements (numpy, scipy, sklearn, pykalman, PyWavelets) |
| Cloudcast [`cloudcast`] | cloud network cost | down | 626.24 | 627.11 | 618.08 | evaluator, simulator, profiles |
| EPLB [`eplb`] | expert-parallel load balancing for MoE | up | 0.1265 | 0.1270 | 0.1459 | needs `torch` and `expert-load.json` from HF `abmfy/eplb-openevolve` (README) |
| LLM-SQL [`llm_sql`] | table layout for LLM prefix-cache reuse | up | 0.6920 | 0.7320 | 0.7222 | 5 CSV datasets, 2.5-34 MB each |
| TXN [`txn_scheduling`] | transaction scheduling, makespan | up | 2724.8 | 4184 | 3906 | simulator and workloads |

The ADRS repo has 10 example folders; the S1 paper uses these 5.
Integrity audit (Table 1, p.8): AIS2 I1 score verification 5/12, I2 specification violation 10/15, I3 hallucinated references 0/159, I4 method-code alignment 5/15; S1 12/12, 0/15, 0/337, 14/15.
The paper says I2 and I4 for AIS2 are confounded by "the BFTS-ADRS design mismatch" and should be excluded from cross-system comparison (p.8).
ScholarPeer average of 15 papers: AIS2 Overall 2.5, 0/15 accepted; S1 4.5, 6/15 (Table 2, p.11).
Scoring: fixed deterministic evaluators run five times with tolerance max(1%, 3σ/|mean|) (p.7); ScholarPeer (Goyal et al., arXiv 2601.22638); human verification of all I1-I3 flags (p.8).
Scores existing outputs: yes for I1 (re-run the evaluator on submitted solver code) and I3 (check the bibliography); only S1's 15 ADRS papers and 15 solver files are released (`generated-papers/adrs/`, `solution-code/adrs/`).
Compute per task: not stated by the paper; from the file lists, four are CPU simulators and EPLB needs PyTorch (held from baselines note §6.4).
Whether ScholarPeer's official code is released: not verified; I found only a third-party re-implementation (`amirkiarafiei/open-scholar-peer`).

### 2.6 MLE-bench as used by ScientistOne, and Parameter Golf

S1 compares only with DeepScientist here; AIS2 and AL are not evaluated (Table 4, p.13).
Task IDs from Table 15 (p.32); split membership checked against `experiments/splits/{medium,high}.txt` in `openai/mle-bench` at `507f92e` (MIT, LICENSE file read):

| Task (S1 name) | MLE-bench ID | Split | S1 score | S1 medal | DeepScientist |
|---|---|---|---|---|---|
| 3D Object Detection | 3d-object-detection-for-autonomous-vehicles | high | 0.1763 | Gold | 0.0000, below median |
| AI4Code | AI4Code | medium | 0.8356 | above median | 0.6964, below median |
| iMet 2020 FGVC7 | imet-2020-fgvc7 | medium | 0.6791 | Silver | 0.6804, Silver |
| RSNA Brain Tumor | rsna-miccai-brain-tumor-radiogenomic-classification | high | 0.6518 | Gold | 0.6377, Gold |
| iNaturalist 2019 FGVC6 | inaturalist-2019-fgvc6 | medium | 0.2445 (lower is better) | Silver | 0.2158, Silver |
| Parameter Golf | (OpenAI live competition) | - | 1.0600 BPB, SOTA at the 2026-04-27 cutoff | - | invalid, size limit exceeded |

Compute: "8xH100 GPUs, 192 CPU cores, and 1TB of RAM"; up to 16 grading-server queries, which "deviates from the official MLE-Bench protocol" (App. F, p.33).
Parameter Golf: 16 MB artifact, under 10 minutes on 8x H100, bits per byte on FineWeb validation (p.33).
Released: S1's 5 MLE papers, 1 Parameter Golf paper and solution code (`generated-papers/{mle,pg}/`, `solution-code/{mle,parameter-golf}/`).
None of these fit one consumer GPU.

### 2.7 Agent Laboratory's own evaluations

Paper: Schmidgall et al., arXiv 2501.04227v2; repo https://github.com/SamuelSchmidgall/AgentLaboratory at `d9017d9` (MIT).

**MLE-bench subset (§4.4 and Figure 9, p.18).**
"all challenges focusing on text and tabular data from the low complexity category of MLE-Bench" (p.18).
Inputs differ from standard MLE-bench: mle-solver gets "distilled knowledge from Kaggle notebooks" and scores on a 20% dev split (p.18).
The backbone model for this experiment is not named in the text (not found on p.18-19).
Figure 9 prints the task names only as display titles; the IDs below are my mapping to `experiments/splits/low.txt`, where all ten appear, and the dataset sizes are from the mle-bench README Lite table.

| Figure 9 title | MLE-bench ID (my mapping) | Data type | Size (GB) | mle-solver score | Above median | Medal |
|---|---|---|---|---|---|---|
| detect insults in commentary | detecting-insults-in-social-commentary | text | 0.002 | 0.839 | yes | gold |
| dec 2021 tab playground | tabular-playground-series-dec-2021 | tabular | 0.7 | 0.961 | yes | gold |
| predict trans. conductors | nomad2018-predict-transparent-conductors | tabular | 0.00624 | 0.062 (min) | yes | silver |
| english text normalization | text-normalization-challenge-english-language | text | 0.01 | 0.990 | yes | bronze |
| may 2022 tab playground | tabular-playground-series-may-2022 | tabular | 0.57 | 0.992 | yes | - |
| random acts of pizza | random-acts-of-pizza | text | 0.003 | 0.643 | yes | - |
| spooky author identification | spooky-author-identification | text | 0.0019 | 0.532 (min) | no | - |
| jigsaw toxic comments | jigsaw-toxic-comment-classification-challenge | text | 0.06 | 0.874 | no | - |
| russian text normalization | text-normalization-challenge-russian-language | text | 0.01 | 0.000 | no | - |
| NYC taxi fare prediction | new-york-city-taxi-fare-prediction | tabular | 5.7 | 6.542 (min) | no | - |

Check on the baselines note §10: "4 medals (2 gold, 1 silver, 1 bronze), above median on 6/10" held.
"AIDE (o1-preview) 5/10" matches the text on p.19, but Figure 9 shows six "above median" marks for AIDE (insults, predict conductors, may 2022, pizza, spooky, jigsaw); text and figure disagree.
Scores existing outputs: yes in principle (mle-bench grader on a submission CSV); AL's submissions are not released (not found in the repo).

**Human study, autonomous mode (§4.1, p.10-11).**
Five research questions, each run with gpt-4o, o1-mini and o1-preview, 15 papers in all (p.10-11):
(1) cognitive biases in language models; (2) image transformers vs CNNs under pixel noise; (3) differential diagnosis prompting on MedQA; (4) word-order sensitivity in multiple-choice benchmarks; (5) gender role play and math accuracy.
Experimental-quality ratings 2.6/5 (gpt-4o), 3.2/5 (o1-mini), 2.9/5 (o1-preview) (p.11).
Not re-runnable as a benchmark: volunteer PhD raters.

**AgentRxiv (Schmidgall and Moor, arXiv 2503.18102v1; https://agentrxiv.github.io links only the paper and the AL repo).**
One task: "Improve accuracy on MATH-500 using reasoning and prompt engineering", with gpt-4o mini as the experiment model (p.7).
Sequential labs: 70.2% baseline to 78.2% with Simultaneous Divergence Averaging, "+11.4%" relative (p.7-8).
Parallel labs: 79.8%, "13.7%" relative (p.11).
Transfer of SDA: GPQA 36.4% to 38.9%, MMLU-Pro 63.1% to 70.8%, MedQA 74.9% to 81.6% (p.8).
The AL repo ships `experiment_configs/MATH_agentlab.yaml` and `MATH_agentrxiv.yaml`; `MATH_agentlab.yaml` sets `llm-backend: "o3-mini"`, `copilot-mode: True`, `mlesolver-max-steps: 3`, `papersolver-max-steps: 1`.
Compute: API calls only.
Generated papers: not released (project page links only paper and code).

### 2.8 AI Scientist v2's own evaluation

Paper: Yamada et al., arXiv 2504.08066v1; repo https://github.com/SakanaAI/AI-Scientist-v2 at `96bd516`, licence "The AI Scientist Source Code License" (README; GitHub reports NOASSERTION); outputs https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment (no licence file reported by the API).
The only evaluation is blind peer review at the ICLR 2025 ICBINB workshop: 3 submissions among 43, 1 accepted with scores 6, 6, 7 (mean 6.33) (p.9); titles in Table 4 (p.31): "Compositional Regularization: Unexpected Obstacles in Enhancing Neural Network Generalization" (accepted), "Unveiling the Impact of Label Noise on Model Calibration in Deep Learning" (rejected), "Real-world Challenges in Pest Detection using Deep Learning" (rejected).
The output repo holds `compositional-regularization/`, `label-noise/`, `pest-detection/` and `ai-reviewing/`.
Compute: "designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch"; experimentation "typically costs around $15-$20 per run" with Claude 3.5 Sonnet plus about $5 for writing (repo README).
These three papers are the "public AIS2 papers" that rows 16-17 of the summary re-score.

### 2.9 Hidden Pitfalls of AI Scientist Systems (SPR)

Numbers are in the baselines note §7 and I did not re-derive them.
New: the task list.
One synthetic Symbolic Pattern Reasoning task with a suite of 20 SPR datasets that form a strict difficulty ladder over rule complexity, vocabulary size (up to 4 shapes and 4 colours) and sequence length, grouped into five hidden tiers, exposed under random five-letter codes, each with 2,000/500/1,000 train/validation/test examples (p.5-6).
Four probes run on it: inappropriate benchmark selection (pick 4 of 20), data leakage, metric misuse, post-hoc selection bias (§4, p.5-9).
Systems: AL with "default LLM API configurations" and AIS2 with its code model swapped to o3-mini (p.9).
Repo https://github.com/niharshah/AIScientistPitfalls at `2725bda` (no top-level licence per the API) has `SPR Task/{BenchmarkIssue,DataLeakage,MetricMisuse,PostHocSelection}`, `pitfall_detection/pitfall_detection.py`, and full copies of both systems with `generated_research/`.
Scores existing outputs: yes; the auditor prompt reads paper, code and logs (App. D, p.23).

### 2.10 REPRODUCE-Bench (AutoReproduce)

Paper: arXiv 2505.20662v4, ACL 2026 Main (arXiv comment); repo https://github.com/AI9Stars/AutoReproduce (no licence reported by the API).
AL (GPT-4o): Paper-Level 63.47, Code-Level 35.32, Mixed-Level 48.64, Exec Rate 23.08%, Perf Gap 82.31 (Table 2, p.6; mean of three runs, o1 judge).
AIS2 and S1: not evaluated.
Task list (Table 1, p.5): IEBins (depth, NYU-Depth-v2), iTransformer (forecasting, Traffic), DKD (distillation, CIFAR-100), SimVP (video prediction, Moving MNIST), HumanMAC (motion, HumanEva-I), SFNet (dehazing, SOTS-Indoor), LSM (PDEs, Darcy), Swin-Unet (segmentation, Synapse), TDGNN-w (node classification, Citeseer), TimeVAE (time-series generation, Sine), WCDM (low-light enhancement, LOLv1), BSPM (collaborative filtering, Gowalla), DAT-S (super-resolution, DF2K/Set5).
Compute: reference results rerun "on Tesla A100 GPUs" (p.12).
This measures code reproduction of a given paper, not open-ended research, and has no manuscript for Gate 3.

### 2.11 ARAC-Bench

| Item | Value |
|---|---|
| Paper | Cui et al., arXiv 2608.12788v1 |
| Repo | https://github.com/cuijiale2004-hash/ARAC-Bench at `8616ad5` (no licence file; README says data may carry "separate licenses or platform terms") |
| Systems reported | AIS2 61.68 and AgentLaboratory 55.65 of 100, same backbone Kimi-K2.6, plus 9 others; best ARC-Full-Auto 67.9 (Table 3, p.9); S1 not evaluated |

Component scores (Table 3, p.9):

| System | Related (5) | Idea (25) | Bench. (10) | Coding (30) | Hyperpar. (5) | Basic (10) | Method. (15) | Total |
|---|---|---|---|---|---|---|---|---|
| AI-Scientist-v2 | 2.24 | 15.21 | 3.54 | 18.77 | 2.72 | 7.13 | 12.07 | 61.68 |
| AgentLaboratory | 2.21 | 14.14 | 2.14 | 18.08 | 2.10 | 6.93 | 10.05 | 55.65 |

Per-theme totals (Table 7, p.15): AIS2 DL 61.59, Diffusion 60.09, LLM 60.50, MM 63.77, RL 62.73; AL DL 57.43, Diffusion 55.96, LLM 54.79, MM 55.39, RL 54.91.
Tasks: 200 ICLR 2026 papers as "Gold References" (p.3); the repo has `<theme>/paper_<n>/` with `GivenTopic-Inspiration.md`, `IdeaDetails.md`, `Benchmark.json`, `CodingModule.json`, `Parameter.json`, `Principle.md`, `RelatedWork.json`, `full.md`.
Counts by folder: DL 32, Diffusion 50, LLM 37, MM 36, RL 45.
Sample topics (first lines of `GivenTopic-Inspiration.md`, paper_1 of each theme): spiking-neuron membrane dynamics beyond first-order ODEs (DL); reducing diffusion sampling cost (Diffusion); LR schedule design in LLM pre-training (LLM); visual token compression in MLLMs (MM); off-policy RL with verifiable rewards (RL).
Scoring: three stages scored separately with ground truth from earlier stages fed in (Proposal 40, Experiment 35, Synthesis 25); ACS rubric; "All AI scoring is done using GPT-5.2" (p.8); literature search cut off at mid-2025 (p.6-7); code scored against a library of 2,869 modules with unit tests (p.8).
Compute: not stated.
Scores existing outputs: no; the protocol feeds gold inputs per stage, and the repo has data only, no evaluation scripts ("Exact installation and evaluation commands will depend on the scripts included in the final release", README).
System outputs: not released.

### 2.12 FML-bench v2

Paper: Zou et al., arXiv 2605.17373v2; repo https://github.com/qrzou/FML-bench at `d336651` (Apache-2.0).
AIS2 appears as "TAS v2", its search strategy running inside FML-bench's shared code editor and executor (§3.2, p.4-5), GPT-5.4 backbone, 3 rounds x 100 steps (p.6); it is not the released AIS2 pipeline and writes no paper.
AIS2 mean normalized test improvement 0.193 ± 0.237 across 18 tasks, second of seven (Tables 2-3, p.7); AL and S1 not evaluated.
Task list (Table 5, p.16): DomainBed-CM (ColoredMNIST), DomainBed-OH (OfficeHome, ResNet-50), EasyFSL (Mini-ImageNet features), USB (FixMatch, CIFAR-100), Lightly (MoCo, CIFAR-10), Solo-learn (Barlow Twins, CIFAR-100), Cont.-Learn. (Synaptic Intelligence, splitMNIST), PyCIL (iCaRL, CIFAR-100), CausalML (Dragonnet, IHDP), gCastle (NOTEARS, 50-node DAGs), ART (dp-instahide, poisoned MNIST), OpenOOD (MSP, CIFAR-10), PrivacyMeter (WRN-28-2, CIFAR-10), Opacus (DP-SGD, CIFAR-10), AIF360 (Adversarial Debiasing, COMPAS), Fairlearn (logistic regression, Adult), Unlearning (TOFU, Llama-3.2-1B), PFLlib (FedAvg, CIFAR-10).
Compute: one validation run under 40 minutes on one GPU (p.4); runs used A100-80GB (p.6).
Scores existing outputs: no.

### 2.13 Re-scoring studies of public papers (evaluation-only)

These score papers that systems already released; none runs a system.

| Study | Scorer | AIS2 | S1 | Where |
|---|---|---|---|---|
| "AI Scientists Fail Without Strong Implementation Capability", arXiv 2506.01372v2 (position paper) | DeepReviewer-14B | n=3: Soundness 1.67, Presentation 1.50, Contribution 1.50, Rating 2.33 | - | Table 2, p.7 |
| Jr. AI Scientist, arXiv 2511.04583, TMLR 2026 | DeepReviewer-14B | n=3: Rating 2.75 | - | Table 2a, p.12 |
| ScientistTwo, arXiv 2609.19644v1 | ScholarPeer; Stanford Agentic Reviewer (paperreview.ai) | n=3: 2.0 ± 1.0, 0.0%; 2.5 ± 0.2, 0.0% | n=21: 3.8 ± 1.2, 14.3%; 4.1 ± 0.7, 0.0% | Table 2, p.10 |

The two DeepReviewer-14B studies give different ratings (2.33 vs 2.75) for what are presumably the same three public AIS2 papers; neither lists the paper files, so this is not resolved.
AL is in none of these tables.
ScientistTwo's own 107 problems (38 NeurIPS 2025, 5 ICLR 2026, 64 ICML 2026 spotlight papers; Tables 12-14, p.28-29) are ScientistTwo-only; the baselines were scored from their public papers, not run on these problems.

### 2.14 BadScientist

No numbers for AIS2, S1 or AL (baselines note §3.3, held: the generator is the authors' GPT-5 agent adapted from AI-Scientist, p.6).
Task list: 25 GPT-5-generated seed topics listed in App. B (p.15), for example "Self-consistent diffusion models that satisfy counterfactual causal constraints", "Graph-grounded RAG: joint learning of knowledge graphs and retrievers for verifiable answers", "RouteBench: measuring strategic routing, tool selection, and delegation in multi-agent LLM systems".
Repo `Bad-Scientist/BadScientist` at `f6fc8ec` has `examples/seed_idea.json`; the full 25-topic file was not found in the tree listing.
Scores existing outputs: yes; it is a reviewer panel over papers.

---

## 3. Recommended comparison tasks

Constraints assumed: one consumer GPU (the baselines note §13 names an 8 GB card), about 28 four-level waves before writing (baselines note §11), DeepSeek backbone for our runs.
Every published number below used a different backbone, so each comparison is different-model unless we re-score released outputs with our own judges.

| Priority | Tasks | Compared against | Why | Caveat |
|---|---|---|---|---|
| 1 | MLR-Bench, all 10 end-to-end tasks (§2.1) | AIS2 per-task review scores and per-task human-verified hallucination labels; MLR-Agent | Only benchmark with per-task AIS2 numbers and released AIS2 papers, code and logs, so our judges can re-score AIS2 on exactly our tasks | GPU need per task unknown; AIS2 used 4x RTX 3090 |
| 2 | ARC-Bench ML01-ML25 (§2.2), or at minimum SAGE's 6 ML topics ML01, ML02, ML12, ML16, ML18, ML20 plus S01, S02 (§2.3) | AIS2 0.419 (25-topic mean) and AIS2 48.2 (SAGE 12-topic mean) | CPU only, under 10 minutes per experiment, AL adapter already in the repo, per-hypothesis manifests feed Gate 2; running the whole published set is the only way to match an aggregate, since no per-topic AIS2 score is published | Repo IDs ML14, ML16, ML18, ML20, ML22, ML24 are different topics from the paper's T14-T24 (§2.2); SAGE's 48.2 needs P03 (MadGraph), Q01, Q03 (Qiskit) and B07 (COBRApy) too |
| 3 | MLReplicate P4 and P6, then P2 (§2.4) | AL (o3-mini) and AIS2 released PDFs on the same spec; AL accepted on both P4 and P6 | Same task, both other-system outputs released, so Gate 3 and our judges can score AL-as-released, AIS2 and AL+GATES side by side; P4 and P6 look CPU-scale from their YAMLs | Compute judged from spec text only; ICAIS reviewers not reproducible; AL's released task notes were written by GPT-5 from the source paper |
| 4 | Re-score only, no runs: the 3 public AIS2 papers and S1's 21 papers (§2.5, §2.13) | S1's I3 0/337 and ScholarPeer 3.8-4.5; AIS2's DeepReviewer 2.33-2.75 | The only S1 comparison that needs no GPU; DeepReviewer-14B is an open model | ScholarPeer's official code not verified; Stanford reviewer is hosted |
| 5 | ADRS TXN, LLM-SQL and Cloudcast (§2.5) | S1 and AIS2 best-of-3 solver scores (Table 3) and I1/I3 audits | Only per-task numbers for S1 that fit a laptop; I1 is Gate 1's contract | New harness; Gemini 3.1 Pro vs our model; only S1's outputs released |
| 6 | AL's own MLE-bench Lite 10 (§2.7) and AgentRxiv MATH-500 (§2.7) | AL's published per-task mle-solver scores; AL's 70.2% to 78.2% trajectory | Same host, small data, a direct check that our L0 host reproduces AL's own published numbers | No AIS2 or S1 numbers; MLE-bench has no manuscript for Gate 3; AgentRxiv used gpt-4o mini as experiment model |

Not recommended for the draft: ARAC-Bench (both AIS2 and AL on one backbone, but no evaluation scripts and a stage-isolated protocol that bypasses the full pipeline the gates wrap), FML-bench v2 (A100-80GB, strategy-only AIS2), S1's MLE-bench tasks and Parameter Golf (8x H100), REPRODUCE-Bench (A100 reference runs, no manuscript).
ARAC-Bench's Table 3 is still worth citing as the only same-backbone AIS2 vs AL comparison: 61.68 vs 55.65.

---

## 4. Open questions and claims not verified

1. Whether ARC-Bench's published 0.419 for AIS2 was measured on the paper's Table 9 topics or on the repo's ML01-ML25; six topic IDs differ between them (§2.2), and App. E (p.20) describes a third mix.
2. Which 6 of 25 ARC-Bench topics AIS2 failed; only "dynamical systems, causal discovery" is said (p.7).
3. SAGE's backbone model name and whether its AR-Eval and Table 1 rubric prompts and outputs are in its repo; I listed only the repo root.
4. The S02 row of SAGE Table 6 did not extract from the PDF text; S02 is named in App. A (p.19) and in Figure 3's axis (p.12).
5. The Figure 9 task-name to MLE-bench-ID mapping for AL's 10 tasks is mine; the paper prints display titles only.
6. Figure 9 of 2501.04227v2 shows AIDE above median on 6 of 10 tasks while the text (p.19) says 5; not resolved.
7. AL's backbone for the MLE-bench experiment is not named (p.18-19).
8. Why two DeepReviewer-14B studies rate the same three public AIS2 papers 2.33 (2506.01372v2) and 2.75 (Jr. AI Scientist); the paper files each used are not listed.
9. MLReplicate per-task compute and whether P2, P4 or P6 run on an 8 GB GPU; judged from YAML text only.
10. MLR-Bench per-task GPU need; not stated in the paper.
11. ARAC-Bench compute, licence and evaluation code; the repo has data only at `8616ad5`.
12. Whether ScholarPeer has an official code release; only a third-party re-implementation was found.
13. AgentRxiv's generated papers are not linked from the project page; not found elsewhere.
14. The full BadScientist 25-topic seed file in the repo; not found in the tree listing, the list is in App. B (p.15).
15. Venue of the Agent Laboratory paper: not stated in the arXiv metadata; a survey (2605.18661v2, Table 11, p.52) says EMNLP'25, which I did not check against the ACL Anthology.

---

## Provenance

| Source | Read from | Parts used |
|---|---|---|
| MLR-Bench 2505.19955v3 | local `Sources/MLR Bench.pdf`; kit review JSONs; repo at `f728d57` via GitHub API | Tables 7, 8; p.3-4; `tasks/` listing; `ai_scientist_v2_papers/o4-mini/`; `human_eval/Hallucination Check.csv` |
| AutoResearchClaw 2605.20025v2 | local PDF; repo at `be4ba47`; HF dataset at `5a5923b` | Tables 2, 4, 9; App. C, D.1, E; all 55 manifests; `config/ml/topics.yaml`; baseline README and adapter listing |
| SAGE 2606.31478v1 | local PDF; repo root listing | Tables 1, 4, 6; App. A; Figure 3 axis |
| MLReplicate 2605.16616v1 | local PDF; repo at `c0bdd77` | Tables 2, 3; Figure 2 labels; App. B.2-B.3; three AL YAMLs; generated and accepted PDF listings |
| ScientistOne 2605.26340v1 | local PDF; `scientist-one/generated-artifacts` at `721f1fb`; `UCB-ADRS/ADRS` at `2139fd8` | Tables 1-4, 15; §6-7; App. F-G; task folder listings, EPLB README, Prism requirements |
| MLE-bench | repo `openai/mle-bench` at `507f92e` | `experiments/splits/{low,medium,high}.txt`, README Lite table, LICENSE |
| Agent Laboratory 2501.04227v2 | arXiv PDF; repo at `d9017d9` | §4.1, §4.4, Figure 9 (rendered at 220 dpi); `experiment_configs/MATH_agentlab.yaml` |
| AgentRxiv 2503.18102v1 | arXiv PDF; project page | p.7-8, p.11 |
| AI Scientist v2 2504.08066v1 | arXiv PDF; repo README at `96bd516`; workshop-experiment repo listing | §4, Table 4 |
| Hidden Pitfalls 2509.08713v2 | arXiv PDF; repo at `2725bda` | §4.1, §5; App. D; folder listings |
| AutoReproduce 2505.20662v4 | arXiv PDF | Tables 1-3; App. A.3 |
| ARAC-Bench 2608.12788v1 | arXiv PDF; repo at `8616ad5` | Tables 3, 4, 7; §3; README; folder counts; five topic files |
| FML-bench 2605.17373v2, 2510.10472v2 | arXiv PDFs; repo at `d336651` | Tables 2, 3, 5; §3-4; App. B |
| ScientistTwo 2609.19644v1 | arXiv PDF | Tables 2, 5, 7, 12-14 |
| AI Scientists Fail 2506.01372v2; Jr. AI Scientist | arXiv PDF; local `Sources/Jr.AI Scientist.pdf` | Table 2; Table 2a |
| BadScientist 2510.18003v2 | local PDF; repo at `f6fc8ec` tree | App. B |
| Surveys 2605.18661v2, 2608.05179v1, 2605.26200v1 | arXiv PDFs | Table 11 p.52; Table 3 p.11; table p.5 |
| Negative checks | arXiv PDFs of 2504.09702v3, 2510.27598, 2505.18705v1, 2509.26603v1, 2605.19156v1, 2606.07462v1, 2606.05080v1, 2606.09550v1, 2608.17271v1, 2605.10246v2 | searched for the three system names |

Text was extracted with `pdftotext -layout` into the session scratchpad; no repository file other than this one was written.
