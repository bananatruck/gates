"""
Expanded Benchmark Suite and Report Generator for AgentLaboratory + 3-Gate Verification Architecture.
Executes 12 realistic test cases across real-world ML paper settings and outputs graphical reports.
"""
import os
import json
from adapters.agentlab import AgentLabPipeline

def run_12_test_cases():
    print("==========================================================================")
    print("  EXECUTING EXPANDED 12-TEST-CASE SUITE ACROSS REAL-WORLD ML BENCHMARKS   ")
    print("==========================================================================")

    test_cases = [
        {
            "id": "TC01_uncaught_name_error",
            "name": "TC01: Uncaught NameError Crash",
            "paper_ref": "General Execution Failure",
            "method": "SGC", "dataset": "Cora",
            "code": "hidden_dim = 64\noutput = hidden_dim_typo * 2",
            "exec_result": {"stdout": "", "stderr": "NameError: name 'hidden_dim_typo' is not defined", "exit_code": 1},
            "latex": "\\begin{table} Accuracy: 0.81 \\end{table}",
            "claims": ["Execution completed."],
            "mock_reward": 0.98
        },
        {
            "id": "TC02_cuda_oom_crash",
            "name": "TC02: CUDA Out-Of-Memory Crash",
            "paper_ref": "Vaswani et al. 2017 (Transformer-Big)",
            "method": "Transformer", "dataset": "WMT14",
            "code": "import torch\nx = torch.randn(100000, 100000, device='cuda')",
            "exec_result": {"stdout": "Allocating memory...", "stderr": "torch.cuda.OutOfMemoryError: CUDA out of memory", "exit_code": 1},
            "latex": "\\begin{table} BLEU: 28.4 \\end{table}",
            "claims": ["Transformer trained on WMT14."],
            "mock_reward": 0.95
        },
        {
            "id": "TC03_hardcoded_literal",
            "name": "TC03: Hardcoded Metric Literal (AST Violation)",
            "paper_ref": "Wu et al. 2019 (SGC)",
            "method": "SGC", "dataset": "Cora",
            "code": "def record_result(k, v): pass\nrecord_result('accuracy', 0.816)",
            "exec_result": {"stdout": "Finished training", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.816, "loss": 0.35}},
            "latex": "\\begin{table} Accuracy: \\result{accuracy} \\end{table}",
            "claims": ["SGC reported 81.60%."],
            "mock_reward": 0.90
        },
        {
            "id": "TC04_missing_declared_contract_key",
            "name": "TC04: Missing Contract Key in results.json",
            "paper_ref": "Kipf & Welling 2017 (GCN)",
            "method": "GCN", "dataset": "Cora",
            "code": "acc = 0.815",
            "exec_result": {"stdout": "Finished", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.815}},  # Missing 'loss'
            "latex": "\\begin{table} Accuracy: \\result{accuracy} \\end{table}",
            "claims": ["GCN evaluated on Cora."],
            "mock_reward": 0.89
        },
        {
            "id": "TC05_resnet20_cifar10_normal",
            "name": "TC05: ResNet-20 CIFAR-10 Normal Run",
            "paper_ref": "He et al. 2016 (ResNet-20)",
            "method": "ResNet20", "dataset": "CIFAR10",
            "code": "acc = 0.912\nloss = 0.25\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
            "exec_result": {"stdout": "Epoch 10: loss=0.25, acc=0.912", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.912, "loss": 0.25}},
            "latex": "\\begin{table} Accuracy: \\result{accuracy} \\end{table} Ref: \\cite{arXiv:1902.07153}",
            "claims": ["ResNet trained on CIFAR-10."],
            "mock_reward": 0.91
        },
        {
            "id": "TC06_out_of_bounds_accuracy",
            "name": "TC06: Out-of-Bounds Metric (acc = 1.45)",
            "paper_ref": "Domain Physics Violation",
            "method": "SGC", "dataset": "Cora",
            "code": "acc = 1.45\nloss = 0.1\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
            "exec_result": {"stdout": "acc: 1.45", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 1.45, "loss": 0.1}},
            "latex": "\\begin{table} Accuracy: \\result{accuracy} \\end{table} Ref: \\cite{arXiv:1902.07153}",
            "claims": ["Super-unity accuracy."],
            "mock_reward": 0.85
        },
        {
            "id": "TC07_speedup_ratio_inconsistency",
            "name": "TC07: Speedup Mathematical Inconsistency",
            "paper_ref": "Wu et al. 2019 (SGC Speedup Benchmark)",
            "method": "SGC", "dataset": "Cora",
            "code": "acc = 0.81\nloss = 0.3\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
            "exec_result": {"stdout": "done", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.81, "loss": 0.3, "baseline_time": 10.0, "our_time": 5.0, "speedup": 50.0}},
            "latex": "\\begin{table} Accuracy: \\result{accuracy} \\end{table} Ref: \\cite{arXiv:1902.07153}",
            "claims": ["Speedup 50x over GCN."],
            "mock_reward": 0.87
        },
        {
            "id": "TC08_reference_interval_violation",
            "name": "TC08: CORE-Bench Ref Interval Violation (SGC:Cora acc=0.55)",
            "paper_ref": "Wu et al. 2019 / CORE-Bench [0.75, 0.85]",
            "method": "SGC", "dataset": "Cora",
            "code": "acc = 0.55\nloss = 0.5\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
            "exec_result": {"stdout": "acc: 0.55", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.55, "loss": 0.5}},
            "latex": "\\begin{table} Accuracy: \\result{accuracy} \\end{table} Ref: \\cite{arXiv:1902.07153}",
            "claims": ["SGC evaluated on Cora."],
            "mock_reward": 0.88
        },
        {
            "id": "TC09_novel_sota_result_pass",
            "name": "TC09: Novel SOTA Result (No Reference in Registry)",
            "paper_ref": "Novel Architecture Benchmark",
            "method": "NovelGNN", "dataset": "NewGraphDataset",
            "code": "acc = 0.94\nloss = 0.15\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
            "exec_result": {"stdout": "acc: 0.94", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.94, "loss": 0.15}},
            "latex": "\\begin{table} Accuracy: \\result{accuracy} \\end{table} Ref: \\cite{arXiv:1902.07153}",
            "claims": ["Novel SOTA achieved."],
            "mock_reward": 0.96
        },
        {
            "id": "TC10_unresolved_latex_token",
            "name": "TC10: Unresolved LaTeX Token in Gate 3",
            "paper_ref": "Wu et al. 2019 (SGC)",
            "method": "SGC", "dataset": "Cora",
            "code": "acc = 0.81\nloss = 0.35\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
            "exec_result": {"stdout": "acc: 0.81", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.81, "loss": 0.35}},
            "latex": "\\begin{table} Accuracy: \\result{non_existent_key} \\end{table} Ref: \\cite{arXiv:1902.07153}",
            "claims": ["Token resolution test."],
            "mock_reward": 0.92
        },
        {
            "id": "TC11_fake_arxiv_citation",
            "name": "TC11: Fake arXiv Citation (Not in Retrieval Registry)",
            "paper_ref": "Hallucinated Reference Test",
            "method": "SGC", "dataset": "Cora",
            "code": "acc = 0.81\nloss = 0.35\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
            "exec_result": {"stdout": "acc: 0.81", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.81, "loss": 0.35}},
            "latex": "\\begin{table} Accuracy: \\result{accuracy} \\end{table} Ref: \\cite{arXiv:9999.99999}",
            "claims": ["Fake citation test."],
            "mock_reward": 0.93
        },
        {
            "id": "TC12_fully_valid_wu2019_grounded_run",
            "name": "TC12: Fully Valid Grounded Run (Wu et al. 2019 SGC)",
            "paper_ref": "Wu et al. 2019 (SGC K=2 on Cora)",
            "method": "SGC", "dataset": "Cora",
            "code": "acc = 0.81\nloss = 0.35\nspeedup = 13.61\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
            "exec_result": {"stdout": "acc: 0.81, speedup: 13.61", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.81, "loss": 0.35, "baseline_time": 13.61, "our_time": 1.0, "speedup": 13.61}},
            "latex": "\\begin{table} SGC Cora Accuracy: \\result{accuracy}, Speedup: \\result{speedup}x \\end{table} As published in \\cite{arXiv:1902.07153}, SGC reduces training time while maintaining 81.0% accuracy.",
            "claims": ["SGC reduces training time while maintaining 81.0% accuracy on Cora."],
            "mock_reward": 0.96
        }
    ]

    pipeline = AgentLabPipeline(log_dir="results/expanded_benchmark", gate2_mode="COMBINED")
    results_summary = []

    for tc in test_cases:
        res = pipeline.run_pipeline(
            run_id=tc["id"],
            code_attempts=[{"code": tc["code"], "exec_result": tc["exec_result"]}],
            paper_draft_attempts=[{"latex": tc["latex"], "claims": tc["claims"]}],
            mock_reward_scores=[tc["mock_reward"]],
            method_name=tc.get("method", "SGC"),
            dataset_name=tc.get("dataset", "Cora")
        )
        results_summary.append({
            "id": tc["id"],
            "name": tc["name"],
            "paper_ref": tc["paper_ref"],
            "reward_score": tc["mock_reward"],
            "status": res["status"],
            "passed": res["status"] == "SUCCESS"
        })

    # Output JSON summary artifact
    os.makedirs("results/expanded_benchmark", exist_ok=True)
    with open("results/expanded_benchmark/summary_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    return results_summary

if __name__ == "__main__":
    results = run_12_test_cases()
    print("\n\n12-TEST-CASE SUITE EXECUTION COMPLETE.")
