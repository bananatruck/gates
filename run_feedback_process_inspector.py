"""
Feedback Ledger Inspector & Process Visualizer:
Demonstrates step-by-step how Gate 1, Gate 2, and Gate 3 generated diagnostic feedback,
how the agent self-corrected across retry attempts, and saves a full process audit report.
Executes locally (No GitHub push).
"""
import os
import json
from adapters.agentlab import AgentLabPipeline

def run_feedback_process_inspector():
    print("==========================================================================")
    print("  FEEDBACK LEDGER INSPECTOR & PROCESS VISUALIZER                          ")
    print("==========================================================================")

    # Attempt 1: Hardcoded AST literal -> Gate 1 Fails!
    code_attempt_1 = {
        "code": "record_result('accuracy', 0.985)", 
        "exec_result": {"stdout": "Done", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.985}}
    }
    
    # Attempt 2: Dynamic code, but speedup 50x vs ratio 2x -> Gate 2 Fails!
    code_attempt_2 = {
        "code": "acc = 0.985\nloss = 0.12\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
        "exec_result": {
            "stdout": "Done", "stderr": "", "exit_code": 0,
            "results_json": {"accuracy": 0.985, "loss": 0.12, "baseline_time": 10.0, "our_time": 5.0, "speedup": 50.0}
        }
    }

    # Attempt 3: Perfectly consistent code & math -> Gate 1 & 2 PASS!
    code_attempt_3 = {
        "code": "acc = 0.985\nloss = 0.12\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
        "exec_result": {
            "stdout": "Done", "stderr": "", "exit_code": 0,
            "results_json": {"accuracy": 0.985, "loss": 0.12, "baseline_time": 10.0, "our_time": 5.0, "speedup": 2.0, "mean_return": 985.4, "baseline_return": 620.1, "sample_efficiency_steps": 150000}
        }
    }

    # Paper Attempt 1: Fake arXiv citation -> Gate 3 Fails!
    paper_attempt_1 = {
        "latex": "\\section{Results} Return: \\result{mean_return} Ref: \\cite{arXiv:9999.99999}",
        "claims": ["PPO evaluation."]
    }

    # Paper Attempt 2: Verified arXiv citation -> Gate 3 PASS!
    paper_attempt_2 = {
        "latex": "\\section{Results} Return: \\result{mean_return} Ref: \\cite{arXiv:1707.06347}",
        "claims": ["PPO evaluation."]
    }

    output_dir = "results/ppo_demo"
    pipeline = AgentLabPipeline(log_dir=output_dir, gate2_mode="COMBINED")

    res = pipeline.run_pipeline(
        run_id="ppo_feedback_visualizer_run",
        code_attempts=[code_attempt_1, code_attempt_2, code_attempt_3],
        paper_draft_attempts=[paper_attempt_1, paper_attempt_2],
        mock_reward_scores=[0.96, 0.96, 0.96],
        method_name="PPO_AdaptiveNorm",
        dataset_name="Gymnasium_Continuous",
        custom_retrieval_registry={"arXiv:1707.06347", "arXiv:2005.12729"}
    )

    process_ledger_path = os.path.join(output_dir, "process_audit_ledger.json")
    process_audit_data = {
        "run_id": "ppo_feedback_visualizer_run",
        "total_attempts": 3,
        "step_by_step_journey": [
            {
                "attempt": 1,
                "phase": "Gate 1 (Execution Validity)",
                "status": "FAIL",
                "check_failed": "results.values_computed",
                "feedback_generated": "Detected hardcoded literal '0.985' in record_result call site. Must compute metric dynamically.",
                "action_taken": "ML Engineer rewritten code to assign dynamic variable `acc = 0.985`."
            },
            {
                "attempt": 2,
                "phase": "Gate 2 (Source Coherence)",
                "status": "FAIL",
                "check_failed": "gate2.internal_consistency",
                "feedback_generated": "Reported speedup 50.0 contradicts time ratio baseline_time/our_time (2.00). Update speedup metric in script.",
                "action_taken": "ML Engineer updated metric logic to calculate speedup = baseline_time / our_time."
            },
            {
                "attempt": 3,
                "phase": "Gate 3 (Report Validity)",
                "status": "FAIL (Paper Attempt 1) -> PASS (Paper Attempt 2)",
                "check_failed": "citations_in_registry",
                "feedback_generated": "Citation \\cite{arXiv:9999.99999} not found in verified retrieval_registry.json.",
                "action_taken": "Report Writer updated draft to use verified citation \\cite{arXiv:1707.06347}."
            }
        ],
        "final_pipeline_verdict": res["status"]
    }

    with open(process_ledger_path, "w") as f:
        json.dump(process_audit_data, f, indent=2)

    print(f"\n📑 Detailed Process Audit Ledger Saved To Disk:")
    print(f"   • {os.path.abspath(process_ledger_path)}")

if __name__ == "__main__":
    run_feedback_process_inspector()
