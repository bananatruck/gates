"""
Interactive Test Harness for AgentLaboratory + 3-Gate Verification Architecture.
Allows testing custom code, results.json metrics, and LaTeX drafts live through the pipeline.
"""
import sys
import os
from adapters.agentlab import AgentLabPipeline

def test_custom_scenario(scenario_name, code_snippet, exec_result, latex_draft, claims, mock_reward=0.95):
    print(f"\n==========================================================================")
    print(f"  LIVE TEST RUN: {scenario_name}")
    print(f"==========================================================================")
    
    pipeline = AgentLabPipeline(log_dir="results/live_tests", gate2_mode="COMBINED")
    
    code_attempts = [{"code": code_snippet, "exec_result": exec_result}]
    paper_drafts = [{"latex": latex_draft, "claims": claims}]
    mock_rewards = [mock_reward]
    
    result = pipeline.run_pipeline(
        run_id=scenario_name.lower().replace(" ", "_"),
        code_attempts=code_attempts,
        paper_draft_attempts=paper_drafts,
        mock_reward_scores=mock_rewards
    )
    
    print("\n--- FINAL SCENARIO SUMMARY ---")
    print(f"Scenario: {scenario_name}")
    print(f"Status:   {result['status']}")
    if result['status'] == "SUCCESS":
        print(f"Rendered Paper Sample: {result.get('final_paper')}")
    print("==========================================================================\n")
    return result

if __name__ == "__main__":
    print("Executing Live Comprehensive Test Suite Across 4 Custom Research Scenarios...\n")
    
    # Test Scenario A: Code Crashed with NameError (Reward model = 0.98)
    test_custom_scenario(
        scenario_name="Scenario A - Crashed Code (NameError)",
        code_snippet="hidden_dim = 64\noutput = hidden_dim * 2",
        exec_result={"stdout": "", "stderr": "NameError: name 'hidden_dim' is not defined", "exit_code": 1},
        latex_draft="\\begin{table} Accuracy: 0.95 \\end{table}",
        claims=["Model trained successfully."],
        mock_reward=0.98
    )

    # Test Scenario B: Hardcoded Metric Literal in Python AST
    test_custom_scenario(
        scenario_name="Scenario B - Hardcoded Metric Literal",
        code_snippet="def record_result(k, v): pass\nrecord_result('accuracy', 0.816)",
        exec_result={"stdout": "done", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.816, "loss": 0.3}},
        latex_draft="\\begin{table} Accuracy: \\result{accuracy} \\end{table}",
        claims=["Valid computed metric."],
        mock_reward=0.90
    )

    # Test Scenario C: Gate 2 Reference Interval Violation (SGC on Cora accuracy = 0.40)
    test_custom_scenario(
        scenario_name="Scenario C - Gate 2 Reference Interval Violation",
        code_snippet="acc = 0.40\nloss = 0.6\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
        exec_result={"stdout": "accuracy: 0.40", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.40, "loss": 0.6}},
        latex_draft="\\begin{table} Accuracy: \\result{accuracy} \\end{table} Ref: \\cite{arXiv:1902.07153}",
        claims=["SGC accuracy on Cora."],
        mock_reward=0.88
    )

    # Test Scenario D: Gate 3 Fake Citation (arXiv ID not in scaffold registry)
    test_custom_scenario(
        scenario_name="Scenario D - Gate 3 Fake Citation",
        code_snippet="acc = 0.81\nloss = 0.35\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
        exec_result={"stdout": "accuracy: 0.81", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.81, "loss": 0.35}},
        latex_draft="\\begin{table} Accuracy: \\result{accuracy} \\end{table} Ref: \\cite{arXiv:9999.99999}",
        claims=["Uses novel method."],
        mock_reward=0.92
    )

    # Test Scenario E: Fully Valid Grounded Research Run
    test_custom_scenario(
        scenario_name="Scenario E - Fully Valid Grounded Research Run",
        code_snippet="acc = 0.81\nloss = 0.35\ndef record_result(k, v): pass\nrecord_result('accuracy', acc)\nrecord_result('loss', loss)",
        exec_result={"stdout": "accuracy: 0.81", "stderr": "", "exit_code": 0, "results_json": {"accuracy": 0.81, "loss": 0.35}},
        latex_draft="\\begin{table} Accuracy: \\result{accuracy} \\end{table} As shown in \\cite{arXiv:1902.07153}, SGC is efficient.",
        claims=["SGC achieves efficient graph aggregation."],
        mock_reward=0.95
    )
