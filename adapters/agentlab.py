import json
import os
import time
import hashlib
from typing import Dict, Any, List, Optional, Set
from gates.gate1 import Gate1Evaluator, Gate1Verdict
from gates.gate2 import Gate2Evaluator, Gate2Verdict
from gates.gate3 import Gate3Evaluator, Gate3Verdict

class AgentLabPipeline:
    """
    Adapter wiring the AgentLaboratory pipeline with the 3 Verification Gates
    and an Iterative Gate 2 Feedback-Driven Refinement Loop.
    Pipeline: LIT -> PLAN -> DATA -> CODE -> EXEC -> GATE 1 -> REWARD -> GATE 2 -> REWRITE LOOP -> INTERP -> WRITE -> GATE 3 -> PDF
    """
    def __init__(self, log_dir: str = "results", gate2_mode: str = "COMBINED"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.divergence_log_path = os.path.join(log_dir, "divergence.jsonl")
        
        self.gate1 = Gate1Evaluator()
        self.gate2 = Gate2Evaluator()
        self.gate2_mode = gate2_mode

    def log_divergence(self, record: Dict[str, Any]):
        """Logs counterfactual divergence ledger entry (what reward model said vs what ran)."""
        with open(self.divergence_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def run_pipeline(
        self,
        run_id: str,
        code_attempts: List[Dict[str, Any]],
        paper_draft_attempts: List[Dict[str, Any]],
        mock_reward_scores: List[float] = None,
        method_name: str = "SGC",
        dataset_name: str = "Cora",
        custom_retrieval_registry: Set[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline with automatic Gate 2 Feedback -> Retry/Rewrite Loops.
        """
        print(f"\n=================== RUNNING AGENTLAB PIPELINE: {run_id} ===================")
        
        gate2_passed = False
        final_exec_result = None
        attempt_idx = 0
        gate2_feedback_history = []

        # COMBINED EXPERIMENT & GATE 1 & GATE 2 ITERATIVE REFINEMENT LOOP
        for attempt_idx in range(len(code_attempts)):
            attempt = code_attempts[attempt_idx]
            code_str = attempt["code"]
            exec_res = attempt["exec_result"]
            reward_score = mock_reward_scores[attempt_idx] if mock_reward_scores and attempt_idx < len(mock_reward_scores) else 0.5

            code_sha256 = hashlib.sha256(code_str.encode()).hexdigest()
            v1 = self.gate1.evaluate(code_str, exec_res)

            divergence_entry = {
                "run_id": run_id,
                "attempt": attempt_idx + 1,
                "gate1_passed": v1.passed,
                "failed_checks": [c.check_id for c in v1.fail_checks],
                "reward_score": reward_score,
                "code_sha256": code_sha256,
                "exit_code": exec_res.get("exit_code", -1),
                "stdout_bytes": len(exec_res.get("stdout", ""))
            }
            self.log_divergence(divergence_entry)

            print(f"[Attempt {attempt_idx+1}] Reward Score: {reward_score:.2f} | Gate 1 Verdict: {'PASS' if v1.passed else 'FAIL'}")
            if not v1.passed:
                print(f"   Failed Checks (Gate 1): {[c.check_id for c in v1.fail_checks]}")
                print(f"   -> Feedback sent to ML Engineer (Rewrite Code Attempt {attempt_idx+1})")
                continue

            metrics = exec_res.get("results_json", {})
            v2 = self.gate2.evaluate(metrics, method_name=method_name, dataset_name=dataset_name, mode=self.gate2_mode)
            
            print(f"[Attempt {attempt_idx+1}] Gate 2 ({self.gate2_mode}) Verdict: {'PASS' if v2.passed else 'FAIL'}")
            
            if v2.passed:
                gate2_passed = True
                final_exec_result = exec_res
                print("   ✅ Gate 2 PASSED: Proceeding to Interpretation & Writing Phase!")
                break
            else:
                feedback_msg = [f"Check '{c.check_id}' failed: {c.reason}" for c in v2.fail_checks]
                gate2_feedback_history.append(feedback_msg)
                
                print(f"   ⚠️ Gate 2 FAILED. Generating Feedback Report for ML Engineer:")
                for fb in feedback_msg:
                    print(f"      • {fb}")
                print(f"   -> Resending feedback into loop (Triggering Code Rewrite / Retry Attempt {attempt_idx+2})...")

        if not gate2_passed:
            print("❌ GATE 2 RETRIES EXHAUSTED: GateFailure — no paper emitted.")
            return {"status": "GateFailure_Gate2", "run_id": run_id, "feedback_history": gate2_feedback_history}

        # 3. INTERPRETATION & WRITE PHASE with GATE 3
        value_registry = final_exec_result.get("results_json", {})
        retrieval_registry = custom_retrieval_registry or {"arXiv:1902.07153", "arXiv:2001.00001", "arXiv:1707.06347", "arXiv:2005.12729"}
        available_figures = {"figure_1.png"}
        gate3 = Gate3Evaluator(value_registry, retrieval_registry, available_figures)

        gate3_passed = False
        final_pdf_content = ""
        max_rewrites = 3

        for j, paper_attempt in enumerate(paper_draft_attempts[:max_rewrites]):
            latex_str = paper_attempt["latex"]
            claims = paper_attempt.get("claims", [])
            v3 = gate3.evaluate(latex_str, claims)

            print(f"[Paper Attempt {j+1}] Gate 3 Verdict: {'PASS' if v3.passed else 'FAIL'}")
            if v3.passed:
                gate3_passed = True
                final_pdf_content = v3.rendered_latex
                break
            else:
                print(f"   Failed Checks (Gate 3): {[c.check_id for c in v3.fail_checks]}")
                print(f"   -> Feedback sent to Report Writer (Rewrite {j+1}/{max_rewrites})")

        if not gate3_passed:
            print("❌ GATE 3 EXHAUSTED: GateFailure — no paper emitted.")
            return {"status": "GateFailure_Gate3", "run_id": run_id}

        print("✅ PIPELINE SUCCESS: Grounded PDF manuscript generated!")
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "final_paper": final_pdf_content,
            "divergence_logged": True
        }
