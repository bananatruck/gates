import json
import os
import time
import hashlib
from typing import Dict, Any, List, Optional
from gates.gate1 import Gate1Evaluator, Gate1Verdict
from gates.gate2 import Gate2Evaluator, Gate2Verdict
from gates.gate3 import Gate3Evaluator, Gate3Verdict

class AgentLabPipeline:
    """
    Adapter wiring the AgentLaboratory pipeline with the 3 Verification Gates.
    Pipeline: LIT -> PLAN -> DATA -> CODE -> EXEC -> GATE 1 -> REWARD -> GATE 2 -> INTERP -> WRITE -> GATE 3 -> PDF
    Includes 3-rewrite turn budget for feedback loops.
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
        dataset_name: str = "Cora"
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline for a given run with up to 3 rewrite turns.
        """
        print(f"\n=================== RUNNING AGENTLAB PIPELINE: {run_id} ===================")
        
        # 1. CODE & EXECUTION PHASE with GATE 1
        gate1_passed = False
        final_exec_result = None
        final_code_str = ""
        attempt_count = 0
        max_rewrites = 3

        for i, attempt in enumerate(code_attempts[:max_rewrites]):
            attempt_count += 1
            code_str = attempt["code"]
            exec_res = attempt["exec_result"]
            reward_score = mock_reward_scores[i] if mock_reward_scores and i < len(mock_reward_scores) else 0.5

            code_sha256 = hashlib.sha256(code_str.encode()).hexdigest()
            v1 = self.gate1.evaluate(code_str, exec_res)

            # Log divergence ledger
            divergence_entry = {
                "run_id": run_id,
                "attempt": attempt_count,
                "gate1_passed": v1.passed,
                "failed_checks": [c.check_id for c in v1.fail_checks],
                "reward_score": reward_score,
                "code_sha256": code_sha256,
                "exit_code": exec_res.get("exit_code", -1),
                "stdout_bytes": len(exec_res.get("stdout", ""))
            }
            self.log_divergence(divergence_entry)

            print(f"[Attempt {attempt_count}] Reward Score: {reward_score:.2f} | Gate 1 Verdict: {'PASS' if v1.passed else 'FAIL'}")
            if not v1.passed:
                print(f"   Failed Checks: {[c.check_id for c in v1.fail_checks]}")

            if v1.passed:
                gate1_passed = True
                final_exec_result = exec_res
                final_code_str = code_str
                break
            else:
                print(f"   -> Feedback sent to ML Engineer (Rewrite {attempt_count}/{max_rewrites})")

        if not gate1_passed:
            print("❌ GATE 1 EXHAUSTED: GateFailure — no paper emitted.")
            return {"status": "GateFailure_Gate1", "run_id": run_id, "divergence_logged": True}

        # 2. GATE 2: SOURCE <-> RESULT COHERENCE
        metrics = final_exec_result.get("results_json", {})
        v2 = self.gate2.evaluate(metrics, method_name=method_name, dataset_name=dataset_name, mode=self.gate2_mode)
        print(f"[Gate 2 ({self.gate2_mode})] Verdict: {'PASS' if v2.passed else 'FAIL'}")
        if not v2.passed:
            print(f"   Failed Checks: {[c.check_id for c in v2.fail_checks]}")
            return {"status": "GateFailure_Gate2", "run_id": run_id, "gate2_fails": [c.check_id for c in v2.fail_checks]}

        # 3. INTERPRETATION & WRITE PHASE with GATE 3
        value_registry = metrics
        retrieval_registry = {"arXiv:1902.07153", "arXiv:2001.00001"}
        available_figures = {"figure_1.png"}
        gate3 = Gate3Evaluator(value_registry, retrieval_registry, available_figures)

        gate3_passed = False
        final_pdf_content = ""
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
                print(f"   Failed Checks: {[c.check_id for c in v3.fail_checks]}")
                print(f"   -> Feedback sent to Report Writer (Rewrite {j+1}/{max_rewrites})")

        if not gate3_passed:
            print("❌ GATE 3 EXHAUSTED: GateFailure — no paper emitted.")
            return {"status": "GateFailure_Gate3", "run_id": run_id}

        print("✅ PIPELINE SUCCESS: Grounded PDF manuscript generated!")
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "final_paper": final_pdf_content[:300] + "...",
            "divergence_logged": True
        }
