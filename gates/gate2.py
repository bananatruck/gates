import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class CheckResult:
    check_id: str
    passed: bool
    severity: str  # "FAIL" (blocking) or "WARN" (non-blocking)
    reason: str

@dataclass
class Gate2Verdict:
    passed: bool
    fail_checks: List[CheckResult] = field(default_factory=list)
    warn_checks: List[CheckResult] = field(default_factory=list)
    llm_calls_in_verdict: int = 0
    option_mode: str = "COMBINED"  # "OPTION_A", "OPTION_B", "OPTION_C", or "COMBINED"

class Gate2Evaluator:
    """
    Gate 2: Source <-> Result Coherence
    Supports testing Option A (Deterministic Only), Option B (Reference Interval),
    Option C (Semantic Tier), and Combined (A+B FAIL, C WARN).
    """
    def __init__(self, reference_registry: Dict[str, Dict[str, List[float]]] = None, llm_evaluator=None):
        # reference_registry format: {"method:dataset": {"accuracy": [lower_bound, upper_bound]}}
        self.reference_registry = reference_registry or {
            "SGC:Cora": {"accuracy": [0.75, 0.85], "speedup": [5.0, 20.0]},
            "GCN:Cora": {"accuracy": [0.78, 0.86], "speedup": [1.0, 5.0]}
        }
        self.llm_evaluator = llm_evaluator

    def evaluate(
        self,
        metrics: Dict[str, float],
        method_name: str = "SGC",
        dataset_name: str = "Cora",
        mode: str = "COMBINED"
    ) -> Gate2Verdict:
        fail_checks = []
        warn_checks = []
        llm_calls = 0

        # -------------------------------------------------------------
        # SETTLED TIER / OPTION A: Deterministic Range & Consistency
        # -------------------------------------------------------------
        # check 1: range_valid
        for k, v in metrics.items():
            if any(term in k.lower() for term in ["acc", "precision", "recall", "f1"]):
                if not (0.0 <= v <= 1.0):
                    fail_checks.append(CheckResult(
                        check_id="gate2.range_valid",
                        passed=False,
                        severity="FAIL",
                        reason=f"Metric '{k}' value {v} outside valid range [0, 1]."
                    ))
            if any(term in k.lower() for term in ["time", "duration", "latency"]):
                if v <= 0:
                    fail_checks.append(CheckResult(
                        check_id="gate2.range_valid",
                        passed=False,
                        severity="FAIL",
                        reason=f"Time metric '{k}' value {v} must be strictly > 0."
                    ))

        # check 2: internal_consistency
        if "baseline_time" in metrics and "our_time" in metrics and "speedup" in metrics:
            calc_speedup = metrics["baseline_time"] / max(metrics["our_time"], 1e-9)
            if not math.isclose(metrics["speedup"], calc_speedup, rel_tol=0.05):
                fail_checks.append(CheckResult(
                    check_id="gate2.internal_consistency",
                    passed=False,
                    severity="FAIL",
                    reason=f"Reported speedup {metrics['speedup']} contradicts time ratio baseline/our ({calc_speedup:.2f})."
                ))

        # -------------------------------------------------------------
        # OPTION B: Reference Interval Check (CORE-Bench style)
        # -------------------------------------------------------------
        if mode in ["OPTION_B", "COMBINED"]:
            key = f"{method_name}:{dataset_name}"
            if key in self.reference_registry:
                ref_bounds = self.reference_registry[key]
                for metric_key, (low, high) in ref_bounds.items():
                    if metric_key in metrics:
                        val = metrics[metric_key]
                        if not (low <= val <= high):
                            fail_checks.append(CheckResult(
                                check_id="gate2.reference_interval",
                                passed=False,
                                severity="FAIL",
                                reason=f"Metric '{metric_key}'={val} violates reference interval [{low}, {high}] for {key}."
                            ))
            # Q1 Resolution: If key not in registry (Novel Result), it passes this check (no baseline to violate)

        # -------------------------------------------------------------
        # OPTION C: Semantic Tier (LLM-Assisted Mismatch Classifier)
        # -------------------------------------------------------------
        if mode in ["OPTION_C", "COMBINED"]:
            llm_calls += 1
            if self.llm_evaluator:
                mismatch_detected, reason = self.llm_evaluator(method_name, dataset_name, metrics)
            else:
                # Mock semantic evaluation rule for benchmark simulation:
                # Flag if method is reported with impossibly high baseline setting mismatch
                mismatch_detected = metrics.get("accuracy", 0.0) > 0.98 and method_name == "SGC"
                reason = "LLM flagged potential method/dataset difficulty mismatch: SGC accuracy > 98% on Cora is atypical."

            if mismatch_detected:
                if mode == "OPTION_C":
                    # If evaluating Option C standalone as blocking
                    fail_checks.append(CheckResult(
                        check_id="gate2.semantic_mismatch",
                        passed=False,
                        severity="FAIL",
                        reason=reason
                    ))
                else:
                    # In COMBINED mode: MUST be WARN tier (Slide 4 invariant: semantic tier is not provable)
                    warn_checks.append(CheckResult(
                        check_id="gate2.semantic_mismatch",
                        passed=False,
                        severity="WARN",
                        reason=reason
                    ))

        is_passed = len(fail_checks) == 0
        return Gate2Verdict(
            passed=is_passed,
            fail_checks=fail_checks,
            warn_checks=warn_checks,
            llm_calls_in_verdict=llm_calls,
            option_mode=mode
        )
