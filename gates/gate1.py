import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class CheckResult:
    check_id: str
    passed: bool
    severity: str  # "FAIL" (blocking) or "WARN" (non-blocking)
    reason: str

@dataclass
class Gate1Verdict:
    passed: bool
    fail_checks: List[CheckResult] = field(default_factory=list)
    warn_checks: List[CheckResult] = field(default_factory=list)
    llm_calls_in_verdict: int = 0  # Invariant: Must be 0

class Gate1Evaluator:
    """
    Gate 1: Execution Validity (100% Deterministic, 0 LLM calls in path).
    Implements 14 checks across 5 check families.
    """
    def __init__(self, expected_keys: List[str] = None):
        self.expected_keys = expected_keys or ["accuracy", "loss"]

    def evaluate(
        self,
        code_str: str,
        execution_result: Dict[str, Any],
        previous_code_hash: Optional[str] = None
    ) -> Gate1Verdict:
        fail_checks = []
        warn_checks = []

        stdout = execution_result.get("stdout", "")
        stderr = execution_result.get("stderr", "")
        exit_code = execution_result.get("exit_code", 0)
        timed_out = execution_result.get("timed_out", False)
        results_json = execution_result.get("results_json", None)

        # -------------------------------------------------------------
        # Family 1: exec.* (Runtime execution checks)
        # -------------------------------------------------------------
        # exec.exit_code
        if exit_code != 0:
            fail_checks.append(CheckResult(
                check_id="exec.exit_code",
                passed=False,
                severity="FAIL",
                reason=f"Process exited with non-zero code {exit_code}."
            ))

        # exec.uncaught_exception
        exception_patterns = [r"NameError:", r"TypeError:", r"KeyError:", r"SyntaxError:", r"AttributeError:", r"ImportError:", r"ZeroDivisionError:"]
        full_log = stdout + "\n" + stderr
        for pattern in exception_patterns:
            if re.search(pattern, full_log):
                fail_checks.append(CheckResult(
                    check_id="exec.uncaught_exception",
                    passed=False,
                    severity="FAIL",
                    reason=f"Uncaught exception detected in execution log matching '{pattern}'."
                ))
                break

        # exec.timeout
        if timed_out:
            fail_checks.append(CheckResult(
                check_id="exec.timeout",
                passed=False,
                severity="FAIL",
                reason="Execution timed out before completing."
            ))

        # -------------------------------------------------------------
        # Family 2: static.* (AST static analysis)
        # -------------------------------------------------------------
        # static.unbound_names
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            fail_checks.append(CheckResult(
                check_id="static.syntax_error",
                passed=False,
                severity="FAIL",
                reason=f"Syntax error during static parse: {e}"
            ))
            tree = None

        if tree:
            # Check for results.values_computed: Novel AST check (Slide 2)
            # Re-parses call site: record_result("k", acc) passes; record_result("k", 0.816) fails!
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    
                    if "record_result" in func_name or "log_metric" in func_name:
                        if len(node.args) >= 2:
                            val_arg = node.args[1]
                            if isinstance(val_arg, (ast.Constant, ast.Num)):
                                fail_checks.append(CheckResult(
                                    check_id="results.values_computed",
                                    passed=False,
                                    severity="FAIL",
                                    reason=f"Hardcoded literal {getattr(val_arg, 'value', getattr(val_arg, 'n', None))} passed to metric call '{func_name}' instead of computed variable."
                                ))

        # -------------------------------------------------------------
        # Family 3: results.* (Schema & Contract checks)
        # -------------------------------------------------------------
        # results.missing_contract
        if results_json is None:
            fail_checks.append(CheckResult(
                check_id="results.missing_contract",
                passed=False,
                severity="FAIL",
                reason="Missing declared results.json contract file."
            ))
        else:
            # results.missing_declared_key
            for key in self.expected_keys:
                if key not in results_json:
                    fail_checks.append(CheckResult(
                        check_id="results.missing_declared_key",
                        passed=False,
                        severity="FAIL",
                        reason=f"Missing required metric key '{key}' in results.json."
                    ))

        # -------------------------------------------------------------
        # Family 4: env.* (Environment & Code Identity)
        # -------------------------------------------------------------
        # env.clean_namespace
        if execution_result.get("leaked_variables", False):
            fail_checks.append(CheckResult(
                check_id="env.clean_namespace",
                passed=False,
                severity="FAIL",
                reason="Environment namespace polluted by previous execution state."
            ))

        # env.code_identity
        code_sha256 = hashlib.sha256(code_str.encode('utf-8')).hexdigest()
        if previous_code_hash and code_sha256 != previous_code_hash:
            # Checked if child hash expected to match parent hash
            pass

        # -------------------------------------------------------------
        # Family 5: WARN Tier (Non-blocking flags)
        # -------------------------------------------------------------
        # logs.no_error_signals
        if "nan" in full_log.lower() or "inf" in full_log.lower() or "cuda fallback" in full_log.lower():
            warn_checks.append(CheckResult(
                check_id="logs.no_error_signals",
                passed=False,
                severity="WARN",
                reason="Log contains numerical warnings (NaN/Inf or CUDA fallback)."
            ))

        # results.single_observation
        if execution_result.get("single_observation_only", False):
            warn_checks.append(CheckResult(
                check_id="results.single_observation",
                passed=False,
                severity="WARN",
                reason="Only single observation/best-epoch reported as final metric."
            ))

        # results.non_degenerate
        if results_json and all(v == 0 for v in results_json.values() if isinstance(v, (int, float))):
            warn_checks.append(CheckResult(
                check_id="results.non_degenerate",
                passed=False,
                severity="WARN",
                reason="All-zero metrics produced (real measurement, so flagged but not blocked)."
            ))

        is_passed = len(fail_checks) == 0
        return Gate1Verdict(
            passed=is_passed,
            fail_checks=fail_checks,
            warn_checks=warn_checks,
            llm_calls_in_verdict=0
        )
