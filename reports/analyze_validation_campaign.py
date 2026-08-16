#!/usr/bin/env python3
"""Summarize the controlled full-workflow Gate 1 validation campaign.

This analyzer is deliberately artifact-only: it makes no model calls and does
not need an API credential.  It compares Gate 1's complete evidence channel
with Agent Laboratory's legacy first-1,000-character view, inventories every
check observed in the Gate reports, and audits both generated papers against
the final execution artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rig.paper_audit import audit


EXPECTED_KEYS = (
    "eff.acc_at_25",
    "eff.acc_at_100",
    "eff.efficiency_ratio",
    "eff.train_s",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _values(path: Path) -> dict[str, float]:
    data = _read_json(path)
    entries = data.get("values") or data.get("metrics") or {}
    result: dict[str, float] = {}
    for key, entry in entries.items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = float(value)
    return result


def _value_visible(text: str, key: str, value: float) -> bool:
    """Conservative visibility test requiring both a key and its value."""
    if key not in text:
        return False
    forms = {
        str(value),
        f"{value:.4f}",
        f"{value:.6f}",
        f"{value * 100:.2f}",
    }
    return any(form in text for form in forms)


def _attempt_summary(directory: Path, *, gated: bool) -> dict[str, Any]:
    stdout = (directory / "stdout.txt").read_text(errors="replace")
    stderr = (directory / "stderr.txt").read_text(errors="replace")
    values_path = directory / ("registry.json" if gated else "results.json")
    values = _values(values_path)
    legacy = stdout[:1000]
    core = {key: values[key] for key in EXPECTED_KEYS if key in values}
    result: dict[str, Any] = {
        "attempt": directory.name,
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "recorded_key_count": len(values),
        "extra_key_count": len(set(values) - set(EXPECTED_KEYS)),
        "required_keys_recorded": sorted(core),
        "required_keys_visible_in_legacy_view": sorted(
            key for key, value in core.items()
            if _value_visible(legacy, key, value)
        ),
        "required_keys_visible_in_full_stdout": sorted(
            key for key, value in core.items()
            if _value_visible(stdout, key, value)
        ),
        "values": core,
    }
    if gated:
        report = _read_json(directory / "gate1_report.json")
        result.update({
            "verdict": report["verdict"],
            "failed_checks": [
                check["id"] for check in report["checks"]
                if check["severity"] == "FAIL" and not check["passed"]
            ],
            "warning_checks": [
                check["id"] for check in report["checks"]
                if check["severity"] == "WARN" and not check["passed"]
            ],
            "checks": report["checks"],
            "citable": _read_json(directory / "registry.json").get("citable"),
        })
    return result


def _paper_summary(arm_dir: Path, final_attempt: Path, *, gated: bool) -> dict:
    paper = arm_dir / "research_dir" / "report.txt"
    code = arm_dir / "research_dir" / "src" / "run_experiments.py"
    stdout = (final_attempt / "stdout.txt").read_text(errors="replace")
    registry = final_attempt / "registry.json" if gated else None
    result = audit(
        paper,
        registry_path=registry,
        code_output=stdout,
        code_text=code.read_text(errors="replace") if code.exists() else "",
    )
    return result.to_dict()


def _workflow_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    raw_scores = re.findall(r"^\$\$\$ Score: (.+)$", text, flags=re.MULTILINE)
    scores: list[float] = []
    for raw in raw_scores:
        try:
            scores.append(float(raw.strip()))
        except ValueError:
            pass
    return {
        "path": str(path.resolve()),
        "bytes": len(text.encode("utf-8")),
        "numeric_scores": scores,
        "score_none_count": sum(raw.strip() == "None" for raw in raw_scores),
        "command_failure_count": len(re.findall(
            r"Code (?:editing|replacement) FAILED", text
        )),
        "traceback_count": text.count("Traceback (most recent call last):"),
        "completed_subtasks": re.findall(
            r"^Subtask '([^']+)' completed", text, flags=re.MULTILINE
        ),
    }


def analyze(root: Path, ungated_root: Path | None = None) -> dict[str, Any]:
    ungated_root = ungated_root or root
    gated_dirs = sorted(
        (root / "gated" / "research_dir" / "gate_artifacts" / "gate1").glob(
            "attempt_*"
        )
    )
    ungated_dirs = sorted(
        (ungated_root / "ungated" / "research_dir" / "gate_artifacts").glob(
            "ungated_*"
        )
    )
    if not gated_dirs or not ungated_dirs:
        raise RuntimeError("both campaign arms must contain execution artifacts")

    gated = [_attempt_summary(path, gated=True) for path in gated_dirs]
    ungated = [_attempt_summary(path, gated=False) for path in ungated_dirs]
    check_inventory: dict[str, dict[str, Any]] = {}
    for attempt in gated:
        for check in attempt["checks"]:
            entry = check_inventory.setdefault(check["id"], {
                "severity": check["severity"],
                "observed": 0,
                "passed": 0,
                "failed": 0,
            })
            entry["observed"] += 1
            entry["passed" if check["passed"] else "failed"] += 1

    final_gated = gated_dirs[-1]
    final_ungated = ungated_dirs[-1]
    gated_config = root / "config.snapshot.yaml"
    ungated_config = ungated_root / "config.snapshot.yaml"
    manifest = root / "manifest.json"
    ungated_manifest = ungated_root / "manifest.json"
    return {
        "schema_version": "1.0",
        "campaign_roots": {
            "gated": str(root.resolve()),
            "ungated": str(ungated_root.resolve()),
        },
        "config_sha256": {
            "gated": hashlib.sha256(gated_config.read_bytes()).hexdigest(),
            "ungated": hashlib.sha256(ungated_config.read_bytes()).hexdigest(),
        },
        "manifests": {
            "gated": _read_json(manifest) if manifest.exists() else None,
            "ungated": (
                _read_json(ungated_manifest) if ungated_manifest.exists() else None
            ),
        },
        "expected_keys": list(EXPECTED_KEYS),
        "gated": {
            "workflow": _workflow_summary(root / "gated" / "workflow.log"),
            "attempts": gated,
            "attempt_count": len(gated),
            "pass_count": sum(item["verdict"] == "PASS" for item in gated),
            "blocking_failure_count": sum(bool(item["failed_checks"]) for item in gated),
            "warning_counts": dict(Counter(
                warning for item in gated for warning in item["warning_checks"]
            )),
            "all_required_recorded_count": sum(
                len(item["required_keys_recorded"]) == len(EXPECTED_KEYS)
                for item in gated
            ),
            "all_required_visible_in_legacy_view_count": sum(
                len(item["required_keys_visible_in_legacy_view"]) == len(EXPECTED_KEYS)
                for item in gated
            ),
            "all_required_visible_in_full_stdout_count": sum(
                len(item["required_keys_visible_in_full_stdout"]) == len(EXPECTED_KEYS)
                for item in gated
            ),
            "check_inventory": check_inventory,
            "paper_audit": _paper_summary(
                root / "gated", final_gated, gated=True
            ),
        },
        "ungated": {
            "workflow": _workflow_summary(
                ungated_root / "ungated" / "workflow.log"
            ),
            "attempts": ungated,
            "attempt_count": len(ungated),
            "legacy_accept_count": len(ungated),
            "all_required_recorded_count": sum(
                len(item["required_keys_recorded"]) == len(EXPECTED_KEYS)
                for item in ungated
            ),
            "all_required_visible_in_legacy_view_count": sum(
                len(item["required_keys_visible_in_legacy_view"]) == len(EXPECTED_KEYS)
                for item in ungated
            ),
            "all_required_visible_in_full_stdout_count": sum(
                len(item["required_keys_visible_in_full_stdout"]) == len(EXPECTED_KEYS)
                for item in ungated
            ),
            "attempts_with_extra_keys": sum(
                item["extra_key_count"] > 0 for item in ungated
            ),
            "paper_audit": _paper_summary(
                ungated_root / "ungated", final_ungated, gated=False
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument(
        "--ungated-root", type=Path,
        help="Optional retry root containing the completed ungated arm.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.campaign_root, args.ungated_root)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
