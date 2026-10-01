"""Hidden Pitfalls' released Agent Laboratory outputs staged as run folders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rig import pitfalls_audit
from rig.stage_pitfalls import agentlab_task_for, stage_all

CLAUDE = "agent:claude/claude-opus-5-5"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def release(tmp_path: Path) -> Path:
    """The authors' layout at 2725bda, cut down to one output per shape."""
    root = tmp_path / "AIScientistPitfalls"
    _write(root / "SPR Task" / "DataLeakage" / "SPR.md", "TASK-DATA-LEAKAGE\n")
    _write(root / "SPR Task" / "BenchmarkIssue" / "SPR.md", "TASK-BENCHMARK\n")
    _write(root / "SPR Task" / "MetricMisuse" / "color-first" / "SPR.md", "TASK-COLOR-FIRST\n")
    _write(root / "SPR Task" / "MetricMisuse" / "shape-first" / "SPR.md", "TASK-SHAPE-FIRST\n")
    research = root / "AgentLaboratory" / "generated_research"
    leak = research / "DataLeakage" / "noise-20%-val_test" / "6"
    _write(leak / "report.txt", "PAPER-DL-6\n")
    _write(leak / "readme.md", "readme\n")
    _write(leak / "src" / "run_experiments.py", "print('CODE-DL-6')\n")
    _write(leak / "src" / "experiment_output.log", "LOG-DL-6\n")
    color = research / "MetricMisuse" / "color-flip" / "color first" / "1"
    _write(color / "report.txt", "PAPER-MM-1\n")
    _write(color / "src" / "load_data.py", "print('CODE-MM-1')\n")
    bench = research / "BenchmarkIssue" / "research_dir" / "research_dir_0_lab_1"
    _write(bench / "report.txt", "PAPER-BI-1\n")
    no_paper = research / "MetricMisuse" / "shape-flip" / "shape first" / "3"
    _write(no_paper / "src" / "experiment_output.log", "crashed\n")
    _write(research / "PostHocSelection" / "projects" / "project1.txt", "candidate\n")
    return root


def _stage(release: Path, out: Path):
    return stage_all(
        release / "AgentLaboratory" / "generated_research",
        task_for=agentlab_task_for(release / "SPR Task"),
        out=out, system="Agent Laboratory", model="o3-mini",
    )


def test_each_released_paper_becomes_one_run_folder(release, tmp_path):
    out = tmp_path / "pitfalls-B"
    staged, skipped = _stage(release, out)

    base = out / "Hidden-Pitfalls"
    assert sorted(staged) == sorted([
        base / "BenchmarkIssue" / "Agent-Laboratory" / "released" / "research-dir-research-dir-0-lab-1",
        base / "DataLeakage" / "Agent-Laboratory" / "released" / "noise-20-val-test-6",
        base / "MetricMisuse-color-first" / "Agent-Laboratory" / "released"
        / "color-flip-color-first-1",
    ])
    assert skipped == [
        "MetricMisuse/shape-flip/shape first/3: no report.txt, so no paper to audit",
    ]


def test_a_staged_folder_carries_the_paper_code_log_and_task_hash(release, tmp_path):
    out = tmp_path / "pitfalls-B"
    _stage(release, out)
    folder = out / "Hidden-Pitfalls" / "DataLeakage" / "Agent-Laboratory" / "released" / "noise-20-val-test-6"

    assert (folder / "paper" / "report.txt").read_text() == "PAPER-DL-6\n"
    assert (folder / "code" / "run_experiments.py").read_text() == "print('CODE-DL-6')\n"
    assert not (folder / "code" / "experiment_output.log").exists()
    assert (folder / "agent_log.txt").read_text() == "LOG-DL-6\n"
    assert (out / "tasks" / "DataLeakage.md").read_text() == "TASK-DATA-LEAKAGE\n"

    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest == {
        "benchmark": "Hidden-Pitfalls",
        "task": "DataLeakage",
        "system": "Agent Laboratory",
        "model": "o3-mini",
        "phase": "rescore",
        "status": "released",
        "paper_present": True,
        "task_file_sha256": hashlib.sha256(b"TASK-DATA-LEAKAGE\n").hexdigest(),
        "source_paper": "report.txt",
        "source_paper_sha256": hashlib.sha256(b"PAPER-DL-6\n").hexdigest(),
        "source_format": "text",
        "text_extractor": "source text",
        "source": str(release / "AgentLaboratory" / "generated_research"
                      / "DataLeakage" / "noise-20%-val_test" / "6"),
        "code_files": ["run_experiments.py"],
        "agent_log": "src/experiment_output.log",
    }


def test_metric_misuse_takes_the_task_text_its_folder_names(release, tmp_path):
    out = tmp_path / "pitfalls-B"
    _stage(release, out)
    folder = (out / "Hidden-Pitfalls" / "MetricMisuse-color-first" / "Agent-Laboratory"
              / "released" / "color-flip-color-first-1")

    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["task"] == "MetricMisuse-color-first"
    assert (out / "tasks" / "MetricMisuse-color-first.md").read_text() == "TASK-COLOR-FIRST\n"
    assert manifest["agent_log"] is None
    assert not (folder / "agent_log.txt").exists()


def test_staging_twice_never_overwrites(release, tmp_path):
    out = tmp_path / "pitfalls-B"
    _stage(release, out)
    folder = out / "Hidden-Pitfalls" / "DataLeakage" / "Agent-Laboratory" / "released" / "noise-20-val-test-6"
    (folder / "paper" / "report.txt").write_text("judged copy\n")

    staged, skipped = _stage(release, out)

    assert staged == []
    assert f"DataLeakage/noise-20%-val_test/6: already staged at {folder}" in skipped
    assert (folder / "paper" / "report.txt").read_text() == "judged copy\n"


def test_a_changed_task_text_is_refused_not_mixed(release, tmp_path):
    out = tmp_path / "pitfalls-B"
    _write(out / "tasks" / "DataLeakage.md", "another task\n")

    staged, skipped = _stage(release, out)

    assert all("DataLeakage" not in str(folder) for folder in staged)
    assert (f"DataLeakage/noise-20%-val_test/6: {out / 'tasks' / 'DataLeakage.md'} "
            "already holds a different task text") in skipped


def test_the_auditor_reads_a_staged_folder(release, tmp_path):
    out = tmp_path / "pitfalls-B"
    _stage(release, out)
    prompts: list[str] = []

    def judge(prompt: str, system: str) -> str:
        prompts.append(prompt)
        return '```json\n{"answer": "NO", "evidence": "none"}\n```'

    assert pitfalls_audit.main(
        ["--runs", str(out), "--mlrbench", str(out), "--judge", CLAUDE],
        model_for=lambda name: judge,
    ) == 0

    leak = [p for p in prompts if "PAPER-DL-6" in p]
    assert leak
    assert all("TASK-DATA-LEAKAGE" in p for p in leak)
    assert any("CODE-DL-6" in p for p in leak)
    assert any("LOG-DL-6" in p for p in leak)
    audits = sorted(out.rglob("audit.json"))
    assert len(audits) == 3
