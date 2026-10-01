"""Released papers laid out as run folders so rig.judge re-scores them (D70 item 3, D80)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from rig import judge, stage_released

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "mlrbench"
PRO = "deepseek-v4-pro"
FREE = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"


def released(
    root: Path,
    task="iclr2025_scsl",
    pdf=True,
    code=True,
    *,
    experiments_dir: bool | None = None,
) -> Path:
    folder = root / task
    want_experiments = experiments_dir if experiments_dir is not None else True
    if want_experiments:
        (folder / "experiments").mkdir(parents=True)
    else:
        folder.mkdir(parents=True, exist_ok=True)
    if pdf:
        (folder / f"{task}.pdf").write_bytes(b"%PDF-1.4 fake")
    if code:
        (folder / "experiments" / "best_solution_1.py").write_text("print('trained')\n")
        (folder / "experiments" / "idea.md").write_text("an idea\n")
    return folder


def fake_extract(pdf: Path) -> str:
    return f"\\section{{Results}} text of {pdf.name}\n"


def stage(tmp_path, papers):
    return stage_released.stage_all(
        papers, tasks=FIXTURE / "tasks", out=tmp_path / "out",
        system="AI Scientist v2", model="o4-mini", extract=fake_extract, extractor="fake 1.0",
    )


def test_a_released_paper_becomes_a_run_folder_the_judge_reads(tmp_path):
    papers = tmp_path / "papers"
    released(papers)
    staged, skipped = stage(tmp_path, papers)
    assert skipped == []
    [folder] = staged
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["task"] == "iclr2025_scsl" and manifest["model"] == "o4-mini"
    assert manifest["phase"] == "rescore" and manifest["paper_present"] is True
    assert manifest["task_file_sha256"] == hashlib.sha256(
        (FIXTURE / "tasks" / "iclr2025_scsl.md").read_bytes()).hexdigest()
    assert manifest["source_pdf_sha256"] == hashlib.sha256(b"%PDF-1.4 fake").hexdigest()
    assert manifest["text_extractor"] == "fake 1.0"
    assert (folder / "paper" / "report.txt").read_text().startswith("\\section{Results}")
    assert (folder / "paper" / "source.pdf").read_bytes() == b"%PDF-1.4 fake"
    assert manifest["code_files"] == ["best_solution_1.py"]
    assert [p.name for p in (folder / "code").iterdir()] == ["best_solution_1.py"]


def test_the_judge_scores_a_staged_paper(tmp_path, monkeypatch):
    monkeypatch.setenv(judge.USAGE_ENV, "unset")
    papers = tmp_path / "papers"
    released(papers)
    [folder], _ = stage(tmp_path, papers)

    def model(prompt, system):
        body = ({"has_hallucination": False, "hallucinations": [], "overall_assessment": "a",
                 "confidence": 4} if "identifying hallucinations" in prompt else
                {k: {"score": 5, "justification": "j"} for k in
                 ("Clarity", "Novelty", "Soundness", "Significance")}
                | {"Overall": {"score": 5, "strengths": [], "weaknesses": []}, "Confidence": 4})
        return "```json\n" + json.dumps(body) + "\n```"

    outcome = judge.judge_run(folder, judges=[PRO, FREE], models={PRO: model, FREE: model},
                              reviews=judge.load_judges(FIXTURE), tasks=FIXTURE / "tasks")
    assert outcome.startswith("scored 5.00")


def test_a_task_without_its_text_or_pdf_is_named_not_staged(tmp_path):
    papers = tmp_path / "papers"
    released(papers, task="not_a_task")
    released(papers, task="iclr2025_scsl", pdf=False)
    staged, skipped = stage(tmp_path, papers)
    assert staged == []
    assert any("not_a_task" in s and "task text" in s for s in skipped)
    assert any("iclr2025_scsl" in s and "pdf" in s for s in skipped)


def test_a_staged_folder_is_never_overwritten(tmp_path):
    papers = tmp_path / "papers"
    released(papers)
    stage(tmp_path, papers)
    staged, skipped = stage(tmp_path, papers)
    assert staged == [] and any("already staged" in s for s in skipped)


def test_a_released_paper_with_no_code_says_so(tmp_path):
    papers = tmp_path / "papers"
    released(papers, code=False, experiments_dir=False)
    assert not (papers / "iclr2025_scsl" / "experiments").exists()
    staged, skipped = stage(tmp_path, papers)
    assert skipped == []
    [folder] = staged
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["code_files"] == []
    assert not (folder / "code").exists()


def test_an_empty_experiments_folder_stages_with_no_code_files(tmp_path):
    papers = tmp_path / "papers"
    released(papers, code=False)
    assert (papers / "iclr2025_scsl" / "experiments").is_dir()
    staged, skipped = stage(tmp_path, papers)
    assert skipped == []
    [folder] = staged
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["code_files"] == []
    assert not (folder / "code").exists()


def test_an_empty_extraction_is_named_not_staged(tmp_path):
    papers = tmp_path / "papers"
    released(papers)
    for empty in ("", "\f\n", "   \t\n"):
        out = tmp_path / f"out-{empty!r}"
        staged, skipped = stage_released.stage_all(
            papers, tasks=FIXTURE / "tasks", out=out,
            system="AI Scientist v2", model="o4-mini",
            extract=lambda p, text=empty: text, extractor="fake 1.0",
        )
        assert staged == []
        assert any("iclr2025_scsl" in s and "empty" in s.lower() for s in skipped)
        assert not (out / "MLR-Bench" / "iclr2025_scsl").exists()


def test_a_failed_extraction_leaves_nothing_behind(tmp_path):
    papers = tmp_path / "papers"
    released(papers)

    def boom(_pdf):
        raise RuntimeError("extract failed")

    staged, skipped = stage_released.stage_all(
        papers, tasks=FIXTURE / "tasks", out=tmp_path / "out",
        system="AI Scientist v2", model="o4-mini", extract=boom, extractor="fake 1.0",
    )
    assert staged == []
    assert any("iclr2025_scsl" in s for s in skipped)
    assert not (tmp_path / "out" / "MLR-Bench" / "iclr2025_scsl").exists()

    staged2, skipped2 = stage(tmp_path, papers)
    assert skipped2 == []
    assert len(staged2) == 1


def test_two_systems_never_share_a_folder(tmp_path):
    papers = tmp_path / "papers"
    released(papers)
    staged1, _ = stage_released.stage_all(
        papers, tasks=FIXTURE / "tasks", out=tmp_path / "out",
        system="AI Scientist v2", model="o4-mini", extract=fake_extract, extractor="fake 1.0",
    )
    assert len(staged1) == 1
    staged2, skipped2 = stage_released.stage_all(
        papers, tasks=FIXTURE / "tasks", out=tmp_path / "out",
        system="AI-Scientist v2", model="o4-mini", extract=fake_extract, extractor="fake 1.0",
    )
    assert staged2 == []
    clash_notes = [s for s in skipped2 if "clash" in s.lower() or "slug" in s.lower()]
    assert len(clash_notes) == 1
    note = clash_notes[0]
    assert "AI Scientist v2" in note and "AI-Scientist v2" in note

    staged3, skipped3 = stage_released.stage_all(
        papers, tasks=FIXTURE / "tasks", out=tmp_path / "out2",
        system="科学家", model="o4-mini", extract=fake_extract, extractor="fake 1.0",
    )
    assert staged3 == []
    assert any("科学家" in s and ("slug" in s.lower() or "clash" in s.lower() or "empty" in s.lower())
               for s in skipped3)


def test_nested_experiment_scripts_are_copied(tmp_path):
    papers = tmp_path / "papers"
    released(papers)
    (papers / "iclr2025_scsl" / "experiments" / "sub").mkdir(parents=True)
    (papers / "iclr2025_scsl" / "experiments" / "sub" / "train.py").write_text("print(1)\n")
    [folder], _ = stage(tmp_path, papers)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert "sub/train.py" in manifest["code_files"]
    assert (folder / "code" / "sub" / "train.py").is_file()


def test_collect_never_puts_a_rescored_paper_in_the_level_table(tmp_path):
    spec = importlib.util.spec_from_file_location("collect", REPO / "paper" / "collect.py")
    collect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect)
    papers = tmp_path / "papers"
    released(papers)
    [folder], _ = stage(tmp_path, papers)
    (folder / "metrics.json").write_text(json.dumps({"integrity_event": False, "task_score": 5}))
    runs, skipped = collect.load_runs(tmp_path / "out")
    assert runs == [] and any("re-scored" in s for s in skipped)
    assert collect.load_crash_runs(tmp_path / "out") == []

