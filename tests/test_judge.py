"""The judge runs MLR-Bench's own prompts, and a person settles every flag (D54, D62).

``tests/fixtures/mlrbench`` is a byte-for-byte copy of the pinned MLR-Bench
files the run kit carries, with their checksums. The judges here are fakes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rig import judge, review_flags

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mlrbench"
TASK = (FIXTURE / "tasks" / "iclr2025_scsl.md").read_bytes()
PRO = "deepseek-v4-pro"
FREE = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"


def overall(score):
    return {k: {"score": score, "justification": "j"} for k in
            ("Clarity", "Novelty", "Soundness", "Significance")} | {
        "Overall": {"score": score, "strengths": [], "weaknesses": []}, "Confidence": 4}


def hallucination(faked):
    found = [{"type": "Faked Experimental Results", "description": "numbers typed",
              "evidence": "accuracy 99.9%"}] if faked else []
    return {"has_hallucination": faked, "hallucinations": found,
            "overall_assessment": "a", "confidence": 4}


def fake(score, faked, prompts=None):
    def model(prompt, system):
        if prompts is not None:
            prompts.append(prompt)
        body = hallucination(faked) if "identifying hallucinations" in prompt else overall(score)
        return "```json\n" + json.dumps(body) + "\n```"
    return model


def make_run(root, *, model="deepseek-flash", paper=True, level=0):
    import hashlib

    folder = root / "MLR-Bench" / "iclr2025_scsl" / "Agent-Lab" / f"L{level}" / "seed0"
    (folder / "paper").mkdir(parents=True)
    (folder / "code").mkdir()
    (folder / "code" / "run_experiments.py").write_text("print('trained')\n")
    if paper:
        (folder / "paper" / "report.txt").write_text("\\section{Results} accuracy 99.9\\%\n")
    else:
        (folder / "paper" / "reason.txt").write_text("No paper was emitted. Status: no_paper:gate.")
    (folder / "manifest.json").write_text(json.dumps({
        "task": "iclr2025_scsl", "model": model, "level": level, "status": "completed",
        "task_file_sha256": hashlib.sha256(TASK).hexdigest(),
    }))
    return folder


@pytest.fixture
def judged(monkeypatch):
    monkeypatch.setenv(judge.USAGE_ENV, "unset")
    reviews = judge.load_judges(FIXTURE)

    def run(folder, models, judges=(PRO, FREE)):
        return judge.judge_run(folder, judges=list(judges), models=models,
                               reviews=reviews, tasks=FIXTURE / "tasks")
    return run


def test_the_prompt_is_mlrbenchs_with_task_paper_and_code(tmp_path, judged):
    folder = make_run(tmp_path)
    prompts = []
    judged(folder, {PRO: fake(5, False, prompts), FREE: fake(7, False)})
    overall_prompt, hallucination_prompt = prompts
    assert "Evaluation Rubric" in overall_prompt and "Soundness (1-10)" in overall_prompt
    assert "identifying hallucinations" in hallucination_prompt
    for prompt in prompts:
        assert TASK.decode() in prompt
        assert "accuracy 99.9" in prompt
        assert "print('trained')" in prompt
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics == {"integrity_event": False, "task_score": 6.0,
                       "judges": [PRO, FREE], "human_reviewed": False}
    meta = json.loads((folder / "judge" / judge.slug(PRO) / "meta.json").read_text())
    assert meta["max_tokens"] == 16384 and meta["temperature"] == 0.0


def test_a_changed_mlrbench_file_is_refused(tmp_path):
    kit = tmp_path / "mlrbench"
    shutil.copytree(FIXTURE, kit)
    path = kit / "judge" / "overall_review.py"
    path.write_text(path.read_text().replace("Avoid giving high scores", "Give high scores"))
    with pytest.raises(judge.KitError, match="overall_review.py"):
        judge.load_judges(kit)


def test_a_judge_never_scores_its_own_models_run(tmp_path, judged):
    folder = make_run(tmp_path, model=PRO)
    judged(folder, {PRO: fake(9, False), FREE: fake(4, False)})
    summary = json.loads((folder / "judge" / "summary.json").read_text())
    assert summary["excluded"] == [PRO] and list(summary["judges"]) == [FREE]
    assert json.loads((folder / "metrics.json").read_text())["task_score"] == 4


def test_one_judge_flagging_is_not_a_candidate(tmp_path, judged):
    folder = make_run(tmp_path)
    judged(folder, {PRO: fake(5, True), FREE: fake(5, False)})
    assert json.loads((folder / "metrics.json").read_text())["integrity_event"] is False


def test_both_judges_flagging_waits_for_a_person_whose_verdict_is_final(tmp_path, judged):
    folder = make_run(tmp_path)
    outcome = judged(folder, {PRO: fake(3, True), FREE: fake(5, True)})
    assert "candidate" in outcome and not (folder / "metrics.json").exists()

    answers = iter(["y", "numbers never printed by the code"])
    assert review_flags.main(["--runs", str(tmp_path), "--reviewer", "Kesh"],
                             ask=lambda _: next(answers)) == 0
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["integrity_event"] is True and metrics["human_reviewed"] is True
    assert metrics["task_score"] == 4.0
    review = json.loads((folder / "judge" / "human_review.json").read_text())
    assert review["reviewer"] == "Kesh" and review["confirmed"] is True
    assert review_flags.candidates(tmp_path) == []


def test_a_person_can_reject_a_flag(tmp_path, judged):
    folder = make_run(tmp_path)
    judged(folder, {PRO: fake(3, True), FREE: fake(5, True)})
    answers = iter(["n", ""])
    review_flags.main(["--runs", str(tmp_path), "--reviewer", "Kesh"], ask=lambda _: next(answers))
    assert json.loads((folder / "metrics.json").read_text())["integrity_event"] is False


def test_a_run_without_a_paper_gets_no_score_and_no_judge(tmp_path, judged):
    folder = make_run(tmp_path, paper=False)
    calls = []
    judged(folder, {PRO: fake(5, False, calls), FREE: fake(5, False, calls)})
    assert calls == []
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["task_score"] is None and metrics["integrity_event"] is False
    assert "no_paper:gate" in metrics["no_paper"]


def test_a_judge_that_returns_no_json_leaves_the_run_for_a_retry(tmp_path, judged):
    folder = make_run(tmp_path)
    outcome = judged(folder, {PRO: lambda p, s: "I cannot comply", FREE: fake(5, False)})
    assert outcome.startswith("incomplete") and not (folder / "metrics.json").exists()
    # A retry asks only the judge that failed.
    for name in ("overall.json", "hallucination.json"):
        (folder / "judge" / judge.slug(PRO) / name).unlink()
    free_calls = []
    judged(folder, {PRO: fake(6, False), FREE: fake(5, False, free_calls)})
    assert free_calls == []
    assert json.loads((folder / "metrics.json").read_text())["task_score"] == 5.5


def test_a_run_given_a_different_task_text_is_refused(tmp_path, judged):
    folder = make_run(tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["task_file_sha256"] = "0" * 64
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(judge.KitError, match="not the task text"):
        judged(folder, {PRO: fake(5, False), FREE: fake(5, False)})
