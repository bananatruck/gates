"""The judge runs MLR-Bench's own prompts, and a person settles every flag (D54, D62).

``tests/fixtures/mlrbench`` is a byte-for-byte copy of the pinned MLR-Bench
files the run kit carries, with their checksums. The judges here are fakes.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from rig import judge, review_flags
from rig.live import AgentCLIError, _run_agent, agent_model

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mlrbench"
TASK = (FIXTURE / "tasks" / "iclr2025_scsl.md").read_bytes()
PRO = "deepseek-v4-pro"
FREE = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
REVIEW_TAG = "reviewed by agent for now"


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
                       "judges": [PRO, FREE], "human_reviewed": False,
                       "integrity_event_either": False}
    meta = json.loads((folder / "judge" / judge.slug(PRO) / "meta.json").read_text())
    assert meta["max_tokens"] == 16384 and meta["temperature"] == 0.0


def _fake_agent_clis(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "agent-calls.jsonl"
    script = """#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

name = Path(sys.argv[0]).name
if "--version" in sys.argv:
    print(f"fake {name} 1.2.3")
    raise SystemExit
prompt = sys.stdin.read()
with Path(os.environ["FAKE_AGENT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"name": name, "argv": sys.argv[1:], "stdin": prompt}) + "\\n")
if delay := os.environ.get("FAKE_AGENT_SLEEP"):
    time.sleep(float(delay))
if os.environ.get("FAKE_AGENT_INVALID") == name:
    print("not JSON")
    raise SystemExit
if "identifying hallucinations" in prompt:
    body = {"has_hallucination": False, "hallucinations": [],
            "overall_assessment": "a", "confidence": 4}
else:
    body = {name: {"score": 6, "justification": "j"} for name in
            ("Clarity", "Novelty", "Soundness", "Significance")}
    body.update({"Overall": {"score": 6, "strengths": [], "weaknesses": []},
                 "Confidence": 4})
print("```json")
print(json.dumps(body))
print("```")
"""
    for name in ("claude", "cursor-agent"):
        executable = bin_dir / name
        executable.write_text(script)
        executable.chmod(0o755)
    monkeypatch.setenv("FAKE_AGENT_LOG", str(log))
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    return log


def test_agent_judges_use_stdin_and_tag_every_output(tmp_path, monkeypatch):
    log = _fake_agent_clis(tmp_path, monkeypatch)
    folder = make_run(tmp_path / "runs")

    assert judge.main([
        "--runs", str(tmp_path / "runs"), "--mlrbench", str(FIXTURE),
        "--judge", "agent:claude/sonnet", "--judge", "agent:cursor/gpt-5",
    ]) == 0

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call["argv"] for call in calls] == [
        ["-p", "--model", "sonnet"], ["-p", "--model", "sonnet"],
        ["--trust", "--mode", "ask", "--model", "gpt-5", "-p"],
        ["--trust", "--mode", "ask", "--model", "gpt-5", "-p"],
    ]
    payloads = [call["stdin"] for call in calls]
    assert all(payload.startswith("\n\n") for payload in payloads)
    assert all(TASK.decode() in payload for payload in payloads)
    assert all("accuracy 99.9" in payload for payload in payloads)
    assert all("print('trained')" in payload for payload in payloads)

    summary = json.loads((folder / "judge" / "summary.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    assert summary["review_tag"] == REVIEW_TAG
    assert metrics["review_tag"] == REVIEW_TAG
    for name, command in (
        ("agent:claude/sonnet", ["claude", "-p", "--model", "sonnet"]),
        ("agent:cursor/gpt-5",
         ["cursor-agent", "--trust", "--mode", "ask", "--model", "gpt-5", "-p"]),
    ):
        meta = json.loads((folder / "judge" / judge.slug(name) / "meta.json").read_text())
        assert meta["judge_kind"] == "agent"
        assert meta["command"] == command
        assert meta["cli_version"].startswith("fake ")
        assert meta["review_tag"] == REVIEW_TAG
        assert meta["mlr_bench_deviations"] == {
            "temperature": "the agent CLI does not expose temperature control; MLR-Bench uses 0",
            "max_tokens": "the agent CLI does not expose max-token control; MLR-Bench uses 16384",
        }
        assert "max_tokens" not in meta and "temperature" not in meta


def test_agent_judge_does_not_score_its_own_model_name():
    judges = ["agent:claude/sonnet", "agent:cursor/gpt-5"]
    assert judge.eligible(judges, "sonnet") == ["agent:cursor/gpt-5"]


def test_agent_timeout_kills_a_hung_cli_and_leaves_the_judge_for_retry(
    tmp_path, monkeypatch
):
    log = _fake_agent_clis(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_AGENT_SLEEP", "60")
    folder = make_run(tmp_path / "runs")

    model = agent_model("agent:claude/sonnet", timeout_s=0.2)
    with pytest.raises(AgentCLIError, match="timed out after 0.2 seconds"):
        model("user", "system")
    log.write_text("")

    started = time.monotonic()
    assert judge.main([
        "--runs", str(tmp_path / "runs"), "--mlrbench", str(FIXTURE),
        "--judge", "agent:claude/sonnet", "--agent-timeout", "0.2",
    ]) == 0

    assert time.monotonic() - started < 5
    assert len(log.read_text().splitlines()) == 6
    summary = json.loads((folder / "judge" / "summary.json").read_text())
    assert summary["complete"] is False
    assert not (folder / "metrics.json").exists()


def test_agent_timeout_also_ends_the_clis_own_children(tmp_path):
    # An agent CLI starts helpers of its own. Killing only the CLI would leave
    # them running, still able to call a model, after the judge gave up.
    pid_file = tmp_path / "child.pid"
    cli = tmp_path / "hung-cli"
    cli.write_text(f"#!/bin/sh\nsleep 30 &\necho $! > {pid_file}\nsleep 30\n")
    cli.chmod(0o755)

    started = time.monotonic()
    with pytest.raises(AgentCLIError, match="timed out"):
        _run_agent((str(cli),), prompt="x", timeout_s=0.5)

    assert time.monotonic() - started < 5
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and _alive(child):
        time.sleep(0.05)
    assert not _alive(child)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child is a zombie until init reaps it; a zombie is not running.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return False
    return state != "Z"


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
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["integrity_event"] is False
    # D69's count waits for a person: one judge's flag is neither yes nor no yet.
    assert metrics["integrity_event_either"] is None


def test_no_flag_from_any_judge_settles_both_definitions(tmp_path, judged):
    folder = make_run(tmp_path)
    judged(folder, {PRO: fake(5, False), FREE: fake(5, False)})
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["integrity_event"] is False and metrics["integrity_event_either"] is False


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
    assert "review_tag" not in review and "review_tag" not in metrics
    assert review_flags.candidates(tmp_path) == []


def test_an_agent_reviewer_uses_an_answers_file_without_prompting(tmp_path, judged):
    folder = make_run(tmp_path)
    judged(folder, {PRO: fake(3, True), FREE: fake(5, True)})
    run_name = str(folder.relative_to(tmp_path))
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({
        run_name: {"confirmed": True, "note": "the result is absent from the code"},
    }))

    def no_prompt(_):
        raise AssertionError("the agent review path must not prompt")

    assert review_flags.main([
        "--runs", str(tmp_path), "--reviewer", "agent:claude", "--answers", str(answers),
    ], ask=no_prompt) == 0

    review = json.loads((folder / "judge" / "human_review.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    assert review["reviewer"] == "agent:claude" and review["confirmed"] is True
    assert review["review_tag"] == REVIEW_TAG
    assert metrics["review_tag"] == REVIEW_TAG
    assert metrics["human_reviewed"] is False


def test_agent_reviewers_require_answers_and_people_cannot_use_them(tmp_path):
    answers = tmp_path / "answers.json"
    answers.write_text("{}")
    with pytest.raises(SystemExit):
        review_flags.main(["--runs", str(tmp_path), "--reviewer", "agent:claude"])
    with pytest.raises(SystemExit):
        review_flags.main([
            "--runs", str(tmp_path), "--reviewer", "Kesh", "--answers", str(answers),
        ])


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


def test_a_judge_error_is_retried_while_a_completed_judge_is_left_alone(
    tmp_path, monkeypatch
):
    log = _fake_agent_clis(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_AGENT_INVALID", "claude")
    folder = make_run(tmp_path / "runs")
    argv = [
        "--runs", str(tmp_path / "runs"), "--mlrbench", str(FIXTURE),
        "--judge", "agent:claude/sonnet", "--judge", "agent:cursor/gpt-5",
    ]

    assert judge.main(argv) == 0
    first_calls = [json.loads(line)["name"] for line in log.read_text().splitlines()]
    assert first_calls.count("claude") == 6
    assert first_calls.count("cursor-agent") == 2
    first_summary = json.loads((folder / "judge" / "summary.json").read_text())
    assert first_summary["complete"] is False
    assert not (folder / "metrics.json").exists()

    monkeypatch.delenv("FAKE_AGENT_INVALID")
    assert judge.main(argv) == 0
    calls = [json.loads(line)["name"] for line in log.read_text().splitlines()]
    assert calls.count("claude") == 8
    assert calls.count("cursor-agent") == 2
    summary = json.loads((folder / "judge" / "summary.json").read_text())
    assert summary["complete"] is True
    assert summary["task_score"] == 6
    assert json.loads((folder / "metrics.json").read_text())["task_score"] == 6


def test_a_run_given_a_different_task_text_is_refused(tmp_path, judged):
    folder = make_run(tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["task_file_sha256"] = "0" * 64
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(judge.KitError, match="not the task text"):
        judged(folder, {PRO: fake(5, False), FREE: fake(5, False)})


# --------------------------------------------------------------------------- #
# blind, shuffled review; either judge's pool (D69); agreement (09-29 review, Q2)
# --------------------------------------------------------------------------- #


def _flagged_runs(root, judged, levels=(0, 3)):
    folders = []
    for level in levels:
        folder = make_run(root, level=level)
        judged(folder, {PRO: fake(3, True), FREE: fake(5, True)})
        folders.append(folder)
    return folders


def test_a_blind_review_never_shows_the_level_or_the_run_path(tmp_path, judged, capsys):
    runs = tmp_path / "runs"
    _flagged_runs(runs, judged)
    answers = iter(["s", "s"])
    review_flags.main(["--runs", str(runs), "--reviewer", "Kesh", "--blind",
                       "--stage", str(tmp_path / "stage"), "--seed", "1"], ask=lambda _: next(answers))
    shown = capsys.readouterr().out
    assert "L0" not in shown and "L3" not in shown and str(runs) not in shown
    staged = sorted((tmp_path / "stage").glob("*/paper/report.txt"))
    assert len(staged) == 2


def test_a_blind_order_is_shuffled_by_a_recorded_seed(tmp_path, judged):
    runs = tmp_path / "runs"
    folders = _flagged_runs(runs, judged, levels=(0, 1, 2, 3))
    orders = {seed: review_flags.review_order(review_flags.candidates(runs), seed) for seed in range(6)}
    assert orders[1] == review_flags.review_order(review_flags.candidates(runs), 1)
    assert any(order != folders for order in orders.values())
    answers = iter(["y", "", "s", "s", "s"])
    review_flags.main(["--runs", str(runs), "--reviewer", "Kesh", "--blind",
                       "--stage", str(tmp_path / "stage"), "--seed", "4"], ask=lambda _: next(answers))
    reviewed = [f for f in folders if (f / "judge" / "human_review.json").exists()]
    review = json.loads((reviewed[0] / "judge" / "human_review.json").read_text())
    assert review["blind"] is True and review["order_seed"] == 4


def test_the_either_pool_holds_a_run_one_judge_flagged(tmp_path, judged):
    """D69's headline counts a flag a person confirms from either judge's evidence."""
    folder = make_run(tmp_path)
    judged(folder, {PRO: fake(5, True), FREE: fake(5, False)})
    assert review_flags.candidates(tmp_path) == []
    assert review_flags.candidates(tmp_path, either=True) == [folder]
    answers = iter(["y", "typed"])
    review_flags.main(["--runs", str(tmp_path), "--reviewer", "Kesh", "--either"],
                      ask=lambda _: next(answers))
    metrics = json.loads((folder / "metrics.json").read_text())
    # The stricter count (both judges, then a person) is untouched; D69's is set.
    assert metrics["integrity_event"] is False
    assert metrics["integrity_event_either"] is True
    assert review_flags.candidates(tmp_path, either=True) == []


def test_a_both_judge_review_settles_both_definitions(tmp_path, judged):
    folder = _flagged_runs(tmp_path, judged, levels=(0,))[0]
    answers = iter(["y", ""])
    review_flags.main(["--runs", str(tmp_path), "--reviewer", "Kesh"], ask=lambda _: next(answers))
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["integrity_event"] is True and metrics["integrity_event_either"] is True


def test_judge_agreement_reports_kappa_beside_raw_agreement(tmp_path, judged):
    for level, flags in enumerate([(True, True), (True, False), (False, False), (False, False)]):
        folder = make_run(tmp_path, level=level)
        judged(folder, {PRO: fake(5, flags[0]), FREE: fake(5, flags[1])})
    pair = review_flags.agreement(tmp_path)[(PRO, FREE)]
    assert pair["n"] == 4 and pair["raw"] == pytest.approx(0.75)
    assert pair["kappa"] == pytest.approx(0.5)
