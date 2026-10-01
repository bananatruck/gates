"""Score every run's paper with MLR-Bench's own judge prompts (D54, D62).

    cd ~/gates && ~/AgentLaboratory-Gemini/.venv/bin/python -m rig.judge \\
        --runs ~/gates-runs/runs-pilot --mlrbench <kit>/mlrbench \\
        --judge deepseek-v4-pro \\
        --judge openrouter/nvidia/nemotron-3-ultra-550b-a55b:free \\
        --key-file <AI_keys.env>

API judges use the host's client through ``rig.live``, which records every API
call's tokens. An ``agent:claude/<model>``, ``agent:cursor/<model>``, or
``agent:codex/<model>[:<effort>]`` judge instead runs the installed agent CLI.
Its prompt is sent through stdin and its outputs carry
``reviewed by agent for now``.

The prompts are not retyped. ``OVERALL_RUBRIC`` and ``HALLUCINATION_RUBRIC``,
the two ``overall_review`` functions that wrap them, and the helpers they call
are read from the pinned MLR-Bench files, checked against the kit's
``SHA256SUMS``, and executed as written. Each judge receives what MLR-Bench's
judges receive: the task text, the paper, and the run's experiment code
(``code/``). Where this differs from MLR-Bench, it is recorded in each
judge's ``meta.json``. The host's client sends an empty system message, has no
JSON mode, and sends no images. MLR-Bench's judges saw the paper's figures,
while Agent Laboratory's report here is LaTeX source. Agent CLIs also provide
no temperature or maximum-token control.

A judge never scores a run of its own model (D54): on a ``deepseek-v4-pro``
run only the other judge counts.

``--opinion`` runs the same review and records the same files, but opinions
never contribute to completion, candidacy, scores, metrics, or agreement.
An opinion also never reviews a run of its own model.

Per run, in ``judge/``:

* ``<name>/overall.json``, ``<name>/hallucination.json``, ``<name>/meta.json``
  and ``<name>/usage.jsonl`` for every eligible judge and opinion;
* ``summary.json``: eligible judges and opinions in separate objects, with
  each overall score and its "Faked Experimental Results" findings.

A paper is a faked-results candidate only when every eligible judge flags it.
A candidate waits for a person (``rig.review_flags``); every other run gets its
``metrics.json`` here. A run with no paper gets no MLR-Judge score and no
integrity event, and stays in the denominator (``paper/PLAN.md`` §3).

The tool resumes: a judge or opinion whose files exist is not asked again, and
a run with ``metrics.json`` is skipped.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import sys
import typing
from pathlib import Path

from rig.live import (
    AGENT_REVIEW_TAG,
    DEFAULT_AGENT_TIMEOUT_S,
    AgentCLIError,
    AgentModel,
    ModelFn,
    agent_judge_parts,
    agent_model,
    load_key_file,
)

FAKED = "Faked Experimental Results"
USAGE_ENV = "AGENTLAB_USAGE_LOG"
MAX_TOKENS = 16384  # MLR-Bench's judge setting: 8192 * 2, temperature 0

_HELPERS = (
    "extract_json_between_markers", "read_text", "load_multimodal_content",
    "resolve_image_path", "read_combine_files",
)


class KitError(RuntimeError):
    """The MLR-Bench copy does not match its checksums."""


def verify_kit(mlrbench: Path, paths: list[str]) -> None:
    sums = {}
    for line in (mlrbench / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name.strip()] = digest.strip()
    for name in paths:
        actual = hashlib.sha256((mlrbench / name).read_bytes()).hexdigest()
        if sums.get(name) != actual:
            raise KitError(f"{name} does not match SHA256SUMS: the judge would not be MLR-Bench's")


def _segments(path: Path, names: tuple[str, ...]) -> str:
    source = path.read_text(encoding="utf-8")
    wanted = []
    for node in ast.parse(source).body:
        named = isinstance(node, ast.FunctionDef) and node.name in names
        assigned = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        )
        if named or assigned:
            wanted.append(ast.get_source_segment(source, node))
    return "\n\n".join(wanted)


def load_judges(mlrbench: Path) -> tuple:
    """MLR-Bench's two review functions, executed from the pinned files."""
    verify_kit(mlrbench, ["judge/overall_review.py", "judge/eval_hallucination.py",
                          "judge/utils.py"])
    base = {"os": os, "re": re, "json": json, "Path": Path, "Union": typing.Union,
            "pymupdf4llm": None}
    exec(compile(_segments(mlrbench / "judge" / "utils.py", _HELPERS),
                 "mlrbench/utils/utils.py", "exec"), base)
    missing = [name for name in _HELPERS if name not in base]
    if missing:
        raise KitError(f"judge/utils.py lacks {missing}")
    reviews = []
    for file, rubric in (("overall_review.py", "OVERALL_RUBRIC"),
                         ("eval_hallucination.py", "HALLUCINATION_RUBRIC")):
        namespace = dict(base)
        exec(compile(_segments(mlrbench / "judge" / file, (rubric, "overall_review")),
                     f"mlrbench/evals/{file}", "exec"), namespace)
        if "overall_review" not in namespace or rubric not in namespace:
            raise KitError(f"judge/{file} lacks {rubric} or overall_review")
        reviews.append(namespace["overall_review"])
    return tuple(reviews)


class Client:
    """The ``client.generate`` MLR-Bench's review functions call."""

    def __init__(self, model: ModelFn):
        self.model = model
        self.images_dropped = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, media=None):
        self.images_dropped += len(media or [])
        self.prompts.append(hashlib.sha256(prompt.encode()).hexdigest())
        return self.model(prompt, ""), None


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def eligible(judges: list[str], model: str) -> list[str]:
    def model_name(judge: str) -> str:
        parts = agent_judge_parts(judge)
        return parts[1] if parts else judge

    return [judge for judge in judges if judge != model and model_name(judge) != model]


def faked_findings(hallucination: dict | None) -> list[dict]:
    if not hallucination:
        return []
    return [h for h in hallucination.get("hallucinations", [])
            if str(h.get("type", "")).strip() == FAKED]


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _judge_outputs_complete(folder: Path) -> bool:
    paths = (folder / "overall.json", folder / "hallucination.json")
    if not all(path.exists() for path in paths):
        return False
    return all("error" not in json.loads(path.read_text(encoding="utf-8")) for path in paths)


def judge_run(folder: Path, *, judges: list[str], models: dict[str, ModelFn],
              reviews: tuple, tasks: Path, opinions: list[str] | None = None) -> str:
    """Judge one run folder. Returns what happened, in a few words."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") == "void":
        return "skipped: void"
    if (folder / "metrics.json").exists():
        return "skipped: already has metrics.json"
    paper = folder / "paper" / "report.txt"
    if not paper.is_file():
        reason_file = folder / "paper" / "reason.txt"
        reason = reason_file.read_text(encoding="utf-8").strip() if reason_file.exists() else ""
        _write(folder / "metrics.json", {
            "integrity_event": False, "task_score": None,
            "no_paper": reason or manifest.get("status", "no paper"),
        })
        return "no paper: metrics written without a score"
    task_file = tasks / f"{manifest['task']}.md"
    if hashlib.sha256(task_file.read_bytes()).hexdigest() != manifest.get("task_file_sha256"):
        raise KitError(f"{task_file} is not the task text this run was given")
    code = folder / "code"
    overall_fn, hallucination_fn = reviews

    opinions = opinions or []
    eligible_judges = eligible(judges, manifest["model"])
    eligible_opinions = eligible(opinions, manifest["model"])
    agent_judged = any(agent_judge_parts(judge) for judge in eligible_judges)
    agent_recorded = agent_judged or any(
        agent_judge_parts(opinion) for opinion in eligible_opinions
    )
    summary = {
        "judges": {},
        "opinions": {},
        "excluded": [
            name for name in judges + opinions
            if name not in eligible_judges and name not in eligible_opinions
        ],
    }
    if agent_recorded:
        summary["review_tag"] = AGENT_REVIEW_TAG
    complete = True
    for role, names in (
        ("judges", eligible_judges),
        ("opinions", eligible_opinions),
    ):
        for judge in names:
            out = folder / "judge" / slug(judge)
            if not _judge_outputs_complete(out):
                out.mkdir(parents=True, exist_ok=True)
                os.environ[USAGE_ENV] = str(out / "usage.jsonl")
                model = models[judge]
                client = Client(model)
                args = dict(
                    paper_path=str(paper),
                    client=client,
                    task_file=str(task_file),
                    code_path=str(code) if code.is_dir() else None,
                )
                overall = overall_fn(**args)
                hallucination = hallucination_fn(**args)
                _write(
                    out / "overall.json",
                    overall[0] if overall else {"error": "no valid JSON after 3 attempts"},
                )
                _write(
                    out / "hallucination.json",
                    hallucination[0]
                    if hallucination
                    else {"error": "no valid JSON after 3 attempts"},
                )
                meta = {
                    "judge": judge,
                    "prompt_sha256": client.prompts,
                    "images_dropped": client.images_dropped,
                    "system_message": "",
                    "json_mode": False,
                    "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                }
                if isinstance(model, AgentModel):
                    meta.update({
                        "judge_kind": "agent",
                        "command": list(model.command),
                        "cli": model.cli,
                        "model": model.model,
                        "effort": model.effort,
                        "cli_version": model.cli_version,
                        "review_tag": AGENT_REVIEW_TAG,
                        "mlr_bench_deviations": {
                            "temperature": (
                                "the agent CLI does not expose temperature control; "
                                "MLR-Bench uses 0"
                            ),
                            "max_tokens": (
                                "the agent CLI does not expose max-token control; "
                                "MLR-Bench uses 16384"
                            ),
                        },
                    })
                else:
                    meta.update({
                        "judge_kind": "api",
                        "max_tokens": MAX_TOKENS,
                        "temperature": 0.0,
                    })
                _write(out / "meta.json", meta)
            overall = json.loads((out / "overall.json").read_text(encoding="utf-8"))
            hallucination = json.loads(
                (out / "hallucination.json").read_text(encoding="utf-8")
            )
            errored = "error" in overall or "error" in hallucination
            if role == "judges" and errored:
                complete = False
            score = (
                (overall.get("Overall") or {}).get("score")
                if "error" not in overall
                else None
            )
            summary[role][judge] = {
                "overall": score,
                "faked": (
                    faked_findings(hallucination)
                    if "error" not in hallucination
                    else None
                ),
            }

    flags = [bool(v["faked"]) for v in summary["judges"].values() if v["faked"] is not None]
    summary["complete"] = complete and bool(summary["judges"])
    summary["candidate"] = summary["complete"] and all(flags) and bool(flags)
    scores = [v["overall"] for v in summary["judges"].values() if v["overall"] is not None]
    summary["task_score"] = sum(scores) / len(scores) if summary["complete"] and scores else None
    _write(folder / "judge" / "summary.json", summary)

    if not summary["complete"]:
        return "incomplete: a judge gave no valid JSON; rerun to retry it"
    if summary["candidate"]:
        return "faked-results candidate: waiting for rig.review_flags"
    metrics = {
        "integrity_event": False,
        # D69's headline: settled here only when no judge flagged; a single
        # judge's flag waits for a person (rig.review_flags --either).
        "integrity_event_either": None if any(flags) else False,
        "task_score": summary["task_score"],
        "judges": list(summary["judges"]), "human_reviewed": False,
    }
    if agent_judged:
        metrics["review_tag"] = AGENT_REVIEW_TAG
    _write(folder / "metrics.json", metrics)
    return f"scored {summary['task_score']:.2f}, not flagged by every judge"


def main(argv: list[str] | None = None, *, model_for=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rig.judge",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True, help="a run root, e.g. runs-pilot")
    parser.add_argument("--mlrbench", type=Path, required=True,
                        help="the kit's pinned MLR-Bench copy (holds SHA256SUMS)")
    parser.add_argument("--judge", action="append", required=True,
                        help="a judge model as the host names it; give two")
    parser.add_argument(
        "--opinion", action="append", default=[],
        help="a non-counting judge model; repeat for more opinions",
    )
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--host", type=Path, default=None)
    parser.add_argument(
        "--agent-timeout", type=float, default=DEFAULT_AGENT_TIMEOUT_S,
        help="seconds allowed for each agent CLI call (default: 1800)",
    )
    args = parser.parse_args(argv)

    try:
        for name in args.judge + args.opinion:
            agent_judge_parts(name)
    except AgentCLIError as error:
        parser.error(str(error))
    overlap = set(args.judge) & set(args.opinion)
    if overlap:
        parser.error(
            "a model cannot be both --judge and --opinion: " + ", ".join(sorted(overlap))
        )
    if args.key_file is not None:
        load_key_file(args.key_file)
    if model_for is None:
        from rig.live import model_from_args

        def model_for(judge):
            if agent_judge_parts(judge):
                return agent_model(judge, timeout_s=args.agent_timeout)
            return model_from_args(argparse.Namespace(
                backend=judge, key_file=None, host=args.host,
                max_tokens=MAX_TOKENS, temp=0.0))

    reviews = load_judges(args.mlrbench)
    models = {name: model_for(name) for name in args.judge + args.opinion}
    tasks = args.mlrbench / "tasks"
    for manifest_path in sorted(args.runs.rglob("manifest.json")):
        folder = manifest_path.parent
        outcome = judge_run(folder, judges=args.judge, models=models,
                            reviews=reviews, tasks=tasks, opinions=args.opinion)
        print(f"{folder.relative_to(args.runs)}: {outcome}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
