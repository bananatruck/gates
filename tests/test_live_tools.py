"""The live rig tools, run end to end with a fake model and a fake host (D61).

The numbers these tools produce need a real model. What is held here is that
the command lines reach the measurement, that the model is built through the
host's own client, and that a key file's values go into the environment and
nowhere else.
"""

from __future__ import annotations

import os
import sys

import pytest

from rig import corpus, tuning
from rig.live import load_key_file, model_from_args

SENTINEL = "sk-sentinel-never-print-this-0000"

PROGRAM = """\
import random
seed = 0
random.seed(seed)
record_metadata("seed", seed)
correct = sum(1 for _ in range(500) if random.random() < 0.8)
record_result("exp1.test_acc", correct / 500, unit="ratio")
"""


def test_tuning_cli_runs_both_arms_and_writes_the_table(tmp_path, capsys):
    out = tmp_path / "arms.txt"
    code = tuning.main(
        ["--backend", "fake", "--workdir", str(tmp_path / "runs"), "--seeds", "1",
         "--max-attempts", "1", "--out", str(out)],
        model_fn=lambda prompt, system: PROGRAM,
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "FEEDBACK ARM COMPARISON" in printed
    # The loop prints each verdict as it goes; the table is what gets saved.
    table = out.read_text(encoding="utf-8").strip()
    assert table.startswith("FEEDBACK ARM COMPARISON")
    assert printed.rstrip().endswith(table)
    assert (tmp_path / "runs" / "template").is_dir()
    assert (tmp_path / "runs" / "generated").is_dir()


def test_tuning_cli_refuses_to_run_without_a_backend(capsys):
    with pytest.raises(SystemExit):
        tuning.main([])
    assert "--backend" in capsys.readouterr().err


def test_corpus_cli_scores_the_shipped_scanner_offline(capsys):
    assert corpus.main([]) == 0
    printed = capsys.readouterr().out
    assert "deterministic (shipped pattern set)" in printed
    assert "few-shot" not in printed


def test_corpus_cli_scores_every_prompt_variant_with_a_model(capsys):
    calls: list[str] = []

    def never_flags(prompt: str, system: str) -> str:
        calls.append(prompt)
        return "NO"

    assert corpus.main(["--backend", "fake"], model_fn=never_flags) == 0
    printed = capsys.readouterr().out
    for variant in ("no few-shot", "few-shot k=2", "few-shot k=3"):
        assert f"deterministic + model, {variant}" in printed
    assert calls, "the model was never asked"


def test_a_key_file_loads_names_into_the_environment_and_returns_no_value(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "already-exported")
    keys = tmp_path / "AI_keys.env"
    keys.write_text(
        f"# comment\nDEEPSEEK_API_KEY = {SENTINEL}\n"
        "GOOGLE_API_KEY = 'from-the-file'\nNOT_A_KEY = x\n",
        encoding="utf-8",
    )
    loaded = load_key_file(keys)
    assert loaded == ["DEEPSEEK_API_KEY"]
    assert os.environ["DEEPSEEK_API_KEY"] == SENTINEL
    # An explicit export wins over the file.
    assert os.environ["GOOGLE_API_KEY"] == "already-exported"
    assert "NOT_A_KEY" not in os.environ
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err


def test_the_model_is_the_hosts_own_client(tmp_path, monkeypatch):
    """model_from_args imports the host's query_model, as the host's runs do."""
    host = tmp_path / "host"
    host.mkdir()
    (host / "inference.py").write_text(
        "CALLS = []\n"
        "def query_model(**kwargs):\n"
        "    CALLS.append(kwargs)\n"
        "    return 'answer'\n",
        encoding="utf-8",
    )
    keys = tmp_path / "keys.env"
    keys.write_text(f"DEEPSEEK_API_KEY = {SENTINEL}\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "inference", raising=False)

    import argparse

    args = argparse.Namespace(
        backend="deepseek-flash", key_file=keys, host=host, max_tokens=4096, temp=0.0
    )
    model = model_from_args(args)
    assert model("prompt", "system") == "answer"

    import inference

    (call,) = inference.CALLS
    assert call["model_str"] == "deepseek-flash"
    assert call["max_tokens"] == 4096
    # The key reaches the client through the environment, never as an argument.
    assert SENTINEL not in repr(call)
    assert os.environ["DEEPSEEK_API_KEY"] == SENTINEL
