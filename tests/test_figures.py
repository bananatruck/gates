"""Paper figures render their CSV inputs without inventing results."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from paper.figures import (
    adjacency_figure,
    agent_judge_figure,
    audit_figure,
    crashes_figure,
    gate_attempts_figure,
    mechanism_figure,
    price_figure,
    redteam_figure,
    tokens_cost_figure,
)
from matplotlib.text import Text


FIELDS = ["benchmark", "level", "seed", "judge", "role", "overall", "faked", "status"]
REPO = Path(__file__).resolve().parents[1]


def _judging_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _visible_text(rendered) -> set[str]:
    return {text.get_text() for text in rendered.figure.texts}


def _drawn_text(rendered) -> set[str]:
    texts = list(rendered.figure.texts)
    labels = set()
    for axis in rendered.figure.axes:
        texts.extend(axis.texts)
        texts.extend((axis.xaxis.label, axis.yaxis.label))
        texts.extend(axis.get_xticklabels())
        texts.extend(axis.get_yticklabels())
        labels.update(axis.get_title(loc=loc) for loc in ("left", "center", "right"))
    return {text.get_text() for text in texts} | labels


def _assert_saved(rendered) -> None:
    assert rendered.png.read_bytes().startswith(b"\x89PNG")
    assert rendered.pdf.read_bytes().startswith(b"%PDF")


def _assert_publication_size(rendered, width: float) -> None:
    assert rendered.figure.get_figwidth() == width
    assert int.from_bytes(rendered.png.read_bytes()[16:20], "big") == round(width * 300)
    undersized = {
        text.get_text(): text.get_fontsize()
        for text in rendered.figure.findobj(match=Text)
        if text.get_text().strip() and text.get_fontsize() < 7
    }
    assert not undersized


def test_agent_judge_fixture_draws_png_and_pdf_without_placeholder(tmp_path):
    fixture = _judging_csv(
        tmp_path / "judging.csv",
        [
            {"benchmark": "MLR-Bench", "level": "L0", "seed": "1", "judge": "j1", "role": "judge", "overall": "4.0", "faked": "true", "status": ""},
            {"benchmark": "MLR-Bench", "level": "L0", "seed": "1", "judge": "j2", "role": "opinion", "overall": "5.0", "faked": "false", "status": ""},
            {"benchmark": "MLR-Bench", "level": "L1", "seed": "2", "judge": "j1", "role": "judge", "overall": "", "faked": "", "status": ""},
            {"benchmark": "MLR-Bench", "level": "L1", "seed": "2", "judge": "j2", "role": "opinion", "overall": "7.0", "faked": "false", "status": ""},
        ],
    )

    rendered = agent_judge_figure(fixture, tmp_path)

    assert rendered is not None
    _assert_saved(rendered)
    assert "PLACEHOLDER" not in _visible_text(rendered)


def test_agent_judge_opinions_are_drawn_but_excluded_from_statistics(tmp_path):
    fixture = _judging_csv(
        tmp_path / "judging.csv",
        [
            {"benchmark": "MLR-Bench", "level": "L0", "seed": "1", "judge": "j1", "role": "judge", "overall": "4.0", "faked": "true", "status": ""},
            {"benchmark": "MLR-Bench", "level": "L0", "seed": "1", "judge": "j2", "role": "opinion", "overall": "10.0", "faked": "false", "status": ""},
            {"benchmark": "MLR-Bench", "level": "L0", "seed": "1", "judge": "j3", "role": "opinion", "overall": "", "faked": "", "status": ""},
        ],
    )

    rendered = agent_judge_figure(fixture, tmp_path)

    assert rendered is not None
    text = _drawn_text(rendered)
    assert "4.0" in text
    assert "7.0" not in text
    assert "flagged 1" in text
    assert not {"clear 1", "error 1"} & text
    assert any("opinion (not counted)" in label for label in text)


def test_dummy_judging_row_keeps_placeholder_stamp(tmp_path):
    fixture = _judging_csv(
        tmp_path / "judging.csv",
        [
            {"benchmark": "MLR-Bench", "level": "L0", "seed": "1", "judge": "j1", "role": "judge", "overall": "4.0", "faked": "false", "status": "dummy"},
        ],
    )

    rendered = agent_judge_figure(fixture, tmp_path)

    assert rendered is not None
    assert "PLACEHOLDER" in _visible_text(rendered)


def test_agent_judge_figure_is_skipped_when_csv_is_absent(tmp_path):
    assert agent_judge_figure(tmp_path / "missing.csv", tmp_path) is None


def test_crashes_fixture_draws_real_causes_without_placeholder(tmp_path):
    path = tmp_path / "crashes.csv"
    path.write_text(
        "benchmark,system,arm,runs,executions,crashed,harness,agent,oom_at_cap,oom,timeout,environment,agent_code,papers,papers_on_crashed_run,papers_on_harness_crash\n"
        "MLR-Bench,Agent Lab,L0,2,8,4,3,1,2,0,1,0,1,2,1,1\n"
        "MLR-Bench,Agent Lab,L1,2,7,2,1,1,0,0,1,0,1,2,0,0\n",
        encoding="utf-8",
    )

    rendered = crashes_figure(path, tmp_path)

    _assert_saved(rendered)
    assert "PLACEHOLDER" not in _visible_text(rendered)


def test_crashes_figure_draws_measured_owner_totals(tmp_path):
    rendered = crashes_figure(REPO / "paper" / "crashes.csv", tmp_path)

    text = _drawn_text(rendered)
    assert "Harness-caused crashes (46 total)" in text
    assert "Agent-caused crashes (13 total)" in text


def test_tokens_and_cost_fixture_draws_seed_points_without_placeholder(tmp_path):
    path = tmp_path / "waves23.csv"
    path.write_text(
        "benchmark,task,level,seed,phase,status,paper,cost_usd,wallclock_h,calls,tokens_prompt,tokens_completion,tokens_reasoning,peak_gpu_mb,executions,crashed,harness_crashes,agent_crashes,g1_attempts,g1_fails,g2_attempts,g2_fails,g3_attempts,g3_fails,judges\n"
        "MLR-Bench,task,L0,1,main,completed,True,0.5,1.0,2,100,40,10,50,1,0,0,0,0,0,0,0,0,0,\n"
        "MLR-Bench,task,L0,2,main,completed,True,0.7,1.0,2,120,30,20,50,1,0,0,0,0,0,0,0,0,0,\n"
        "MLR-Bench,task,L1,1,main,completed,True,1.0,1.0,2,200,60,40,50,1,0,0,0,2,1,0,0,0,0,\n"
        "MLR-Bench,task,L1,2,main,completed,True,1.2,1.0,2,240,50,30,50,1,0,0,0,3,1,0,0,0,0,\n",
        encoding="utf-8",
    )

    rendered = tokens_cost_figure(path, tmp_path)

    _assert_saved(rendered)
    assert "PLACEHOLDER" not in _visible_text(rendered)


def test_tokens_draw_prompt_plus_completion_without_double_counting_reasoning(tmp_path):
    path = tmp_path / "waves23.csv"
    path.write_text(
        "level,cost_usd,tokens_prompt,tokens_completion,tokens_reasoning\n"
        "L0,0.50,1000000,2000000,750000\n",
        encoding="utf-8",
    )

    rendered = tokens_cost_figure(path, tmp_path)

    assert "3.00M" in _drawn_text(rendered)


def test_gate_attempts_fixture_draws_rejections_without_placeholder(tmp_path):
    path = tmp_path / "waves23.csv"
    path.write_text(
        "level,g1_attempts,g1_fails,g2_attempts,g2_fails,g3_attempts,g3_fails\n"
        "L0,0,0,0,0,0,0\n"
        "L1,4,2,0,0,0,0\n"
        "L2,5,1,2,1,0,0\n"
        "L3,3,1,2,0,4,2\n",
        encoding="utf-8",
    )

    rendered = gate_attempts_figure(path, tmp_path)

    _assert_saved(rendered)
    assert "PLACEHOLDER" not in _visible_text(rendered)


def test_red_team_fixture_draws_each_strategy_without_placeholder(tmp_path):
    path = tmp_path / "redteam.csv"
    path.write_text(
        "id,description,outcome\n"
        "S1,literal at the call,blocked\n"
        "S11,every epoch recorded,warned\n"
        "S18,literal through an alias,silent\n",
        encoding="utf-8",
    )

    rendered = redteam_figure(path, tmp_path)

    _assert_saved(rendered)
    assert "PLACEHOLDER" not in _visible_text(rendered)


def test_red_team_figure_draws_measured_outcome_totals(tmp_path):
    rendered = redteam_figure(REPO / "paper" / "redteam.csv", tmp_path)

    text = _drawn_text(rendered)
    assert {"Blocked\n7", "Warned\n1", "Silent\n10"} <= text


def test_adjacency_fixture_draws_catches_and_passes_without_placeholder(tmp_path):
    path = tmp_path / "adjacency.csv"
    path.write_text(
        "kind,shape,caught,passed\n"
        "attack,token followed by a digit,true,\n"
        "attack,token followed by text,false,\n"
        "honest,token followed by punctuation,,true\n"
        "honest,plain prose with no token,,false\n",
        encoding="utf-8",
    )

    rendered = adjacency_figure(path, tmp_path)

    _assert_saved(rendered)
    assert "PLACEHOLDER" not in _visible_text(rendered)


def test_adjacency_figure_draws_measured_probe_totals(tmp_path):
    rendered = adjacency_figure(REPO / "paper" / "adjacency.csv", tmp_path)

    text = _drawn_text(rendered)
    assert "23/27 caught" in text
    assert "35/36 passed" in text


def test_mechanism_fixture_draws_vector_and_raster_without_placeholder(tmp_path):
    path = tmp_path / "mechanism.csv"
    path.write_text(
        "id,panel,measure,arm,k,n,interval,source\n"
        "g1_on,Gate 1,Required values delivered,Gate 1 on,4,4,yes,signed campaign\n"
        "g2_detect,Gate 2,Defective registries rejected,,3,4,yes,model-free rig\n",
        encoding="utf-8",
    )

    rendered = mechanism_figure(path, tmp_path)

    _assert_saved(rendered)
    assert "PLACEHOLDER" not in _visible_text(rendered)


def test_figures_use_publication_widths_and_legible_text(tmp_path):
    judging = _judging_csv(
        tmp_path / "judging.csv",
        [
            {"benchmark": "MLR-Bench", "level": "L0", "seed": "1", "judge": "j1", "role": "judge", "overall": "4.0", "faked": "false", "status": ""},
        ],
    )
    full_width = [
        crashes_figure(output_dir=tmp_path),
        tokens_cost_figure(output_dir=tmp_path),
        gate_attempts_figure(output_dir=tmp_path),
        redteam_figure(output_dir=tmp_path),
        mechanism_figure(output_dir=tmp_path),
        agent_judge_figure(judging, tmp_path),
    ]

    for rendered in full_width:
        assert rendered is not None
        _assert_publication_size(rendered, 7)
    _assert_publication_size(adjacency_figure(output_dir=tmp_path), 3.3)


def test_public_run_aggregates_contain_no_machine_path():
    with open(REPO / "paper" / "waves23.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 8
    assert "folder" not in (reader.fieldnames or [])
    assert not any("/home/" in value for row in rows for value in row.values())


def test_rig_csvs_hold_the_measured_totals():
    with open(REPO / "paper" / "redteam.csv", newline="", encoding="utf-8") as handle:
        redteam = list(csv.DictReader(handle))
    with open(REPO / "paper" / "adjacency.csv", newline="", encoding="utf-8") as handle:
        adjacency = list(csv.DictReader(handle))

    assert Counter(row["outcome"] for row in redteam) == {
        "blocked": 7,
        "warned": 1,
        "silent": 10,
    }
    attacks = [row for row in adjacency if row["kind"] == "attack"]
    honest = [row for row in adjacency if row["kind"] == "honest"]
    assert (sum(row["caught"] == "true" for row in attacks), len(attacks)) == (23, 27)
    assert (sum(row["passed"] == "true" for row in honest), len(honest)) == (35, 36)


def test_audit_figure_draws_every_run_and_rater_from_the_csvs(tmp_path):
    rendered = audit_figure(output_dir=tmp_path)

    _assert_saved(rendered)
    _assert_publication_size(rendered, 7)
    text = _drawn_text(rendered)
    assert "PLACEHOLDER" not in _visible_text(rendered)
    assert {"35/52", "246/301", "12/627", "0/330", "24/472", "11/418", "1/418", "0/324"} <= text
    assert {"L0 seed 1", "L3 seed 2"} <= text


def test_audit_figure_shows_a_missing_judge_answer_as_none(tmp_path):
    rendered = audit_figure(output_dir=tmp_path)

    texts = [t.get_text() for axis in rendered.figure.axes for t in axis.texts]
    assert texts.count("none") == 2  # L3 seed 2, GPT-5.6 Sol: flag and score
    assert texts.count("open") == 2  # L2 rule A waits on a person (D99)


def test_provenance_csv_holds_the_ledger_counts():
    with open(REPO / "paper" / "provenance.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert sum(int(r["numerals"]) for r in rows) == 2942
    assert [r["rule_a_faked"] for r in rows if r["level"] == "L0"] == ["true", "true"]
    assert not any("/home/" in value for row in rows for value in row.values())


def test_price_figure_draws_ratios_and_crash_totals(tmp_path):
    rendered = price_figure(output_dir=tmp_path)

    _assert_saved(rendered)
    _assert_publication_size(rendered, 7)
    text = _drawn_text(rendered)
    assert "(a) Cost/run, L3/L0 2.21x" in text
    assert "(b) Tokens/run, L3/L0 2.30x" in text
    assert {"14/18", "19/32", "16/36", "10/35", "13/15"} <= text
