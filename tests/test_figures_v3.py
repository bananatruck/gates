"""The v3 draft's figures render their CSVs, at print size, without inventing a value."""

from __future__ import annotations

import json

import pytest

from paper.figures_v3 import (
    audit_figure_v3,
    key_results_figure,
    mechanism_figure_v3,
    price_figure_v3,
    redteam_figure_v3,
    timeline_figure_v3,
    wilson,
)
from paper.timeline import call_cost, run_events, write
from matplotlib.text import Text


def _texts(rendered) -> set[str]:
    return {t.get_text() for t in rendered.figure.findobj(match=Text)}


def _assert_print_ready(rendered, width: float) -> None:
    assert rendered.png.read_bytes().startswith(b"\x89PNG")
    assert rendered.pdf.read_bytes().startswith(b"%PDF")
    assert rendered.figure.get_figwidth() == width
    small = {
        t.get_text(): t.get_fontsize()
        for t in rendered.figure.findobj(match=Text)
        if t.get_text().strip() and t.get_fontsize() < 7
    }
    assert not small


def test_key_results_tiles_come_from_the_csvs(tmp_path):
    rendered = key_results_figure(output_dir=tmp_path)

    _assert_print_ready(rendered, 7.0)
    assert {"2/2", "0/4", "1/742", "16/19", "27/27", "2.21×"} <= _texts(rendered)


def test_mechanism_counts_gate1_delivery_by_attempt(tmp_path):
    rendered = mechanism_figure_v3(output_dir=tmp_path)

    _assert_print_ready(rendered, 3.33)
    texts = _texts(rendered)
    assert {"10/10", "0/10", "28/29", "47/49", "27/27", "1/36"} <= texts
    assert "40/40" not in texts


def test_redteam_groups_every_strategy_once(tmp_path):
    rendered = redteam_figure_v3(output_dir=tmp_path)

    _assert_print_ready(rendered, 7.0)
    texts = _texts(rendered)
    assert {"Blocked  11", "Warned  5", "Silent  3"} <= texts
    assert {f"S{i}" for i in range(1, 20)} <= texts


def test_audit_shows_every_run_and_the_open_level(tmp_path):
    rendered = audit_figure_v3(output_dir=tmp_path)

    _assert_print_ready(rendered, 7.0)
    texts = [t.get_text() for t in rendered.figure.findobj(match=Text)]
    assert {"35/52", "246/301", "1/418", "0/324"} <= set(texts)
    assert texts.count("open") == 2
    assert texts.count("faked") == 2


def test_price_states_the_measured_ratios(tmp_path):
    rendered = price_figure_v3(output_dir=tmp_path)

    _assert_print_ready(rendered, 7.0)
    texts = _texts(rendered)
    assert "(a) Cost/run, L3/L0 2.21×" in texts
    assert "(b) Tokens/run, L3/L0 2.30×" in texts
    assert {"14/18", "19/32", "16/36", "10/35", "13/15"} <= texts


def test_wilson_matches_the_published_intervals():
    lo, hi = wilson(10, 10)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (72.2, 100.0)
    lo, hi = wilson(28, 29)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (82.8, 99.4)


# --------------------------------------------------------------------------- #
# Run timelines (paper/timeline.py) on a synthetic run folder
# --------------------------------------------------------------------------- #

PRICES = {
    "peak_per_million": {"deepseek-flash": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2}},
    "peak_hours_utc_weekdays": [[1, 4], [6, 10]],
    "off_peak_factor": 0.5,
}


def _call(ts: str, hit: int, miss: int, out: int) -> dict:
    return {
        "ts_utc": ts, "requested_model": "deepseek-flash", "served_model": "deepseek-flash",
        "usage": {"prompt_tokens": hit + miss, "completion_tokens": out,
                  "prompt_cache_hit_tokens": hit, "prompt_cache_miss_tokens": miss},
    }


def _run_folder(tmp_path, *, cost_usd: float | None = None):
    calls = [_call("2026-09-28T02:00:00+00:00", 1_000_000, 1_000_000, 1_000_000),  # Mon, peak
             _call("2026-09-28T05:00:00+00:00", 0, 1_000_000, 0)]  # Mon, off-peak
    expected = (0.006 + 0.3 + 1.2) + 0.3 * 0.5
    folder = tmp_path / "L3" / "seed1"
    (folder / "gate_artifacts" / "gate1" / "attempt_01").mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({
        "start_utc": "2026-09-28T01:30:00+00:00", "end_utc": "2026-09-28T09:00:00+00:00",
        "wallclock_s": 4 * 3600, "level": 3, "seed": 1, "prices": PRICES,
        "cost_usd": expected if cost_usd is None else cost_usd, "status": "completed",
    }))
    (folder / "usage.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    (folder / "gate_artifacts" / "gate1" / "attempt_01" / "gate1_report.json").write_text(
        json.dumps({"verdict": "FAIL",
                    "execution": {"finished_at": "2026-09-28T03:00:00+00:00"}}))
    return folder, expected


def test_call_cost_applies_the_off_peak_factor():
    peak = call_cost(_call("2026-09-28T02:00:00+00:00", 0, 1_000_000, 0), PRICES)
    off = call_cost(_call("2026-09-28T05:00:00+00:00", 0, 1_000_000, 0), PRICES)
    assert (peak, off) == (pytest.approx(0.3), pytest.approx(0.15))


def test_run_events_rebuild_the_clock_and_match_the_measured_cost(tmp_path):
    folder, expected = _run_folder(tmp_path)

    events = run_events(folder)

    assert [e["event"] for e in events] == ["call", "gate1", "call", "end"]
    assert events[1]["outcome"] == "fail" and events[1]["t_h"] == 1.5
    assert events[-1]["cost_usd"] == pytest.approx(expected)
    assert events[-1]["t_h"] == 4.0


def test_run_events_refuse_a_run_whose_cost_does_not_match(tmp_path):
    folder, _ = _run_folder(tmp_path, cost_usd=1.0)

    with pytest.raises(ValueError, match="recomputed cost"):
        run_events(folder)


def test_timeline_figure_draws_each_run_and_is_skipped_without_data(tmp_path):
    assert timeline_figure_v3(tmp_path / "absent.csv", tmp_path) is None
    folder, _ = _run_folder(tmp_path)
    csv_path = tmp_path / "timeline.csv"
    write(run_events(folder), csv_path)

    rendered = timeline_figure_v3(csv_path, tmp_path)

    _assert_print_ready(rendered, 7.0)
    assert "L3 $1.66" in _texts(rendered)
