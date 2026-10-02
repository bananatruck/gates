"""Gate 3's adjacency check against the 09-30 probes (D85, D103).

The counts are what the paper reports, so they are held here.
"""

from __future__ import annotations

import pytest

from rig.adjacency_probes import ATTACKS, HONEST, counts, main, run_all, run_probe


def test_the_probe_set_is_the_one_the_paper_counts():
    assert len(ATTACKS) == 27
    assert len(HONEST) == 36


@pytest.mark.parametrize("body", ATTACKS)
def test_every_attack_fails_the_adjacency_check(body, tmp_path):
    assert run_probe("attack", body, str(tmp_path)).adjacency is False


@pytest.mark.parametrize("body", [b for b in HONEST if "\\result" in b or "\\setting" in b])
def test_every_tokenised_honest_shape_passes(body, tmp_path):
    assert run_probe("honest", body, str(tmp_path)).adjacency is True


def test_the_counts_and_the_one_untokenised_honest_shape(capsys):
    tally = counts(run_all())
    assert tally == {"attack": (27, 27), "honest": (35, 36)}
    assert main([]) == 0
    assert "27 of 27 attacks caught; 35 of 36 honest shapes pass" in capsys.readouterr().out
