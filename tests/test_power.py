"""Power analysis.

Goes in the paper's appendix, so it is checked against textbook values rather
than trusted because it looks plausible.
"""

from __future__ import annotations

import pytest

from analysis.power import (
    PairedScenario,
    mcnemar_min_n,
    mcnemar_power,
    report,
    unpaired_mde,
    unpaired_n,
)


def test_unpaired_n_matches_the_textbook_value():
    """0.50 vs 0.40, alpha .05 two-sided, power .80 -> ~388 per group."""
    assert unpaired_n(0.50, 0.40) == pytest.approx(388, abs=2)


def test_unpaired_n_shrinks_as_the_effect_grows():
    assert unpaired_n(0.5, 0.3) < unpaired_n(0.5, 0.4)


def test_unpaired_n_is_symmetric_in_its_arguments():
    assert unpaired_n(0.5, 0.35) == unpaired_n(0.35, 0.5)


def test_unpaired_n_rejects_a_null_effect():
    with pytest.raises(ValueError):
        unpaired_n(0.4, 0.4)


def test_higher_power_costs_more_cases():
    assert unpaired_n(0.5, 0.4, power=0.90) > unpaired_n(0.5, 0.4, power=0.80)


def test_mde_at_the_planned_n_is_about_sixteen_points():
    """The number that says n=150 is underpowered for a 10-point claim."""
    assert unpaired_mde(150, 0.5, 0.80) == pytest.approx(0.162, abs=0.005)


def test_mde_falls_as_n_rises():
    assert unpaired_mde(400) < unpaired_mde(150)


# ------------------------------------------------------------- McNemar


def test_mcnemar_power_rises_with_n():
    sc = PairedScenario(p_en=0.5, delta=0.10, discordance=0.25)
    assert mcnemar_power(sc, 100, iters=600) < mcnemar_power(sc, 400, iters=600)


def test_mcnemar_power_rises_as_discordance_falls():
    """Fewer flipped cases means a cleaner paired signal at the same effect."""
    low = PairedScenario(p_en=0.5, delta=0.10, discordance=0.15)
    high = PairedScenario(p_en=0.5, delta=0.10, discordance=0.50)
    assert mcnemar_power(low, 150, iters=600) > mcnemar_power(high, 150, iters=600)


def test_mcnemar_beats_unpaired_when_discordance_is_low():
    """The reason the paired design is worth insisting on."""
    sc = PairedScenario(p_en=0.5, delta=0.10, discordance=0.15)
    assert mcnemar_min_n(sc, iters=600) < unpaired_n(0.50, 0.40)


def test_effect_cannot_exceed_discordance():
    """b - c = delta*n and b + c = disc*n forces |delta| <= discordance."""
    sc = PairedScenario(p_en=0.5, delta=0.40, discordance=0.20)
    with pytest.raises(ValueError):
        mcnemar_power(sc, 150, iters=100)


def test_mcnemar_power_is_seeded_and_reproducible():
    sc = PairedScenario(p_en=0.5, delta=0.12, discordance=0.30)
    assert mcnemar_power(sc, 150, iters=600) == mcnemar_power(sc, 150, iters=600)


def test_min_n_actually_reaches_the_target_power():
    sc = PairedScenario(p_en=0.5, delta=0.15, discordance=0.25)
    n = mcnemar_min_n(sc, target_power=0.80, iters=600)
    assert mcnemar_power(sc, n, iters=600) >= 0.80
    assert mcnemar_power(sc, n - 1, iters=600) < 0.82  # genuinely near the boundary


def test_report_states_the_decision_rule():
    """The report is the appendix; it must carry the post-Phase-2 rule."""
    text = report(150, iters=200)
    assert "16.2 points" in text
    assert "McNemar" in text
    assert "discordance <= 0.20" in text
    assert "$" in text  # the cost table rendered
