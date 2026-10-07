"""analysis/stats.py against hand-computed values."""

from __future__ import annotations

import pytest

from analysis.stats import (
    compare_attack_success,
    compare_utility,
    efficacy_ci,
    efficacy_point,
    holm,
    mcnemar_exact,
    paired,
)
from harness.scoring import CaseResult


def cr(i: int, *, atk=False, util=False, ach=True, err=False, defense="d") -> CaseResult:
    return CaseResult(case_id=f"c{i}", lang="en", defense=defense, model="m",
                      attacker_task_success=atk, user_task_success=util,
                      attacker_task_achievable=ach, errored=err)


def test_mcnemar_exact_matches_hand_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(6, 0) == pytest.approx(2 / 64)          # 0.03125
    assert mcnemar_exact(12, 1) == pytest.approx(2 * 14 / 8192)  # the en-vs-hi probe result
    assert mcnemar_exact(14, 1) == pytest.approx(2 * 16 / 32768)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def test_paired_counts_only_shared_keys():
    p = paired({"a": True, "b": True, "c": False, "x": True}, {"a": True, "b": False, "c": True})
    assert (p.only_a, p.only_b, p.both, p.neither, p.n) == (1, 1, 1, 0, 3)


def test_attack_comparison_uses_the_uasr_set():
    undefended = [cr(0, atk=True), cr(1, atk=True), cr(2, atk=True, ach=False), cr(3, atk=True, err=True)]
    defended = [cr(0), cr(1, atk=True), cr(2), cr(3)]
    p = compare_attack_success(undefended, defended)
    # case 2 unachievable and case 3 errored drop out; only case 0 flips
    assert (p.only_a, p.only_b, p.both, p.n) == (1, 0, 1, 2)


def test_utility_comparison_ignores_achievability():
    a = [cr(0, util=True, ach=False), cr(1, util=True)]
    b = [cr(0), cr(1, util=True)]
    assert compare_utility(a, b).only_a == 1


def test_efficacy_point_and_undefined_cases():
    und = [cr(i, atk=i < 4) for i in range(8)]          # uASR 0.5
    dfd = [cr(i, atk=i < 1) for i in range(8)]          # uASR 0.125
    assert efficacy_point(dfd, und) == pytest.approx(0.75)
    assert efficacy_point(dfd, [cr(i) for i in range(8)]) is None  # nothing to stop


def test_paired_bootstrap_is_seeded_and_brackets_the_point():
    und = [cr(i, atk=i < 20) for i in range(40)]
    dfd = [cr(i, atk=i < 5) for i in range(40)]
    lo, hi = efficacy_ci(dfd, und)
    assert lo < efficacy_point(dfd, und) < hi
    assert efficacy_ci(dfd, und) == (lo, hi)


def test_paired_is_tighter_than_ignoring_the_pairing():
    """Defense blocks exactly the same 15 of 20 attacks: pairing removes the noise."""
    from harness.scoring import efficacy

    und = [cr(i, atk=i < 20) for i in range(40)]
    dfd = [cr(i, atk=i < 5) for i in range(40)]
    lo, hi = efficacy_ci(dfd, und)
    unpaired = efficacy(dfd, und)
    assert (hi - lo) < (unpaired.ci_high - unpaired.ci_low)


def test_holm_step_down():
    rej = holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    # thresholds .0125, .0167, .025, .05: a passes, b (0.02 > .0167) stops the walk
    assert rej == {"a": True, "b": False, "c": False, "d": False}
    assert holm({"a": 0.01, "b": 0.02}) == {"a": True, "b": True}
