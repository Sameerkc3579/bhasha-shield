"""Paired statistics for the grid -- every comparison in the paper goes through here.

Every condition runs the *same* case ids, so outcomes are paired: a case either
flips between two conditions or it does not. That makes McNemar's exact test
the right significance test and a case-resampling bootstrap the right interval.
Treating the columns as independent samples (as `harness.scoring.efficacy`'s
bootstrap does) throws the pairing away and gives intervals that are too wide;
keep that one for continuity, quote these.

No SciPy, like `analysis/power.py`: exact binomial tails are a few lines, and
every number stays auditable.

Pairing keys: a case is identified by `case_id` alone. A language condition is
a different *run* of the same case, so cross-language comparisons pair `en`
case X with `hi` case X -- the case_id carries no language.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb
from typing import Callable, Hashable, Mapping, Sequence

from harness.scoring import CaseResult


@dataclass(frozen=True)
class Paired:
    """A 2x2 paired comparison, reduced to what McNemar needs."""

    only_a: int   # outcome true under A, false under B
    only_b: int   # outcome true under B, false under A
    both: int
    neither: int
    p: float      # exact two-sided McNemar

    @property
    def n(self) -> int:
        return self.only_a + self.only_b + self.both + self.neither

    def __str__(self) -> str:
        return f"{self.only_a} vs {self.only_b} discordant of {self.n}, exact McNemar p={self.p:.4g}"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: binomial(b+c, 0.5) tail, doubled, capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(a: Mapping[Hashable, bool], b: Mapping[Hashable, bool]) -> Paired:
    """Compare two outcome maps on the keys they share. Unshared keys are dropped."""
    keys = a.keys() & b.keys()
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    both = sum(a[k] and b[k] for k in keys)
    return Paired(only_a, only_b, both, len(keys) - only_a - only_b - both,
                  mcnemar_exact(only_a, only_b))


def outcomes(results: Sequence[CaseResult], field: str, *,
             achievable_only: bool = False) -> dict[str, bool]:
    """case_id -> outcome, skipping errored runs (and unachievable ones if asked)."""
    return {
        r.case_id: bool(getattr(r, field))
        for r in results
        if not r.errored and (r.attacker_task_achievable or not achievable_only)
    }


def compare_attack_success(a: Sequence[CaseResult], b: Sequence[CaseResult]) -> Paired:
    """Attack success under A vs B, on cases achievable and live in both (uASR's set)."""
    return paired(outcomes(a, "attacker_task_success", achievable_only=True),
                  outcomes(b, "attacker_task_success", achievable_only=True))


def compare_utility(a: Sequence[CaseResult], b: Sequence[CaseResult]) -> Paired:
    """User-task success under A vs B -- the utility side of the frontier."""
    return paired(outcomes(a, "user_task_success"), outcomes(b, "user_task_success"))


def paired_bootstrap(
    a: Sequence[CaseResult],
    b: Sequence[CaseResult],
    stat: Callable[[list[CaseResult], list[CaseResult]], float | None],
    *,
    iters: int = 2000,
    seed: int = 20260916,
) -> tuple[float | None, float | None]:
    """95% percentile interval for stat(A, B), resampling case ids jointly.

    Each draw keeps a case's A and B outcomes together, which is what pairing
    means. Draws where `stat` is undefined (e.g. a zero denominator) are skipped
    and do not count toward the percentiles.
    """
    ia = {r.case_id: r for r in a}
    ib = {r.case_id: r for r in b}
    keys = sorted(ia.keys() & ib.keys())
    if not keys:
        return None, None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iters):
        draw = [rng.choice(keys) for _ in keys]
        v = stat([ia[k] for k in draw], [ib[k] for k in draw])
        if v is not None:
            samples.append(v)
    if not samples:
        return None, None
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]


def _uasr(rs: Sequence[CaseResult]) -> float | None:
    live = [r for r in rs if not r.errored and r.attacker_task_achievable]
    return sum(r.attacker_task_success for r in live) / len(live) if live else None


def efficacy_point(defended: Sequence[CaseResult], undefended: Sequence[CaseResult]) -> float | None:
    """E = 1 - uASR(D)/uASR(undefended); None when the baseline is undefined or zero."""
    d, u = _uasr(defended), _uasr(undefended)
    if d is None or u is None or u == 0:
        return None
    return 1.0 - d / u


def efficacy_ci(defended: Sequence[CaseResult], undefended: Sequence[CaseResult],
                **kw) -> tuple[float | None, float | None]:
    """Paired-bootstrap interval for efficacy. Prefer over scoring.efficacy's CI."""
    return paired_bootstrap(defended, undefended, efficacy_point, **kw)


def holm(pvalues: Mapping[Hashable, float], alpha: float = 0.05) -> dict[Hashable, bool]:
    """Holm-Bonferroni: which hypotheses are rejected at family-wise level `alpha`.

    The grid tests 8 defenses x 5 languages; uncorrected, a few "significant"
    cells would appear by chance alone.
    """
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(order)
    out: dict[Hashable, bool] = {k: False for k in pvalues}
    for i, k in enumerate(order):
        if pvalues[k] <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out
