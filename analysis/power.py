"""How many cases do we actually need?

`implementation-plan.md` Phase 1 step 1 says to justify N with a power
calculation and put it in the appendix, because reviewers ask. This is that
calculation, and it is run before the money is spent rather than after.

Two designs are computed, because they give very different answers:

*Unpaired* -- treat each language column as an independent sample. This is the
conservative calculation and the one a reviewer will run in their head.

*Paired (McNemar)* -- the same case IDs run in every language, so each case is
its own control. Language changes the wording, not the task, so outcomes are
strongly correlated and most pairs are concordant. McNemar conditions on the
discordant pairs only, which is far more powerful. This is the design we
actually have, so it is the one that should size the study -- but it depends on
the discordance rate, which is unknown until Phase 2, so we report a range.

No SciPy: the normal approximations are closed-form and the McNemar power is
simulated, which keeps the dependency list short and the numbers auditable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Standard normal quantiles for the usual alpha/power settings.
Z_ALPHA_2 = 1.959963984540054   # two-sided alpha = 0.05
Z_POWER_80 = 0.8416212335729143  # power = 0.80
Z_POWER_90 = 1.2815515655446004  # power = 0.90


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Unpaired two-proportion design
# --------------------------------------------------------------------------

def unpaired_n(p1: float, p2: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Cases per group to detect p1 vs p2 with a two-sided two-proportion test."""
    if p1 == p2:
        raise ValueError("no effect to detect")
    z_a = Z_ALPHA_2 if alpha == 0.05 else abs(_z_from_p(1 - alpha / 2))
    z_b = Z_POWER_80 if power == 0.80 else Z_POWER_90
    pbar = (p1 + p2) / 2
    num = (
        z_a * math.sqrt(2 * pbar * (1 - pbar))
        + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def unpaired_mde(n: int, p: float = 0.5, power: float = 0.80) -> float:
    """Smallest difference detectable at this n. p=0.5 is the worst case."""
    z_b = Z_POWER_80 if power == 0.80 else Z_POWER_90
    return (Z_ALPHA_2 + z_b) * math.sqrt(2 * p * (1 - p) / n)


def _z_from_p(p: float) -> float:
    """Inverse normal CDF by bisection. Only used for non-default alphas."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------
# Paired design (McNemar)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedScenario:
    """A paired setup described by what actually varies between two languages.

    `p_en` is the rate in the reference language. `delta` is the true difference.
    `discordance` is the share of cases whose outcome differs between the two
    languages -- the only cases McNemar uses. Low discordance means language
    rarely flips an individual case, which is what makes the paired design win.
    """

    p_en: float
    delta: float
    discordance: float


def mcnemar_power(
    scenario: PairedScenario,
    n: int,
    alpha: float = 0.05,
    iters: int = 4000,
    seed: int = 20260916,
) -> float:
    """Simulated power of an exact-ish McNemar test at this n.

    The two discordant cells are split so their difference reproduces `delta`:
        b - c = delta * n, b + c = discordance * n
    """
    b_rate = (scenario.discordance + scenario.delta) / 2
    c_rate = (scenario.discordance - scenario.delta) / 2
    if b_rate < 0 or c_rate < 0:
        raise ValueError(
            f"discordance {scenario.discordance} too small for delta {scenario.delta}; "
            f"|delta| cannot exceed discordance"
        )

    rng = random.Random(seed)
    hits = 0
    for _ in range(iters):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            if u < b_rate:
                b += 1
            elif u < b_rate + c_rate:
                c += 1
        if b + c == 0:
            continue
        # Normal approximation to McNemar's chi-square, continuity-corrected.
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        if stat > 3.841458820694124:  # chi2(1) critical value at alpha = 0.05
            hits += 1
    return hits / iters


def mcnemar_min_n(
    scenario: PairedScenario,
    target_power: float = 0.80,
    lo: int = 20,
    hi: int = 1200,
    **kw,
) -> int:
    """Smallest n reaching the target power, by bisection on the simulation."""
    if mcnemar_power(scenario, hi, **kw) < target_power:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_power(scenario, mid, **kw) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def report(n_planned: int = 150, iters: int = 4000) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"POWER ANALYSIS  (planned n = {n_planned} cases per language)")
    a("=" * 72)
    a("")
    a("Unpaired two-proportion design (conservative; what a reviewer checks)")
    a("-" * 72)
    a(f"  {'baseline':>10} {'effect':>8} {'n needed':>10}")
    for p1, p2 in ((0.50, 0.40), (0.50, 0.35), (0.50, 0.30), (0.30, 0.20), (0.70, 0.55)):
        a(f"  {p1:>10.2f} {p1 - p2:>8.2f} {unpaired_n(p1, p2):>10d}")
    a("")
    a(f"  Minimum detectable effect at n={n_planned}, worst case p=0.5:")
    a(f"    power 0.80 -> {unpaired_mde(n_planned, 0.5, 0.80) * 100:.1f} points")
    a(f"    power 0.90 -> {unpaired_mde(n_planned, 0.5, 0.90) * 100:.1f} points")
    a("")
    a("Paired McNemar design (what we actually have: same cases, every language)")
    a("-" * 72)
    a(f"  {'discordance':>12} {'effect':>8} {'power@' + str(n_planned):>12} {'n for 0.80':>12}")
    for disc in (0.15, 0.25, 0.35, 0.50):
        for delta in (0.10, 0.15):
            if delta > disc:
                continue
            sc = PairedScenario(p_en=0.5, delta=delta, discordance=disc)
            pw = mcnemar_power(sc, n_planned, iters=iters)
            need = mcnemar_min_n(sc, iters=iters)
            a(f"  {disc:>12.2f} {delta:>8.2f} {pw:>12.2f} {need:>12d}")
    a("")
    a("Reading this")
    a("-" * 72)
    a("  Discordance is the share of cases whose outcome flips between two")
    a("  languages. It is unknown until the Phase 2 capability probe measures it.")
    a("  Low discordance favours the paired design; high discordance pushes the")
    a("  requirement back towards the unpaired numbers above.")
    a("")
    a("  Every headline comparison in this study is paired -- the same case IDs")
    a("  run in every language -- so the paired rows are the ones that size it.")
    a("  Report McNemar, not a two-sample z-test, and say so in the methods.")
    a("")
    a("What n costs (full grid: n x 5 languages x 9 defense configs, primary model)")
    a("-" * 72)
    try:
        from harness.budget import PRIMARY_MODEL, SPOTCHECK_MODEL, estimate_grid_cost

        # The spot-check slice is FIXED at 30 cases, not a percentage of n. Its
        # job is to show the finding is not an artefact of one model, not to
        # power comparisons of its own -- and at frontier pricing a slice that
        # scales with n doubles the bill for no extra claim.
        SLICE_CASES = 30
        slice_ = estimate_grid_cost(
            n_cases=SLICE_CASES, n_langs=5, n_defenses=9, model=SPOTCHECK_MODEL
        )
        a(f"  {'n':>6} {'runs':>8} {'primary $':>11} {'slice $':>9} {'total $':>10}")
        for n in (150, 200, 250, 300, 400):
            runs = n * 5 * 9
            primary = estimate_grid_cost(
                n_cases=n, n_langs=5, n_defenses=9, model=PRIMARY_MODEL
            )
            a(f"  {n:>6} {runs:>8} {primary:>11.2f} {slice_:>9.2f} {primary + slice_:>10.2f}")
        a("")
        a(f"  Spot-check slice fixed at {SLICE_CASES} cases on {SPOTCHECK_MODEL} (${slice_:.2f}),")
        a("  independent of n. plan.md section 6 budgeted $60-110 total.")
    except Exception as e:  # pragma: no cover
        a(f"  (cost table unavailable: {e})")
    a("")
    a("Decision")
    a("-" * 72)
    a("  Do NOT resize now. The Phase 2 capability probe measures discordance")
    a("  directly -- it runs the same cases across all five languages and records")
    a("  which ones flip. That is the missing input to this table, it is cheap,")
    a("  and it lands before the expensive Phase 5 grid.")
    a("")
    a("  Rule to apply after Phase 2:")
    a("    discordance <= 0.20  -> n = 150 holds, proceed")
    a("    0.20 < disc <= 0.35  -> raise to n = 250 (+ ~$26, still in budget)")
    a("    discordance > 0.35   -> n = 300+, or narrow the claim to 15-point")
    a("                            effects and say so explicitly in the paper")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(150))
