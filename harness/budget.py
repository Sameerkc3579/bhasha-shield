"""Hard spend cap.

The grid is ~7,500 agent runs and it will be re-run more than once. Without a
cap that actually halts execution, a bad loop spends the project's entire API
budget overnight. This module is that cap, and it is checked before every call,
not after.

Prices are $ per million tokens, verified 2026-09-16. Cache multipliers follow
Anthropic's published model: a cache write costs ~1.25x the input rate, a cache
read ~0.1x. Caching matters enormously here -- every run in the grid resends the
same system prompt and tool definitions, so the cached prefix is most of the
input.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

# ($ per 1M input tokens, $ per 1M output tokens)
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

# plan.md section 6: one cheap fast model for the full grid, one frontier model
# for a 20% stratified slice. Override with AGENTFW_PRIMARY_MODEL.
PRIMARY_MODEL = os.environ.get("AGENTFW_PRIMARY_MODEL", "claude-haiku-4-5")
SPOTCHECK_MODEL = os.environ.get("AGENTFW_SPOTCHECK_MODEL", "claude-opus-5")


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would breach the cap."""


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def cost_usd(model: str, usage: Usage) -> float:
    """Dollar cost of one API response.

    Unknown models raise rather than silently costing zero -- a zero-cost model
    would make the cap useless exactly when a typo introduced it.
    """
    if model not in PRICES:
        raise KeyError(
            f"No price for model {model!r}. Add it to PRICES rather than "
            f"letting it bill as free."
        )
    in_rate, out_rate = PRICES[model]
    return (
        usage.input_tokens * in_rate
        + usage.cache_creation_input_tokens * in_rate * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * in_rate * CACHE_READ_MULTIPLIER
        + usage.output_tokens * out_rate
    ) / 1_000_000


class BudgetState(BaseModel):
    phase: str
    cap_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    by_model: dict[str, float] = {}


class Budget:
    """Per-phase spend tracker that persists across process restarts.

    State lives on disk so that killing and resuming a grid run does not reset
    the cap back to zero -- which is the failure mode that actually empties an
    account.
    """

    def __init__(self, phase: str, cap_usd: float, path: Path | str | None = None):
        self.path = Path(path or Path(".budget") / f"{phase}.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.state = BudgetState.model_validate_json(self.path.read_text())
            self.state.cap_usd = cap_usd  # allow raising the cap between runs
        else:
            self.state = BudgetState(phase=phase, cap_usd=cap_usd)
            self._flush()

    @property
    def remaining(self) -> float:
        return self.state.cap_usd - self.state.spent_usd

    @property
    def fraction_used(self) -> float:
        return self.state.spent_usd / self.state.cap_usd if self.state.cap_usd else 1.0

    def check(self, estimated_usd: float = 0.0) -> None:
        """Call before every request. Raises rather than overspending."""
        if self.state.spent_usd + estimated_usd > self.state.cap_usd:
            raise BudgetExceeded(
                f"phase {self.state.phase!r}: ${self.state.spent_usd:.2f} spent "
                f"of ${self.state.cap_usd:.2f} cap; refusing a call estimated at "
                f"${estimated_usd:.4f}. Raise the cap deliberately or de-scope "
                f"(implementation-plan.md, de-scoping ladder)."
            )

    def record(self, model: str, usage: Usage) -> float:
        c = cost_usd(model, usage)
        self.state.spent_usd += c
        self.state.calls += 1
        self.state.by_model[model] = self.state.by_model.get(model, 0.0) + c
        self._flush()
        return c

    def warning(self) -> str | None:
        if self.fraction_used >= 0.90:
            return (
                f"BUDGET {self.fraction_used:.0%} of ${self.state.cap_usd:.2f} used "
                f"in phase {self.state.phase!r}. Checkpoint now."
            )
        return None

    def _flush(self) -> None:
        self.path.write_text(self.state.model_dump_json(indent=2))

    def __str__(self) -> str:
        return (
            f"[{self.state.phase}] ${self.state.spent_usd:.3f} / "
            f"${self.state.cap_usd:.2f} over {self.state.calls} calls"
        )


def estimate_grid_cost(
    *,
    n_cases: int,
    n_langs: int,
    n_defenses: int,
    model: str,
    tokens_in_per_run: int = 8000,
    tokens_out_per_run: int = 700,
    cache_hit_fraction: float = 0.8,
) -> float:
    """Back-of-envelope cost for a grid, for sizing a cap before spending.

    `cache_hit_fraction` matters more than anything else here: the system prompt
    and tool definitions are identical across every run, so if caching is
    working most input tokens bill at a tenth of the rate. If a dry run shows
    cache_read_input_tokens near zero, stop and fix that before running the grid
    -- it is roughly a 3x cost difference.
    """
    runs = n_cases * n_langs * n_defenses
    cached = int(tokens_in_per_run * cache_hit_fraction)
    usage = Usage(
        input_tokens=tokens_in_per_run - cached,
        cache_read_input_tokens=cached,
        output_tokens=tokens_out_per_run,
    )
    return cost_usd(model, usage) * runs


if __name__ == "__main__":
    # Sizing the Phase 5 cap (implementation-plan.md).
    for model in (PRIMARY_MODEL, SPOTCHECK_MODEL):
        full = estimate_grid_cost(
            n_cases=150, n_langs=5, n_defenses=9, model=model
        )
        print(f"{model:20s} full grid (150x5x9 = 6750 runs): ${full:,.2f}")
    print(
        f"{SPOTCHECK_MODEL:20s} 20% slice: "
        f"${estimate_grid_cost(n_cases=30, n_langs=5, n_defenses=9, model=SPOTCHECK_MODEL):,.2f}"
    )
