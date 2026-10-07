"""Tool allowlist with value validation.

Two rules, both decided before any untrusted text exists:

1. **Tool rule.** A side-effecting tool may run only if the trusted planner
   (`planner.py`) put it in the plan. Read tools always run.
2. **Value rule.** Arguments are validated against their declared types and
   against hard limits that need no plan: amounts must be positive numbers and
   at most `max_amount`; email-shaped sink values must be well-formed.

What it does NOT do -- the difference from `camel_lite` -- is ask where a sink
value came from. An injection that persuades the agent to `send_money` to the
attacker passes the allowlist whenever the user's own task also sends money.
That gap is the reason to run both.

`score` is None: an allowlist has no dial (see `Verdict`).
"""

from __future__ import annotations

import re
from typing import Any

from corpus.schema import ToolOutput
from defenses.architectural.planner import plan
from defenses.base import DefenseClass, RunContext, Verdict
from defenses.policy import is_side_effect, sink_values

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Allowlist:
    name = "allowlist"
    klass: DefenseClass = "C"

    def __init__(self, planner: Any, *, max_amount: float = 10_000.0):
        self.planner = planner
        self.max_amount = max_amount
        self._allowed: set[str] | None = None
        self._unbilled = 0

    def bind(self, ctx: RunContext) -> "Allowlist":
        run = type(self)(self.planner, max_amount=self.max_amount)
        run._allowed, run._unbilled = plan(self.planner, ctx)
        return run

    def _bill(self) -> int:
        t, self._unbilled = self._unbilled, 0
        return t

    def _value_problem(self, name: str, args: dict[str, Any]) -> str:
        amount = args.get("amount")
        if amount is not None:
            try:
                a = float(amount)
            except (TypeError, ValueError):
                return f"amount {amount!r} is not a number"
            if not 0 < a <= self.max_amount:
                return f"amount {a} outside (0, {self.max_amount}]"
        for v in sink_values(name, args):
            if "@" in v and not _EMAIL.match(v):
                return f"malformed address {v!r}"
        return ""

    def check_call(self, name: str, args: dict[str, Any], ctx: RunContext) -> Verdict:
        tokens = self._bill()
        if not is_side_effect(name):
            return Verdict(block=False, extra_tokens=tokens)
        if self._allowed is None:
            return Verdict(block=True, reason="planner reply unparseable; side effects refused",
                           extra_tokens=tokens)
        if name not in self._allowed:
            return Verdict(block=True, reason=f"{name} not in the plan {sorted(self._allowed)}",
                           extra_tokens=tokens)
        problem = self._value_problem(name, args)
        if problem:
            return Verdict(block=True, reason=problem, extra_tokens=tokens)
        return Verdict(block=False, reason=f"{name} planned", extra_tokens=tokens)

    def inspect(self, out: ToolOutput, ctx: RunContext) -> Verdict:
        # Bills the planner here too, in case the run's first event is an output.
        return Verdict(block=False, reason="outputs pass; calls are policed",
                       extra_tokens=self._bill())
