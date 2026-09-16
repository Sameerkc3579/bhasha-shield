"""The Defense contract.

Frozen in Phase 0 alongside corpus/schema.py. Every defense in Classes A, B and
C plugs in here, which is what lets the grid treat them uniformly.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from corpus.schema import Lang, ToolOutput

# A = text classifiers, B = prompting defenses, C = architectural / policy.
DefenseClass = Literal["A", "B", "C"]


class RunContext(BaseModel):
    """What a defense is allowed to know about the run in progress.

    Deliberately small. A defense that needs more than this is reaching into the
    agent loop, which makes it unportable and its numbers incomparable.

    `lang` is included because per-language calibration (contribution C5) needs
    it -- but note that a defense reading `lang` is assuming the deployment
    knows the user's language, which is an assumption worth stating in the
    paper rather than hiding in code.
    """

    case_id: str
    lang: Lang
    model: str
    user_task: str                       # trusted
    tool_call_index: int = 0
    prior_tool_names: list[str] = []


class Verdict(BaseModel):
    """A defense's decision about one tool output.

    `score` is the continuous signal used to sweep thresholds and draw the
    frontier. Class C defenses return None: an allowlist has no dial. That
    asymmetry is a finding, not a gap -- an architectural defense cannot be
    tuned to trade security for utility, which is either its best or its worst
    property depending on the deployment.
    """

    block: bool
    score: float | None = None
    reason: str = ""
    latency_ms: float = 0.0
    extra_tokens: int = 0
    # Set by defenses that rewrite rather than reject (spotlighting, sanitising).
    rewritten_content: str | None = None


@runtime_checkable
class Defense(Protocol):
    name: str
    klass: DefenseClass

    def inspect(self, out: ToolOutput, ctx: RunContext) -> Verdict: ...


class NullDefense:
    """The undefended baseline. Its numbers are the denominator for efficacy."""

    name = "undefended"
    klass: DefenseClass = "C"

    def inspect(self, out: ToolOutput, ctx: RunContext) -> Verdict:
        return Verdict(block=False, score=None, reason="no defense")
