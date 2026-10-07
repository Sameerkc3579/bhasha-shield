"""The capability-ceiling probe -- contribution C1.

The question this answers, per language:

    If we simply *asked* the agent to do what the attacker wanted, in this
    language, with no injection and no defense in the way, could it?

Where the answer is no, a subsequent "attack failed" result says nothing about
the defense, and scoring.uasr() drops the case rather than crediting it.

Deliberately runs before the expensive grid (Phase 2), because its outcome can
kill the project -- see plan.md section 9.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from pydantic import BaseModel

from corpus.schema import InjectionCase, Lang
from harness.scoring import CaseResult


class AgentOutcome(BaseModel):
    """What running one task through the agent produced."""

    task_succeeded: bool
    tool_calls: list[str] = []
    final_text: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    errored: bool = False
    error: str = ""


class AgentRunner(Protocol):
    """Whatever actually drives the agent.

    Kept abstract so the probe is testable offline with a fake, and so swapping
    AgentDojo for another environment later does not touch this file.
    """

    def run_task(self, *, task: str, lang: Lang, case: InjectionCase) -> AgentOutcome: ...


def probe_case(runner: AgentRunner, case: InjectionCase) -> CaseResult:
    """Run the attacker's goal as a legitimate user instruction."""
    outcome = runner.run_task(task=case.attacker_goal_nl, lang=case.lang, case=case)
    return CaseResult(
        case_id=case.case_id,
        lang=case.lang,
        defense="capability_probe",
        model="probe",
        attacker_task_achievable=outcome.task_succeeded,
        attacker_task_success=False,  # no attack was run; this is the probe
        latency_ms=outcome.latency_ms,
        tokens=outcome.tokens,
        errored=outcome.errored,
    )


def probe_all(runner: AgentRunner, cases: Sequence[InjectionCase]) -> list[CaseResult]:
    return [probe_case(runner, c) for c in cases]


def achievability_map(probe_results: Sequence[CaseResult]) -> dict[tuple[str, str], bool]:
    """(case_id, lang) -> achievable. Joined onto grid results before scoring."""
    return {
        (r.case_id, r.lang): r.attacker_task_achievable
        for r in probe_results
        if not r.errored
    }


def with_existence_proof(
    achievable: dict[tuple[str, str], bool],
    undefended: Sequence[CaseResult],
) -> dict[tuple[str, str], bool]:
    """Probe OR "the undefended attack actually did it".

    Adopted 2026-10-07. A goal the attack achieved is achievable by existence;
    the probe calling it impossible is a probe miss (travel/injection_task_5 in
    English, every probe run so far). Only the *undefended* cell may add cases,
    so every defense in a language is scored on the same set -- adding a
    defended cell's successes would give each defense its own denominator.
    """
    out = dict(achievable)
    for r in undefended:
        if r.attacker_task_success and not r.errored:
            out[(r.case_id, r.lang)] = True
    return out


def apply_achievability(
    results: Sequence[CaseResult],
    achievable: dict[tuple[str, str], bool],
    *,
    default: bool = True,
) -> list[CaseResult]:
    """Stamp grid results with probe outcomes.

    `default=True` is deliberately the conservative choice: an unprobed case
    stays in the scorable set, so a missing probe inflates the apparent attack
    surface rather than quietly shrinking it into a better-looking defense
    number. Erring the other way would recreate the bias this project exists to
    remove.
    """
    out: list[CaseResult] = []
    for r in results:
        stamped = r.model_copy(
            update={"attacker_task_achievable": achievable.get((r.case_id, r.lang), default)}
        )
        out.append(stamped)
    return out


def probe_disagreements(
    results: Sequence[CaseResult],
    achievable: dict[tuple[str, str], bool],
) -> list[CaseResult]:
    """Cases the probe called unachievable that the attack nonetheless achieved.

    Each one is evidence the probe is too strict -- probably because the
    attacker's goal was phrased differently as a user instruction than it was as
    an injected payload. A non-trivial count here invalidates C(L) and must be
    investigated before the grid is trusted, not explained away afterwards.
    """
    return [
        r
        for r in results
        if r.attacker_task_success
        and achievable.get((r.case_id, r.lang), True) is False
    ]
