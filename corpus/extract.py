"""Turn the vendored AgentDojo suites into `InjectionCase` records.

AgentDojo stores a security case as a (user task, injection task) pair: the
legitimate thing the user asked for, and the thing the attacker wants instead.
Its `BaseInjectionTask.GOAL` field is already "precise and LLM-understandable"
because attacks are generated from it -- which makes it directly usable as the
capability probe's instruction, so C1 needs no hand-authored attacker goals for
the upstream cases.

Nothing here calls a model, so it runs free and offline.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from corpus.schema import InjectionCase

DEFAULT_BENCHMARK_VERSION = "v1.2.1"


def _ground_truth_functions(task: Any, env: Any) -> list[str]:
    """Function names a task's reference solution calls.

    Used for stratification and for attack-goal typing. Returns [] rather than
    raising: a suite whose ground truth needs a live environment should degrade
    to an unstratified case, not kill extraction.
    """
    try:
        calls = task.ground_truth(env)
    except Exception:
        return []
    out: list[str] = []
    for c in calls:
        name = getattr(c, "function", None)
        out.append(name if isinstance(name, str) else str(name))
    return out


def classify_goal(functions: Sequence[str], goal_text: str) -> str:
    """Coarse attacker-goal type for stratification.

    Prefers the attacker's reference solution -- what it actually *calls* -- over
    keywords in the prose, because function names are invariant while prose is
    not.

    Some injection tasks legitimately have no ground-truth calls: their success
    is judged on the model's output rather than on environment state (AgentDojo
    checks these in `security()`). Roughly a third of cases are in that group,
    so those fall back to English prose keywords.

    That fallback does not break cross-language comparability, because **the
    label is assigned once on the English case and inherited by every
    translation** (see `inherit_goal_type`). It is never recomputed on
    translated text, which is what would make the language columns diverge.
    """
    joined = " ".join(functions).lower()
    if functions:
        if any(k in joined for k in ("send_money", "schedule_transaction", "update_scheduled")):
            return "unauthorised-transaction"
        if any(k in joined for k in ("send_email", "send_direct_message", "send_channel_message", "post")):
            return "exfiltration-via-message"
        if any(k in joined for k in ("delete", "remove")):
            return "destructive-action"
        if any(k in joined for k in ("update", "set_", "add_", "create", "append")):
            return "state-tampering"
        if any(k in joined for k in ("get_", "read_", "search", "list_")):
            return "information-disclosure"
        return "other"

    # Prose fallback, English source text only.
    t = goal_text.lower()
    if any(k in t for k in ("send money", "transaction", "transfer", "iban", "payment")):
        return "unauthorised-transaction"
    if any(k in t for k in ("email the content", "send the content", "email me", "send an email", "send individual emails", "forward")):
        return "exfiltration-via-message"
    if any(k in t for k in ("say that", "tell the user", "recommend", "convince", "claim that")):
        return "content-manipulation"
    if any(k in t for k in ("delete", "remove", "erase")):
        return "destructive-action"
    if any(k in t for k in ("change", "update", "set ", "add ")):
        return "state-tampering"
    return "unclassified"


def inherit_goal_type(english_cases: Sequence[InjectionCase]) -> dict[str, str]:
    """source_case_id -> attacker_goal_type, fixed on the English corpus.

    Phase 3 stamps translated cases from this map instead of re-running
    `classify_goal` on Hindi or Bengali text. Recomputing per language would let
    the stratification drift between language columns and quietly destroy their
    comparability.
    """
    return {
        (c.source_case_id or c.case_id): c.attacker_goal_type for c in english_cases
    }


def enumerate_cases(
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION,
    suites: Iterable[str] | None = None,
) -> list[InjectionCase]:
    """Every (user task, injection task) pair in the vendored suites, as English cases."""
    from agentdojo.task_suite.load_suites import get_suites  # local: heavy import

    all_suites = get_suites(benchmark_version)
    wanted = set(suites) if suites else set(all_suites)

    cases: list[InjectionCase] = []
    for suite_name, suite in all_suites.items():
        if suite_name not in wanted:
            continue
        try:
            env = suite.load_and_inject_default_environment({})
        except Exception:
            env = None

        for user_task in suite.user_tasks.values():
            user_fns = _ground_truth_functions(user_task, env) if env is not None else []
            for inj_task in suite.injection_tasks.values():
                inj_fns = _ground_truth_functions(inj_task, env) if env is not None else []
                cases.append(
                    InjectionCase(
                        case_id=f"{suite_name}__{user_task.ID}__{inj_task.ID}",
                        lang="en",
                        domain=suite_name,
                        user_task=user_task.PROMPT,
                        attacker_goal_nl=inj_task.GOAL,
                        attacker_goal_type=classify_goal(inj_fns, inj_task.GOAL),
                        payload="",  # filled by the attack, per defense config
                        injected_into_tool="",  # resolved at run time from the vector
                        tool_call_depth=max(1, len(user_fns)),
                        source_case_id=f"{suite_name}/{user_task.ID}/{inj_task.ID}",
                    )
                )
    return cases


# --------------------------------------------------------------------------
# Stratified sampling (implementation-plan.md, Phase 1 step 1)
# --------------------------------------------------------------------------

def stratum_of(case: InjectionCase) -> tuple[str, str, str]:
    """(domain, attacker goal type, depth band).

    Depth is banded rather than exact: exact tool-call counts produce dozens of
    near-empty strata and the allocation degenerates into an arbitrary sample.
    """
    depth = "shallow" if case.tool_call_depth <= 1 else "deep" if case.tool_call_depth >= 3 else "medium"
    return (case.domain, case.attacker_goal_type, depth)


def stratified_sample(
    cases: Sequence[InjectionCase],
    n: int,
    seed: int = 20260916,
) -> list[InjectionCase]:
    """Largest-remainder allocation across strata, then a seeded draw within each.

    Proportional allocation with the remainder given to the largest strata, so
    the subset keeps the benchmark's shape instead of over-representing whatever
    happens to be alphabetically first. Seeded, and the seed goes in the paper.
    """
    if n >= len(cases):
        return list(cases)

    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[InjectionCase]] = defaultdict(list)
    for c in cases:
        buckets[stratum_of(c)].append(c)

    total = len(cases)
    exact = {k: len(v) * n / total for k, v in buckets.items()}
    alloc = {k: int(v) for k, v in exact.items()}

    # Every non-empty stratum gets at least one case, so rare goal types are not
    # silently dropped -- they are often the interesting ones.
    for k in buckets:
        if alloc[k] == 0:
            alloc[k] = 1

    # Largest-remainder adjustment to hit n exactly.
    def _fix(target: int) -> None:
        while sum(alloc.values()) != target:
            diff = target - sum(alloc.values())
            order = sorted(
                buckets,
                key=lambda k: exact[k] - alloc[k],
                reverse=diff > 0,
            )
            for k in order:
                if diff > 0 and alloc[k] < len(buckets[k]):
                    alloc[k] += 1
                    break
                if diff < 0 and alloc[k] > 1:
                    alloc[k] -= 1
                    break
            else:
                return  # cannot reach target without emptying a stratum

    _fix(n)

    out: list[InjectionCase] = []
    for k, bucket in sorted(buckets.items()):
        take = min(alloc[k], len(bucket))
        out.extend(rng.sample(sorted(bucket, key=lambda c: c.case_id), take))
    return sorted(out, key=lambda c: c.case_id)


def stratification_table(cases: Sequence[InjectionCase]) -> list[tuple[tuple[str, str, str], int]]:
    """Counts per stratum -- goes in the paper appendix."""
    return sorted(Counter(stratum_of(c) for c in cases).items())


if __name__ == "__main__":
    import json
    import sys

    cases = enumerate_cases()
    print(f"enumerated {len(cases)} security cases from AgentDojo {DEFAULT_BENCHMARK_VERSION}")

    by_domain = Counter(c.domain for c in cases)
    by_goal = Counter(c.attacker_goal_type for c in cases)
    print("\nby domain:")
    for k, v in sorted(by_domain.items()):
        print(f"  {k:12s} {v:5d}")
    print("\nby attacker goal type:")
    for k, v in sorted(by_goal.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:5d}")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    subset = stratified_sample(cases, n)
    print(f"\nstratified subset: {len(subset)} cases over {len(stratification_table(subset))} strata")
    print("\nsubset by goal type:")
    for k, v in sorted(Counter(c.attacker_goal_type for c in subset).items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:5d}")

    out = "corpus/injections/en.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for c in subset:
            fh.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")
