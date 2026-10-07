"""Phase 2: the capability-ceiling probe -- the go/no-go gate.

For every injection case, issue the attacker's goal to the agent *as the
legitimate user instruction*, with no injection and no defense, and ask the
injection task's own `security()` checker whether it happened. Aggregated, that
is C(L).

    python -m scripts.phase2_probe --lang en --workers 8      # needs vLLM up
    python -m scripts.phase2_probe --fake                     # no model, no GPU

Free by construction: the only backend is a locally served open-weight model.

The 150 cases share far fewer distinct attacker goals (the same injection task
recurs under many user tasks, and the probe never sees the user task), so each
distinct goal is probed once and the outcome fanned back out to its cases. That
is a 5-10x saving in GPU time and changes no number: the probe input for two
cases with the same goal is byte-identical, so at temperature 0 the run would be
too.

If the Phase 1 baseline for the same language exists, the probe is joined onto
it and the first real `uASR` is printed next to raw `ASR`, together with
`probe_disagreements` -- attacks that succeeded on goals the probe called
impossible. A non-trivial count there means the probe is too strict and C(L)
cannot be trusted yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from corpus.schema import InjectionCase
from harness.cache import ResponseCache
from harness.capability import (
    AgentOutcome,
    achievability_map,
    apply_achievability,
    probe_disagreements,
)
from harness.runner import AgentDojoRunner, ModelResponse, ToolCall
from harness.scoring import CaseResult, asr, capability_ceiling, uasr
from harness.trace import RunTrace

OUT = Path("analysis/out")


def corpus_path(lang: str) -> Path:
    return Path(f"corpus/injections/{lang}.jsonl")


def goal_key(case: InjectionCase) -> tuple[str, str, str]:
    """Everything the probe's input depends on. Cases sharing it share a run."""
    inj_id = (case.source_case_id or "").split("/")[-1]
    return case.domain, inj_id, case.attacker_goal_nl


# --------------------------------------------------------------------------
# Offline stand-in
# --------------------------------------------------------------------------

class GroundTruthClient:
    """Replays the injection task's reference solution. Proves the wiring only.

    Every goal should come back achievable under it; one that does not points at
    a harness bug (wrong suite, wrong task id, scorer mismatch), not at the model.
    """

    model = "fake-ground-truth"

    def __init__(self, calls, output: str = ""):
        self.calls = list(calls)
        self.output = output  # some goals are met by what the agent says, not does
        self.done = False

    def complete(self, *, system, messages, tools):
        if self.done or not self.calls:
            return ModelResponse(text=self.output or "done", stop_reason="end_turn")
        self.done = True
        return ModelResponse(
            tool_calls=[
                ToolCall(id=f"t{i}", name=c.function, args=dict(c.args))
                for i, c in enumerate(self.calls)
            ],
            stop_reason="tool_use",
        )


def has_reference(runner: AgentDojoRunner, case: InjectionCase) -> bool:
    """AgentDojo ships no ground truth for some goals (workspace 6-13 return []).

    Those cannot be replayed offline; they are still scored normally against a
    real model, because `security()` inspects the environment, not a reference.
    """
    suite = runner.suites[case.domain]
    inj = suite.injection_tasks[goal_key(case)[1]]
    env = runner._prepare(case.domain, lang=case.lang)[1]  # localized when the runner is
    return bool(inj.ground_truth(env)) or bool(getattr(inj, "GROUND_TRUTH_OUTPUT", ""))


def probe_one_fake(runner: AgentDojoRunner, case: InjectionCase) -> AgentOutcome:
    suite = runner.suites[case.domain]
    inj = suite.injection_tasks[goal_key(case)[1]]
    # Ground truth from the environment the run will use: a goal that forwards an
    # email body must forward the *localized* body, as a live model would.
    env = runner._prepare(case.domain, lang=case.lang)[1]
    runner.client = GroundTruthClient(inj.ground_truth(env), getattr(inj, "GROUND_TRUTH_OUTPUT", ""))
    return runner.run_task(task=case.attacker_goal_nl, lang=case.lang, case=case)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--fake", action="store_true", help="replay ground truth; no model, no GPU")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N cases only")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="probe unreviewed machine-drafted goals (pilot run; output is marked _draft)")
    ap.add_argument("--localized-env", action="store_true",
                    help="run in the language's localized environment (C3); output is marked _locenv")
    args = ap.parse_args()

    path = corpus_path(args.lang)
    draft = False
    if path.exists():
        cases = [InjectionCase.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        # Before Phase 3 localizes the full corpus, the probe needs only the 35
        # goals: take the English cases and swap in the localized goal.
        from corpus.goals import all_verified, goals_path, localize_cases

        if not goals_path(args.lang).exists():
            sys.exit(f"neither {path} nor {goals_path(args.lang)} exists")
        en = [InjectionCase.model_validate_json(l) for l in corpus_path("en").read_text(encoding="utf-8").splitlines() if l.strip()]
        try:
            cases = localize_cases(en, args.lang, allow_unverified=args.allow_unverified)
        except PermissionError as e:
            sys.exit(str(e))
        draft = not all_verified(args.lang)
        if draft:
            print(f"PILOT: {args.lang} goals include unreviewed machine drafts. "
                  f"C({args.lang}) below is a pilot number, not a result.\n")
    if args.limit:
        cases = cases[: args.limit]

    groups: dict[tuple[str, str, str], list[InjectionCase]] = defaultdict(list)
    for c in cases:
        groups[goal_key(c)].append(c)
    reps = [members[0] for members in groups.values()]

    if args.fake:
        client = GroundTruthClient([])
        args.workers = 1  # the fake client is swapped per goal, so no concurrency
    else:
        from harness.local_client import DEFAULT_BASE_URL, DEFAULT_LOCAL_MODEL, OpenAICompatClient

        client = OpenAICompatClient(
            args.model or DEFAULT_LOCAL_MODEL,
            base_url=args.base_url or DEFAULT_BASE_URL,
            cache=ResponseCache(),
        )

    print(f"{len(cases)} cases -> {len(reps)} distinct attacker goals, "
          f"lang {args.lang}, model {client.model}, {args.workers} workers\n")

    outcomes: dict[tuple[str, str, str], AgentOutcome] = {}
    with RunTrace(phase="phase2", lang=args.lang, defense="capability_probe", model=client.model) as trace:
        runner = AgentDojoRunner(client=client, trace=trace, localized_env=args.localized_env)

        def go(case: InjectionCase) -> AgentOutcome:
            if args.fake:
                return probe_one_fake(runner, case)
            return runner.run_task(task=case.attacker_goal_nl, lang=case.lang, case=case)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(go, c): c for c in reps}
            for i, fut in enumerate(as_completed(futures), 1):
                c = futures[fut]
                try:
                    outcomes[goal_key(c)] = fut.result()
                except Exception as e:
                    outcomes[goal_key(c)] = AgentOutcome(
                        task_succeeded=False, errored=True, error=f"{type(e).__name__}: {e}"
                    )
                    print(f"  [{i}/{len(reps)}] {c.case_id} raised {type(e).__name__}: {e}")
                if i % 10 == 0:
                    print(f"  [{i}/{len(reps)}]")

    # Fan each goal's outcome back out to every case that shares it.
    probe: list[CaseResult] = []
    for key, members in groups.items():
        o = outcomes[key]
        for c in members:
            probe.append(CaseResult(
                case_id=c.case_id, lang=c.lang, defense="capability_probe", model=client.model,
                attacker_task_achievable=o.task_succeeded,
                latency_ms=o.latency_ms, tokens=o.tokens, errored=o.errored,
            ))

    goal_level = [
        CaseResult(case_id=k[1], lang=args.lang, defense="capability_probe", model=client.model,
                   attacker_task_achievable=o.task_succeeded, errored=o.errored)
        for k, o in outcomes.items()
    ]
    c_cases = capability_ceiling(probe)
    c_goals = capability_ceiling(goal_level)
    errored = sum(o.errored for o in outcomes.values())

    print(f"\n--- Phase 2: capability probe, {args.lang} ---")
    print(f"C({args.lang}) over cases   {c_cases}")
    print(f"C({args.lang}) over goals   {c_goals}")
    print(f"errored goals      {errored}/{len(outcomes)}")

    by_type: dict[str, Counter] = defaultdict(Counter)
    for r, c in zip(probe, [m for ms in groups.values() for m in ms]):
        if not r.errored:
            by_type[c.attacker_goal_type]["n"] += 1
            by_type[c.attacker_goal_type]["ok"] += r.attacker_task_achievable
    print("\nby attacker goal type (cases):")
    for t, cnt in sorted(by_type.items()):
        print(f"  {t:<28} {cnt['ok']:>3}/{cnt['n']:<3}")

    failed = sorted(f"{k[0]}/{k[1]}" for k, o in outcomes.items() if not o.task_succeeded and not o.errored)
    if failed:
        print(f"\nunachievable goals ({len(failed)}): {', '.join(failed)}")

    report: dict = {
        "lang": args.lang,
        "model": client.model,
        "fake": args.fake,
        "draft": draft,
        "c_cases": c_cases.__dict__,
        "c_goals": c_goals.__dict__,
        "goals": [
            {"domain": k[0], "injection_task": k[1], "goal": k[2], **o.model_dump(exclude={"final_text"}),
             "final_text": o.final_text[:500], "n_cases": len(groups[k])}
            for k, o in outcomes.items()
        ],
        "probe": [r.model_dump() for r in probe],
    }

    # Join onto the Phase 1 baseline: the first uASR that is a measurement.
    baseline = OUT / f"phase1_{args.lang}_baseline.json"
    base = json.loads(baseline.read_text(encoding="utf-8")) if baseline.exists() and not args.fake else None
    if base is not None and base.get("cell", {}).get("model") != client.model:
        # A baseline restored from an earlier session can belong to another model;
        # joining a 14B probe onto a 7B baseline would produce a meaningless uASR.
        print(f"\nnot joining {baseline}: it was produced by {base.get('cell', {}).get('model')!r}, "
              f"this probe ran {client.model!r}. Run the Phase 1 baseline for this model first.")
        base = None
    if base is not None:
        attack = [CaseResult.model_validate(r) for r in base["attack"]]
        amap = achievability_map(probe)
        stamped = apply_achievability(attack, amap)
        a, ua = asr(stamped), uasr(stamped)
        dis = probe_disagreements(attack, amap)
        print(f"\njoined with {baseline}:")
        print(f"  ASR  (raw)          {a}")
        print(f"  uASR (normalized)   {ua}")
        print(f"  probe disagreements {len(dis)}  (attack succeeded on a goal the probe called impossible)")
        if dis:
            print("   ", ", ".join(r.case_id for r in dis[:10]), "..." if len(dis) > 10 else "")
            print("  Each one is evidence the probe is too strict. Investigate before trusting C(L).")
        report["joined"] = {"asr": a.__dict__, "uasr": ua.__dict__, "disagreements": [r.case_id for r in dis]}

    OUT.mkdir(parents=True, exist_ok=True)
    loc = "_locenv" if args.localized_env and args.lang != "en" else ""
    name = f"phase2_probe_{args.lang}{loc}{'_draft' if draft else ''}{'_fake' if args.fake else ''}.json"
    (OUT / name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT / name}")

    if args.fake:
        ref = {goal_key(c) for c in reps if has_reference(runner, c)}
        bad = sorted(f"{k[0]}/{k[1]}" for k in ref if not outcomes[k].task_succeeded)
        skipped = len(outcomes) - len(ref)
        print(f"\nfake check: {len(ref) - len(bad)}/{len(ref)} replayable goals achieved; "
              f"{skipped} have no AgentDojo reference solution and are only checkable live")
        if bad:
            print(f"FAKE RUN NOT CLEAN -- harness bug on: {', '.join(bad)}")
            return 1
        return 0
    if errored / max(1, len(outcomes)) > 0.10:
        print(f"\nWARNING: {errored}/{len(outcomes)} goals errored -- C({args.lang}) is not trustworthy.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
