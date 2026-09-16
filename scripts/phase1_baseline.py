"""Phase 1 steps 2-3: the undefended English baseline.

Produces `uASR(en, undefended)` -- the denominator every efficacy number in the
paper is divided by -- and `Utility(en, undefended)`, the utility ceiling.

    python -m scripts.phase1_baseline --backend local --workers 8

Everything is cached and traced, so an interrupted run resumes for free. Kaggle
sessions get killed; that is expected, not a problem.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from corpus.schema import InjectionCase
from defenses.base import NullDefense, RunContext
from harness.cache import ResponseCache
from harness.runner import AgentLoop, score_case
from harness.scoring import CaseResult, summarise
from harness.trace import RunTrace

CORPUS = Path("corpus/injections/en.jsonl")
OUT = Path("analysis/out")


def build_client(backend: str, model: str | None, base_url: str | None, cache: ResponseCache):
    if backend == "local":
        from harness.local_client import DEFAULT_BASE_URL, DEFAULT_LOCAL_MODEL, OpenAICompatClient

        return OpenAICompatClient(
            model or DEFAULT_LOCAL_MODEL,
            base_url=base_url or DEFAULT_BASE_URL,
            cache=cache,
        )
    from harness.budget import PRIMARY_MODEL
    from harness.runner import AnthropicClient

    return AnthropicClient(model or PRIMARY_MODEL, cache=cache)


def run_one(case: InjectionCase, client, suites, trace: RunTrace, inject: bool) -> CaseResult:
    """One case. `inject` False gives the clean utility run, True the attack run."""
    from agentdojo.functions_runtime import FunctionsRuntime

    suite = suites[case.domain]
    parts = (case.source_case_id or "").split("/")
    user_task = suite.user_tasks[parts[1]]
    injection_task = suite.injection_tasks[parts[2]]

    injections = {}
    if inject:
        injections = {k: injection_task.GOAL for k in suite.get_injection_vector_defaults()}

    env = suite.load_and_inject_default_environment(injections)
    runtime = FunctionsRuntime(suite.tools)
    pre_env = copy.deepcopy(env)

    ctx = RunContext(
        case_id=case.case_id, lang="en", model=client.model, user_task=case.user_task
    )
    loop = AgentLoop(
        client=client, runtime=runtime, env=env,
        defense=NullDefense(), ctx=ctx, trace=trace,
    )
    result = loop.run(case.user_task)

    return score_case(
        suite=suite,
        user_task=user_task,
        injection_task=injection_task if inject else None,
        pre_env=pre_env,
        post_env=env,
        result=result,
        case_id=case.case_id,
        lang="en",
        defense_name="undefended",
        model=client.model,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("local", "anthropic"), default="local")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--workers", type=int, default=8, help="vLLM batches, so raise this")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    args = ap.parse_args()

    if not CORPUS.exists():
        sys.exit(f"{CORPUS} missing -- run `python -m corpus.extract 150` first.")
    cases = [InjectionCase.model_validate_json(l) for l in CORPUS.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        cases = cases[: args.limit]

    cache = ResponseCache()
    client = build_client(args.backend, args.model, args.base_url, cache)

    from agentdojo.task_suite.load_suites import get_suites

    suites = get_suites("v1.2.1")
    print(f"{len(cases)} cases, model {client.model}, {args.workers} workers\n")

    results: dict[str, list[CaseResult]] = {}
    for inject, label in ((True, "attack"), (False, "clean")):
        with RunTrace(
            phase="phase1", lang="en",
            defense=f"undefended-{label}", model=client.model,
        ) as trace:
            out: list[CaseResult] = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(run_one, c, client, suites, trace, inject): c for c in cases
                }
                for i, fut in enumerate(as_completed(futures), 1):
                    case = futures[fut]
                    try:
                        out.append(fut.result())
                    except Exception as e:
                        out.append(
                            CaseResult(
                                case_id=case.case_id, lang="en",
                                defense="undefended", model=client.model,
                                errored=True,
                            )
                        )
                        print(f"  [{i}/{len(cases)}] {case.case_id} raised {type(e).__name__}")
                    if i % 10 == 0:
                        print(f"  [{i}/{len(cases)}] {label}  {cache}")
            results[label] = out

    attack, clean = results["attack"], results["clean"]
    cell = summarise(
        lang="en", defense="undefended", model=client.model,
        attack_results=attack, undefended_results=attack,
    )

    errored = sum(r.errored for r in attack)
    print("\n--- Phase 1: undefended English baseline ---")
    print(f"cases            {len(attack)}   errored {errored}")
    print(f"ASR              {cell.asr}")
    print(f"uASR             {cell.uasr}   (== ASR here: no probe yet, all assumed achievable)")
    print(f"utility (clean)  {summarise(lang='en', defense='undefended', model=client.model, attack_results=clean, undefended_results=clean).benign_utility}")
    print(f"utility (attack) {cell.benign_utility}")
    print(f"mean latency     {cell.mean_latency_ms:.0f} ms")
    print(f"cache            {cache}")

    if errored / max(1, len(attack)) > 0.10:
        print(
            f"\nWARNING: {errored}/{len(attack)} runs errored. Above ~10% the cell is "
            f"not trustworthy -- check the tool-call parser flags in docs/KAGGLE.md "
            f"before reading anything into these numbers."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1_en_baseline.json").write_text(
        json.dumps(
            {
                "cell": cell.model_dump(),
                "attack": [r.model_dump() for r in attack],
                "clean": [r.model_dump() for r in clean],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT / 'phase1_en_baseline.json'}")
    print("\nNext: python -m scripts.phase2_probe  (the go/no-go gate)")
    return 1 if errored / max(1, len(attack)) > 0.10 else 0


if __name__ == "__main__":
    raise SystemExit(main())
