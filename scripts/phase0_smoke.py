"""Phase 0 exit criterion: one case runs, traced, scored, cached, under budget.

    python -m scripts.phase0_smoke --fake     # no key, no spend: proves the wiring
    python -m scripts.phase0_smoke            # live, needs credentials, costs ~$0.005

`--fake` replays the attacker's own ground-truth calls through the real loop,
real environment and real scorers. It exercises everything except the HTTP call
to the model, so the live run is the only step that needs a key.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from corpus.schema import InjectionCase
from defenses.base import NullDefense, RunContext
from harness.budget import Budget, Usage
from harness.cache import ResponseCache
from harness.runner import AgentLoop, AgentDojoRunner, ModelResponse, ToolCall, score_case
from harness.trace import RunTrace

CORPUS = Path("corpus/injections/en.jsonl")


def load_case(index: int = 0) -> InjectionCase:
    if not CORPUS.exists():
        sys.exit(f"{CORPUS} missing -- run `python -m corpus.extract 150` first.")
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    return InjectionCase.model_validate_json(lines[index])


class GroundTruthClient:
    """Replays the attacker's reference solution. Offline stand-in for a model."""

    model = "fake-ground-truth"

    def __init__(self, calls):
        self.calls = calls
        self.done = False

    def complete(self, *, system, messages, tools):
        if self.done:
            return ModelResponse(text="finished", stop_reason="end_turn", usage=Usage(output_tokens=2))
        self.done = True
        return ModelResponse(
            text="",
            tool_calls=[
                ToolCall(id=f"t{i}", name=c.function, args=dict(c.args))
                for i, c in enumerate(self.calls)
            ],
            stop_reason="tool_use",
            usage=Usage(input_tokens=120, output_tokens=30),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="no model at all; replays ground truth")
    ap.add_argument(
        "--backend",
        choices=("local", "anthropic"),
        default="local",
        help="local = open-weight model served by vLLM (default, free)",
    )
    ap.add_argument("--base-url", default=None, help="vLLM endpoint for --backend local")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cap", type=float, default=1.00, help="phase budget cap in USD")
    args = ap.parse_args()

    case = load_case(args.index)
    print(f"case      {case.case_id}")
    print(f"domain    {case.domain}   lang {case.lang}   goal {case.attacker_goal_type}")
    print(f"user task {case.user_task[:78]}")
    print(f"attacker  {case.attacker_goal_nl[:78]}")
    print()

    budget = Budget("phase0", cap_usd=args.cap)
    cache = ResponseCache()

    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suites

    suites = get_suites("v1.2.1")
    suite = suites[case.domain]
    parts = (case.source_case_id or "").split("/")
    user_task = suite.user_tasks[parts[1]]
    injection_task = suite.injection_tasks[parts[2]]

    vectors = suite.get_injection_vector_defaults()
    env = suite.load_and_inject_default_environment(
        {k: injection_task.GOAL for k in vectors}
    )
    runtime = FunctionsRuntime(suite.tools)
    pre_env = copy.deepcopy(env)

    if args.fake:
        client = GroundTruthClient(injection_task.ground_truth(env))
        model_name = client.model
    elif args.backend == "local":
        try:
            from harness.local_client import (
                DEFAULT_BASE_URL,
                DEFAULT_LOCAL_MODEL,
                OpenAICompatClient,
            )

            client = OpenAICompatClient(
                args.model or DEFAULT_LOCAL_MODEL,
                base_url=args.base_url or DEFAULT_BASE_URL,
                cache=cache,
            )
            model_name = client.model
        except Exception as e:
            print(f"could not build a local client: {type(e).__name__}: {e}")
            print("\nStart the vLLM server first -- see docs/KAGGLE.md.")
            print("  python -m scripts.phase0_smoke --fake   # verifies wiring meanwhile")
            return 2
    else:
        try:
            from harness.budget import PRIMARY_MODEL
            from harness.runner import AnthropicClient

            client = AnthropicClient(
                args.model or PRIMARY_MODEL, budget=budget, cache=cache
            )
            model_name = client.model
        except Exception as e:
            print(f"could not build a live client: {type(e).__name__}: {e}")
            print("\nSet credentials first:")
            print("  ant auth login          (SDK picks up the profile automatically)")
            print("  or cp .env.example .env and fill in ANTHROPIC_API_KEY")
            print("\nMeanwhile: python -m scripts.phase0_smoke --fake")
            return 2

    with RunTrace(phase="phase0", lang=case.lang, defense="undefended", model=model_name) as trace:
        ctx = RunContext(
            case_id=case.case_id, lang=case.lang, model=model_name, user_task=case.user_task
        )
        loop = AgentLoop(
            client=client,
            runtime=runtime,
            env=env,
            defense=NullDefense(),
            ctx=ctx,
            trace=trace,
        )
        result = loop.run(case.user_task)

        scored = score_case(
            suite=suite,
            user_task=user_task,
            injection_task=injection_task,
            pre_env=pre_env,
            post_env=env,
            result=result,
            case_id=case.case_id,
            lang=case.lang,
            defense_name="undefended",
            model=model_name,
        )

    print("--- result ---")
    print(f"steps            {result.steps}")
    print(f"tool calls       {[tc.name for tc in result.tool_calls]}")
    print(f"errored          {result.errored}  {result.error}")
    print(f"user task ok     {scored.user_task_success}")
    print(f"attack succeeded {scored.attacker_task_success}")
    print(f"latency          {result.latency_ms:.0f} ms")
    print(f"tokens           in={result.usage.input_tokens} out={result.usage.output_tokens} "
          f"cache_read={result.usage.cache_read_input_tokens}")
    print()
    print(f"trace            {trace.path}  ({len(RunTrace.read(trace.path))} events)")
    print(f"cache            {cache}")
    print(f"budget           {budget}")

    # An errored run must never report the exit criterion as met. A harness that
    # prints success over a failure is how a broken grid gets trusted for a week.
    if result.errored or scored.errored:
        print(f"\nFAILED: run errored -- {result.error}")
        # The SDK resolves credentials lazily, so a missing key surfaces here at
        # call time rather than at construction. Say what to do about it.
        low = result.error.lower()
        if "connection" in low or "connect" in low or "refused" in low:
            print("\nNo model server reachable. Start vLLM first -- see docs/KAGGLE.md:")
            print("  python -m vllm.entrypoints.openai.api_server \\")
            print("    --model Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice \\")
            print("    --tool-call-parser hermes --port 8000")
            print("\nTo verify the wiring with no model at all:")
            print("  python -m scripts.phase0_smoke --fake")
        elif "authentication" in low or "api_key" in low:
            print("\nNo credentials. Set them with either:")
            print("  ant auth login                     (SDK reads the profile automatically)")
            print("  cp .env.example .env               (then fill in ANTHROPIC_API_KEY)")
            print("\nTo verify the wiring without a key:")
            print("  python -m scripts.phase0_smoke --fake")
        print("\nPhase 0 exit criterion NOT met.")
        return 1

    if not args.fake and result.usage.cache_read_input_tokens == 0:
        print(
            "\nNOTE: cache_read_input_tokens is 0. Expected on the very first call "
            "(nothing cached yet). If it stays 0 across repeated runs, stop and fix "
            "it before the grid -- roughly a 3x cost difference."
        )

    print("\nPHASE 0 EXIT: one case ran, traced, scored, cached, under budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
