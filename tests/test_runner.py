"""The agent loop, the defense hook, and scoring against the real benchmark.

The fake model client replays a scripted list of tool calls, so the loop runs
end to end against genuine AgentDojo environments, tools and scorers with no API
key and no spend. That covers every part of Phase 0's exit criterion except the
live model call itself.
"""

from __future__ import annotations

import copy

import pytest

from corpus.schema import InjectionCase, ToolOutput
from defenses.base import Defense, NullDefense, RunContext, Verdict
from harness.budget import Usage
from harness.runner import (
    SAMPLING_PARAMS_AVAILABLE,
    AgentLoop,
    ModelResponse,
    ToolCall,
    tool_schemas,
)

agentdojo = pytest.importorskip("agentdojo", reason="vendored benchmark not installed")


# ----------------------------------------------------------------- fakes


class FakeClient:
    """Replays scripted turns. Anything past the script is a plain end_turn."""

    model = "fake-model"

    def __init__(self, script: list[ModelResponse]):
        self.script = list(script)
        self.calls: list[dict] = []

    def complete(self, *, system, messages, tools):
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if self.script:
            return self.script.pop(0)
        return ModelResponse(text="done", stop_reason="end_turn", usage=Usage(output_tokens=3))


def turn(*calls: tuple[str, dict], text: str = "") -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=[ToolCall(id=f"t{i}", name=n, args=a) for i, (n, a) in enumerate(calls)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=20),
    )


class BlockAll:
    name = "block-all"
    klass = "A"

    def inspect(self, out: ToolOutput, ctx: RunContext) -> Verdict:
        return Verdict(block=True, score=1.0, reason="test", extra_tokens=5)


class Redactor:
    name = "redactor"
    klass = "A"

    def inspect(self, out: ToolOutput, ctx: RunContext) -> Verdict:
        return Verdict(block=False, score=0.1, rewritten_content="[redacted]")


class RecordingDefense:
    name = "recorder"
    klass = "A"

    def __init__(self):
        self.seen: list[ToolOutput] = []
        self.ctxs: list[RunContext] = []

    def inspect(self, out: ToolOutput, ctx: RunContext) -> Verdict:
        self.seen.append(out)
        self.ctxs.append(ctx)
        return Verdict(block=False, score=0.0)


def banking():
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suites

    suite = get_suites("v1.2.1")["banking"]
    env = suite.load_and_inject_default_environment({})
    return suite, env, FunctionsRuntime(suite.tools)


def ctx_for(lang="en") -> RunContext:
    return RunContext(case_id="c1", lang=lang, model="fake-model", user_task="do a thing")


# ------------------------------------------------------------- schemas


def test_tool_schemas_are_sorted_for_cache_stability():
    """Unstable tool order would invalidate the prompt cache on every call."""
    _, _, runtime = banking()
    names = [t["name"] for t in tool_schemas(runtime.functions)]
    assert names == sorted(names)
    assert len(names) > 3


def test_tool_schemas_carry_an_input_schema():
    _, _, runtime = banking()
    for t in tool_schemas(runtime.functions):
        assert t["input_schema"]["type"] == "object"
        assert "title" not in t["input_schema"]


# ---------------------------------------------------------------- loop


def test_loop_executes_a_tool_and_feeds_the_result_back():
    suite, env, runtime = banking()
    client = FakeClient([turn(("get_balance", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for())
    result = loop.run("What is my balance?")

    assert [tc.name for tc in result.tool_calls] == ["get_balance"]
    assert not result.errored
    # Second model call must contain the tool_result
    second = client.calls[1]["messages"]
    assert second[-1]["role"] == "user"
    assert second[-1]["content"][0]["type"] == "tool_result"


def test_parallel_tool_results_return_in_one_user_message():
    """Splitting them trains the model out of parallel tool calls."""
    suite, env, runtime = banking()
    client = FakeClient([turn(("get_balance", {}), ("get_iban", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for())
    loop.run("balance and iban")

    last_user = client.calls[1]["messages"][-1]
    assert last_user["role"] == "user"
    assert len(last_user["content"]) == 2
    assert all(b["type"] == "tool_result" for b in last_user["content"])


def test_loop_stops_when_the_model_stops_asking_for_tools():
    suite, env, runtime = banking()
    client = FakeClient([ModelResponse(text="all done", stop_reason="end_turn")])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for())
    result = loop.run("hi")
    assert result.final_text == "all done"
    assert result.steps == 1
    assert result.tool_calls == []


def test_max_steps_is_recorded_not_silently_ignored():
    suite, env, runtime = banking()
    client = FakeClient([turn(("get_balance", {})) for _ in range(10)])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for(), max_steps=3)
    result = loop.run("loop forever")
    assert result.steps == 3
    assert "max_steps" in result.error


def test_unknown_tool_becomes_an_error_result_not_a_crash():
    suite, env, runtime = banking()
    client = FakeClient([turn(("no_such_tool", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for())
    result = loop.run("do something impossible")
    assert not result.errored
    block = client.calls[1]["messages"][-1]["content"][0]
    assert block.get("is_error") is True
    assert "ToolNotFoundError" in block["content"]


def test_model_exception_marks_the_run_errored():
    """An API failure must not be scored as 'the attack failed'."""

    class Boom:
        model = "fake-model"

        def complete(self, **kw):
            raise RuntimeError("api exploded")

    suite, env, runtime = banking()
    loop = AgentLoop(client=Boom(), runtime=runtime, env=env, ctx=ctx_for())
    result = loop.run("x")
    assert result.errored
    assert "api exploded" in result.error


# ------------------------------------------------------- defense hook


def test_defense_sees_tool_output_marked_untrusted():
    """Tool output is the channel indirect injection arrives on."""
    suite, env, runtime = banking()
    rec = RecordingDefense()
    client = FakeClient([turn(("get_balance", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for("hi"), defense=rec)
    loop.run("balance")

    assert len(rec.seen) == 1
    assert rec.seen[0].provenance == "untrusted"
    assert rec.seen[0].tool_name == "get_balance"
    assert rec.seen[0].lang == "hi"


def test_blocked_output_is_withheld_but_the_agent_continues():
    """Blocking must still let us measure whether the user's task survives."""
    suite, env, runtime = banking()
    client = FakeClient([turn(("get_balance", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for(), defense=BlockAll())
    result = loop.run("balance")

    assert result.blocked == 1
    content = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert "blocked by block-all" in content
    assert result.steps == 2  # loop continued after the block


def test_rewritten_content_replaces_the_tool_result():
    suite, env, runtime = banking()
    client = FakeClient([turn(("get_balance", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for(), defense=Redactor())
    result = loop.run("balance")

    assert result.blocked == 0
    assert client.calls[1]["messages"][-1]["content"][0]["content"] == "[redacted]"


def test_defense_overhead_tokens_are_counted():
    suite, env, runtime = banking()
    client = FakeClient([turn(("get_balance", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for(), defense=BlockAll())
    result = loop.run("balance")
    # 100 in from the scripted turn, plus the defense's 5 extra tokens
    assert result.usage.input_tokens >= 105


def test_tool_call_index_increments_within_a_turn():
    suite, env, runtime = banking()
    rec = RecordingDefense()
    client = FakeClient([turn(("get_balance", {}), ("get_iban", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for(), defense=rec)
    loop.run("both")
    assert [c.tool_call_index for c in rec.ctxs] == [0, 1]


def test_null_defense_never_blocks():
    suite, env, runtime = banking()
    client = FakeClient([turn(("get_balance", {}))])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for(), defense=NullDefense())
    assert loop.run("balance").blocked == 0


# ------------------------------------------- end-to-end against real scorers


def test_ground_truth_calls_satisfy_the_benchmark_security_checker():
    """The load-bearing integration test.

    Replay the attacker's own reference solution through our loop and confirm
    AgentDojo's `security()` agrees the attack succeeded. If this passes, the
    loop, the runtime, the environment mutation and the scoring wiring are all
    connected correctly -- which is what the grid depends on.
    """
    suite, env, runtime = banking()
    inj = suite.injection_tasks["injection_task_0"]
    pre_env = copy.deepcopy(env)

    gt = inj.ground_truth(env)
    assert gt, "expected this injection task to have ground-truth calls"

    client = FakeClient([turn(*[(c.function, dict(c.args)) for c in gt])])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for())
    result = loop.run("(replaying attacker ground truth)")

    assert not result.errored
    assert inj.security(result.final_text, pre_env, env) is True


def test_doing_nothing_does_not_satisfy_the_security_checker():
    """The negative control: without the calls, the attack must not register."""
    suite, env, runtime = banking()
    inj = suite.injection_tasks["injection_task_0"]
    pre_env = copy.deepcopy(env)

    client = FakeClient([ModelResponse(text="I will not do that.", stop_reason="end_turn")])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for())
    result = loop.run("(agent refuses)")

    assert inj.security(result.final_text, pre_env, env) is False


def test_a_blocking_defense_prevents_the_attack_reaching_the_environment():
    """Blocking tool *output* does not stop a tool *call* -- and that matters.

    This pins a real property of the threat model: an output-inspection defense
    cannot undo a side effect that has already happened. It is the reason
    Class C policy defenses exist, and the reason the paper reports them
    separately.
    """
    suite, env, runtime = banking()
    inj = suite.injection_tasks["injection_task_0"]
    pre_env = copy.deepcopy(env)
    gt = inj.ground_truth(env)

    client = FakeClient([turn(*[(c.function, dict(c.args)) for c in gt])])
    loop = AgentLoop(client=client, runtime=runtime, env=env, ctx=ctx_for(), defense=BlockAll())
    result = loop.run("(replaying attacker ground truth under a blocking defense)")

    assert result.blocked >= 1
    assert inj.security(result.final_text, pre_env, env) is True


# ----------------------------------------------------------- model params


def test_no_sampling_params_are_sent():
    """SDK 1.6.0 dropped `temperature` from messages.create() entirely.

    Sending it raises TypeError before the request is even made. Pinned here
    because the obvious "set temperature=0 for determinism" instinct is now a
    crash, and the consequence -- runs are not bit-deterministic -- is what the
    Phase 5 variance measurement exists to quantify.
    """
    import inspect
    from importlib.metadata import version

    import anthropic

    assert SAMPLING_PARAMS_AVAILABLE is False
    # The hosted path is unused (zero spend), so an older SDK on a dev box is not
    # a failure of this repo -- only the SDK this note was written against is pinned.
    major, minor = (int(x) for x in version("anthropic").split(".")[:2])
    if (major, minor) < (1, 6):
        pytest.skip(f"anthropic {version('anthropic')} predates the 1.6.0 signature change")
    params = inspect.signature(anthropic.Anthropic().messages.create).parameters
    assert "temperature" not in params
    assert "cache_control" in params  # the cost lever we do rely on
