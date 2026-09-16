"""OpenAI-compatible backend: format translation and response parsing.

The translation layer is where a silent bug costs the most -- a dropped tool
result or a swallowed argument does not raise, it just makes the agent look
incompetent, which this study would then misread as a low capability ceiling.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from harness.local_client import (
    OpenAICompatClient,
    _parse_arguments,
    messages_to_openai,
    tools_to_openai,
)


# ------------------------------------------------------------ tool defs


def test_tools_translate_to_openai_function_shape():
    anthropic_tools = [
        {"name": "get_balance", "description": "Get balance", "input_schema": {"type": "object", "properties": {}}}
    ]
    out = tools_to_openai(anthropic_tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "get_balance"
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


# ------------------------------------------------------------- messages


def test_system_prompt_becomes_the_first_message():
    out = messages_to_openai("be helpful", [{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_assistant_tool_use_becomes_tool_calls_with_json_arguments():
    messages = [
        {"role": "user", "content": "balance?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t0", "name": "get_balance", "input": {"account": "main"}},
            ],
        },
    ]
    out = messages_to_openai("sys", messages)
    asst = out[-1]
    assert asst["role"] == "assistant"
    assert asst["content"] == "checking"
    assert asst["tool_calls"][0]["id"] == "t0"
    assert asst["tool_calls"][0]["function"]["name"] == "get_balance"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"account": "main"}


def test_every_parallel_tool_result_becomes_its_own_tool_message():
    """The structural difference between the two APIs.

    Anthropic batches all results into one user message; OpenAI wants one
    `role: "tool"` message each. Collapsing them would silently drop every
    result after the first.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t0", "content": "100"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "DE89"},
            ],
        }
    ]
    out = messages_to_openai("sys", messages)
    tool_msgs = [m for m in out if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert [m["tool_call_id"] for m in tool_msgs] == ["t0", "t1"]
    assert [m["content"] for m in tool_msgs] == ["100", "DE89"]


def test_assistant_with_no_text_sends_null_content():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t0", "name": "f", "input": {}}],
        }
    ]
    out = messages_to_openai("sys", messages)
    assert out[-1]["content"] is None


def test_unicode_arguments_survive_translation():
    """Payloads are Devanagari and Bengali; ascii escaping would corrupt them."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t0", "name": "send", "input": {"body": "मेरा ईमेल"}}
            ],
        }
    ]
    out = messages_to_openai("sys", messages)
    args = out[-1]["tool_calls"][0]["function"]["arguments"]
    assert "मेरा ईमेल" in args
    assert json.loads(args)["body"] == "मेरा ईमेल"


# ------------------------------------------------------------ arguments


def test_arguments_parse_from_json_string():
    assert _parse_arguments('{"a": 1}') == {"a": 1}


def test_malformed_arguments_degrade_to_empty_not_an_exception():
    """A weak model fumbling its own tool syntax is signal, not noise.

    Raising here would mark the run `errored` and drop it from every rate --
    deleting exactly the cases we expect to see more of off-English.
    """
    assert _parse_arguments("{not json") == {}
    assert _parse_arguments("") == {}
    assert _parse_arguments(None) == {}
    assert _parse_arguments("[1,2,3]") == {}  # valid JSON, wrong shape


def test_arguments_passed_through_when_already_a_dict():
    assert _parse_arguments({"a": 1}) == {"a": 1}


# --------------------------------------------------------------- client


class FakeOpenAI:
    """Minimal stand-in for the OpenAI SDK surface the client touches."""

    def __init__(self, message, usage=(10, 5)):
        self._message = message
        self._usage = usage
        self.seen: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self._message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=self._usage[0], completion_tokens=self._usage[1]),
        )


def make_client(message, cache=None):
    return OpenAICompatClient(
        "test-model", cache=cache, client=FakeOpenAI(message)
    )


def test_plain_text_response_ends_the_turn():
    client = make_client(SimpleNamespace(content="all done", tool_calls=None))
    resp = client.complete(system="s", messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.text == "all done"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10


def test_tool_calls_set_stop_reason_to_tool_use():
    """The loop branches on this, so the vocabulary must be normalised."""
    msg = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(id="c0", function=SimpleNamespace(name="get_balance", arguments='{"x":1}'))
        ],
    )
    client = make_client(msg)
    resp = client.complete(system="s", messages=[], tools=[])
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "get_balance"
    assert resp.tool_calls[0].args == {"x": 1}


def test_temperature_zero_is_sent_to_a_local_model():
    """Local serving accepts sampling params, so determinism comes back."""
    client = make_client(SimpleNamespace(content="x", tool_calls=None))
    client.complete(system="s", messages=[], tools=[])
    assert client.client.seen["temperature"] == 0.0


def test_missing_tool_call_id_is_backfilled():
    msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id=None, function=SimpleNamespace(name="f", arguments="{}"))],
    )
    resp = make_client(msg).complete(system="s", messages=[], tools=[])
    assert resp.tool_calls[0].id == "call_0"


def test_response_is_cached_and_replayed(tmp_path):
    from harness.cache import ResponseCache

    cache = ResponseCache(tmp_path)
    client = make_client(SimpleNamespace(content="cached", tool_calls=None), cache=cache)
    a = client.complete(system="s", messages=[{"role": "user", "content": "q"}], tools=[])
    b = client.complete(system="s", messages=[{"role": "user", "content": "q"}], tools=[])
    assert a.text == b.text == "cached"
    assert cache.hits == 1
