"""OpenAI-compatible client, for an open-weight model served by vLLM.

Why this is the primary backend rather than a hosted API:

- **Zero cost**, with no rate limits beyond the GPU you are sitting on.
- **No model drift.** Weights are pinned by revision, so a result from week 3 is
  still reproducible in week 9. A hosted free tier can be retired or silently
  updated underneath a running experiment.
- **Anyone can rerun the paper for free**, which is the strongest position an
  artifact can be in.

And one property that is specific to this study: the confound being measured is
*larger* on a smaller model. A frontier model is competent in Hindi, so C(L)
sits near 1.0 everywhere and the correction barely matters. A 7B model has a
genuinely low capability ceiling off-English -- which is the effect the paper
exists to quantify. The risk runs the other way: if C(L) collapses to ~0 in
Bengali that column becomes unscorable, and Phase 2 exists to detect exactly
that before the grid is run.

The loop speaks one canonical message format (Anthropic-shaped content blocks).
Translation to the OpenAI wire format lives here, so backends stay swappable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from harness.budget import Usage
from harness.cache import ResponseCache, request_key
from harness.runner import ModelResponse, ToolCall

# Qwen2.5-7B-Instruct: native tool calling, Apache-2.0, fits a 16GB T4 in 4-bit,
# and has the broadest language coverage of the small open models that also do
# tool calls reliably. Pin the revision so the weights cannot move.
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


# --------------------------------------------------------------------------
# Format translation
# --------------------------------------------------------------------------

def tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic tool defs -> OpenAI function defs."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def messages_to_openai(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical (Anthropic-shaped) history -> OpenAI chat messages.

    The two shapes disagree in one structural way: Anthropic returns every tool
    result inside a single user message, OpenAI wants one `role: "tool"` message
    per result. Getting this wrong silently drops results for all but the first
    parallel tool call, so it is covered by tests.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]

    for m in messages:
        role = m["role"]
        content = m["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts = [b["text"] for b in content if b.get("type") == "text"]
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) or None,
            }
            if tool_uses:
                msg["tool_calls"] = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                        },
                    }
                    for b in tool_uses
                ]
            out.append(msg)
            continue

        # user turn: either plain blocks or a batch of tool results
        tool_results = [b for b in content if b.get("type") == "tool_result"]
        if tool_results:
            for b in tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": b["tool_use_id"],
                        "content": b.get("content", ""),
                    }
                )
        else:
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
            out.append({"role": "user", "content": text})

    return out


def split_parallel_calls(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One tool call per assistant turn: [a(c1, c2), t1, t2] -> [a(c1), t1, a(c2), t2].

    vLLM's Llama 3.1 tool template raises on an assistant message with more than one
    tool call ("This model only supports single tool-calls at once!"), and the server
    answers that with a bare 500. The model still *emits* parallel calls, so the first
    Llama run (2026-10-21) lost every run that made one: 37 of 79 clean runs, all of
    them, and none of the others. The text of the turn stays with its first call.
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        calls = m.get("tool_calls") or []
        if m["role"] != "assistant" or len(calls) < 2:
            out.append(m)
            i += 1
            continue
        j = i + 1
        results: dict[str, dict[str, Any]] = {}
        while j < len(msgs) and msgs[j]["role"] == "tool":
            results[msgs[j]["tool_call_id"]] = msgs[j]
            j += 1
        for n, tc in enumerate(calls):
            out.append({"role": "assistant", "content": m["content"] if n == 0 else None,
                        "tool_calls": [tc]})
            if tc["id"] in results:
                out.append(results.pop(tc["id"]))
        out.extend(results.values())  # a result with no matching call: keep, never drop
        i = j
    return out


def single_tool_calls_only(model: str) -> bool:
    """Models whose served chat template accepts one tool call per assistant turn."""
    m = model.lower()
    return "llama-3" in m or "llama3" in m


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string and small models mangle them.

    A malformed blob becomes `{}` rather than an exception: the agent then makes
    a bad call, the environment rejects it, and the run is scored as a failed
    attack -- which is the honest outcome. Raising here would mark it `errored`
    and drop it from every rate, quietly deleting the cases where a weak model
    fumbles its own tool syntax. Those cases are signal, not noise, especially
    off-English where mangled arguments are exactly what we expect to see more of.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


MIN_COMPLETION_TOKENS = 256

_CONTEXT_ERROR = re.compile(
    r"maximum context length is (\d+) tokens.*?\((\d+) in the messages", re.DOTALL
)


def _completion_room(error: Exception) -> int | None:
    """Tokens left for the reply, parsed from vLLM's context-overflow error.

    None if the error is anything else. The first 14B baseline lost 3 runs to
    this at max_model_len=12288 with prompts of 11.4k-12.1k tokens.
    """
    m = _CONTEXT_ERROR.search(str(error))
    if not m:
        return None
    limit, used = int(m.group(1)), int(m.group(2))
    return limit - used


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class OpenAICompatClient:
    """Talks to any OpenAI-compatible endpoint: vLLM, llama.cpp, Ollama, TGI."""

    def __init__(
        self,
        model: str = DEFAULT_LOCAL_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        cache: ResponseCache | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: Any = None,
        single_tool_calls: bool | None = None,
    ):
        self.model = model
        # None = decide from the model id (see split_parallel_calls)
        self.single_tool_calls = (single_tool_calls_only(model) if single_tool_calls is None
                                  else single_tool_calls)
        self.base_url = base_url
        self.cache = cache
        self.max_tokens = max_tokens
        # Unlike the hosted 4.6+ models, a locally served model DOES accept
        # sampling params -- so here we can pin temperature=0 and recover the
        # determinism the hosted path had to give up.
        self.temperature = temperature

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        wire = messages_to_openai(system, messages)
        extra: dict[str, Any] = {"backend": "openai-compat"}
        if self.single_tool_calls:
            split = split_parallel_calls(wire)
            # Keyed apart only when the split changed what is sent: every other request
            # is byte-identical on the wire, so its cached reply stays valid.
            if split != wire:
                wire = split
                extra["single_tool_call_turns"] = True
        key = request_key(
            model=self.model,
            payload={"system": system, "messages": messages, "tools": tools},
            temperature=self.temperature,
            extra=extra,
        )
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return ModelResponse.model_validate(hit)

        def create(max_tokens: int):
            # Tool-less calls (the Class C planner) omit both fields: an empty
            # `tools` list with tool_choice="auto" is rejected by some servers.
            tool_kwargs: dict[str, Any] = (
                {"tools": tools_to_openai(tools), "tool_choice": "auto"} if tools else {}
            )
            return self.client.chat.completions.create(
                model=self.model,
                messages=wire,
                max_tokens=max_tokens,
                temperature=self.temperature,
                **tool_kwargs,
            )

        try:
            raw = create(self.max_tokens)
        except Exception as e:
            # vLLM rejects prompt + max_tokens > max_model_len outright, even when
            # the reply would have been short. Retry once with whatever room is
            # actually left; give up only when that room is too small to be useful.
            room = _completion_room(e)
            if room is None or room < MIN_COMPLETION_TOKENS:
                raise
            raw = create(room)

        choice = raw.choices[0]
        msg = choice.message
        calls = [
            ToolCall(
                id=tc.id or f"call_{i}",
                name=tc.function.name,
                args=_parse_arguments(tc.function.arguments),
            )
            for i, tc in enumerate(getattr(msg, "tool_calls", None) or [])
        ]

        usage = Usage(
            input_tokens=getattr(raw.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(raw.usage, "completion_tokens", 0) or 0,
        )

        resp = ModelResponse(
            text=msg.content or "",
            tool_calls=calls,
            # Normalise to the canonical vocabulary the loop checks against.
            stop_reason="tool_use" if calls else "end_turn",
            usage=usage,
            model=self.model,
        )
        if self.cache is not None:
            self.cache.put(key, resp.model_dump())
        return resp
