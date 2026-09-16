"""Budget cap, response cache, and traces.

Unglamorous, but each of these has a failure mode that costs real money or a
week of work: a cap that resets on restart, a cache key that varies between
runs, a trace that loses everything when a run is killed.
"""

from __future__ import annotations

import pytest

from harness.budget import (
    Budget,
    BudgetExceeded,
    Usage,
    cost_usd,
    estimate_grid_cost,
)
from harness.cache import ResponseCache, request_key
from harness.trace import RunTrace


# ------------------------------------------------------------------ budget


def test_cost_matches_published_rates():
    # Haiku 4.5: $1.00 / $5.00 per MTok
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", usage) == pytest.approx(6.00)
    # Opus 5: $5.00 / $25.00 per MTok
    assert cost_usd("claude-opus-5", usage) == pytest.approx(30.00)


def test_cache_reads_are_billed_at_a_tenth():
    read = Usage(cache_read_input_tokens=1_000_000)
    plain = Usage(input_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", read) == pytest.approx(
        cost_usd("claude-haiku-4-5", plain) * 0.10
    )


def test_cache_writes_cost_a_quarter_more():
    write = Usage(cache_creation_input_tokens=1_000_000)
    plain = Usage(input_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", write) == pytest.approx(
        cost_usd("claude-haiku-4-5", plain) * 1.25
    )


def test_unknown_model_raises_instead_of_billing_free():
    """A typo'd model name must not silently make the cap unenforceable."""
    with pytest.raises(KeyError):
        cost_usd("claude-not-a-model", Usage(input_tokens=1000))


def test_budget_blocks_the_call_that_would_breach_the_cap(tmp_path):
    b = Budget("test", cap_usd=1.00, path=tmp_path / "b.json")
    b.record("claude-haiku-4-5", Usage(input_tokens=900_000))  # $0.90
    b.check(estimated_usd=0.05)  # still fits
    with pytest.raises(BudgetExceeded):
        b.check(estimated_usd=0.20)


def test_budget_survives_a_restart(tmp_path):
    """The failure mode that actually empties an account: resume resets to zero."""
    p = tmp_path / "b.json"
    b1 = Budget("phase5", cap_usd=10.0, path=p)
    b1.record("claude-haiku-4-5", Usage(input_tokens=2_000_000))  # $2.00
    b2 = Budget("phase5", cap_usd=10.0, path=p)
    assert b2.state.spent_usd == pytest.approx(2.00)
    assert b2.remaining == pytest.approx(8.00)


def test_budget_warns_at_ninety_percent(tmp_path):
    b = Budget("test", cap_usd=1.00, path=tmp_path / "b.json")
    assert b.warning() is None
    b.record("claude-haiku-4-5", Usage(input_tokens=950_000))
    assert "BUDGET" in b.warning()


def test_grid_estimate_is_sensitive_to_cache_hit_rate():
    """If caching silently breaks, the grid costs several times more."""
    warm = estimate_grid_cost(
        n_cases=150, n_langs=5, n_defenses=9, model="claude-haiku-4-5",
        cache_hit_fraction=0.8,
    )
    cold = estimate_grid_cost(
        n_cases=150, n_langs=5, n_defenses=9, model="claude-haiku-4-5",
        cache_hit_fraction=0.0,
    )
    assert cold > warm * 2


# ------------------------------------------------------------------- cache


def test_request_key_is_stable_across_dict_ordering():
    """Unsorted JSON would produce a new key per run and disable the cache."""
    a = request_key(model="m", payload={"a": 1, "b": {"x": 1, "y": 2}})
    b = request_key(model="m", payload={"b": {"y": 2, "x": 1}, "a": 1})
    assert a == b


def test_request_key_separates_models_and_seeds():
    base = dict(payload={"p": 1})
    assert request_key(model="a", **base) != request_key(model="b", **base)
    assert request_key(model="a", seed=1, **base) != request_key(model="a", seed=2, **base)
    assert (
        request_key(model="a", temperature=0.0, **base)
        != request_key(model="a", temperature=1.0, **base)
    )


def test_cache_roundtrip_and_hit_accounting(tmp_path):
    c = ResponseCache(tmp_path)
    k = request_key(model="m", payload={"q": "hello"})
    assert c.get(k) is None
    c.put(k, {"text": "hi"})
    assert c.get(k) == {"text": "hi"}
    assert (c.hits, c.misses) == (1, 1)
    assert c.hit_rate == pytest.approx(0.5)


def test_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    c = ResponseCache(tmp_path)
    k = request_key(model="m", payload={"q": 1})
    c.put(k, {"ok": True})
    c._path(k).write_text("{truncated", encoding="utf-8")
    assert c.get(k) is None  # interrupted write must not kill a grid run


def test_cache_survives_unicode_payloads(tmp_path):
    """Corpus content is Devanagari and Bengali; ascii-only would corrupt keys."""
    c = ResponseCache(tmp_path)
    k = request_key(model="m", payload={"q": "मेरा ईमेल पढ़ो"})
    c.put(k, {"text": "উত্তর"})
    assert c.get(k) == {"text": "উত্তর"}


# ------------------------------------------------------------------- trace


def test_trace_writes_jsonl_and_reads_back(tmp_path):
    with RunTrace(phase="p0", lang="hi", defense="undefended", model="m", root=tmp_path) as t:
        t.write("run_start", run_id="r1", case_id="c1", payload={"task": "x"})
        t.write("run_end", run_id="r1", case_id="c1", payload={"ok": True})
    events = RunTrace.read(t.path)
    kinds = [e.kind for e in events]
    assert kinds == ["note", "run_start", "run_end"]
    assert events[0].payload["git_sha"]  # provenance recorded
    assert events[-1].payload["ok"] is True


def test_trace_is_append_only_across_sessions(tmp_path):
    for _ in range(2):
        with RunTrace(phase="p0", lang="en", defense="d", model="m", root=tmp_path) as t:
            t.write("note", run_id="r", payload={})
    assert len(RunTrace.read(t.path)) == 4  # 2 session notes + 2 events


def test_trace_cells_are_separate_files(tmp_path):
    a = RunTrace(phase="p", lang="hi", defense="d", model="m", root=tmp_path)
    b = RunTrace(phase="p", lang="bn", defense="d", model="m", root=tmp_path)
    assert a.path != b.path  # a crashed cell loses only itself
