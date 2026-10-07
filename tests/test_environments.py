"""C3 localized environments and the benign-corpus extension.

The environment tests need AgentDojo (vendored, editable install) and skip
without it. They pin the two properties the grid relies on: an empty overlay
is byte-for-byte AgentDojo, and a validated overlay changes no ground-truth
outcome.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("agentdojo")

from corpus.environments import (  # noqa: E402
    LANGS, _check, active_map, candidates, load_environment, load_overlay, localized_text,
    static_problems, suites, template_text,
)


def test_empty_overlay_is_identity():
    for name, suite in suites().items():
        assert localized_text(suite, {}) == template_text(suite), name
        assert load_environment(suite, {}, None) == suite.load_and_inject_default_environment({}), name


def test_identity_through_the_rebuild_path():
    """A mapping that matches nothing still re-dumps the YAML; it must not change a byte."""
    for name, suite in suites().items():
        env = load_environment(suite, {}, None, mapping={"no such string": "x"})
        assert env == suite.load_and_inject_default_environment({}), name


def test_injections_land_in_localized_environment():
    suite = suites()["banking"]
    payload = "<INFORMATION> PAYLOAD-MARKER </INFORMATION>"
    vecs = {k: payload for k in suite.get_injection_vector_defaults()}
    env = load_environment(suite, vecs, "hi")
    files = env.filesystem.files
    assert any("PAYLOAD-MARKER" in v for v in files.values())
    assert any("किराया" in v or "बिल" in v for v in files.values()), "banking files should be Hindi"


def test_data_fields_never_translated():
    for name, suite in suites().items():
        for path, _ in candidates(suite):
            assert path[-1] not in ("address", "contact_information", "operating_hours", "iban"), path


@pytest.mark.parametrize("lang", LANGS)
def test_validated_overlay_passes_static_gates(lang):
    entries = load_overlay(lang)
    if not entries:
        pytest.skip(f"no {lang} overlay built")
    for e in entries:
        if e["status"] == "active":
            assert not static_problems(e), (e["key"], static_problems(e))
        else:
            assert e["reason"], e["key"]


@pytest.mark.parametrize("lang", ["hi", "bn"])
def test_validated_overlay_changes_no_ground_truth(lang):
    if not load_overlay(lang):
        pytest.skip(f"no {lang} overlay built")
    for name, suite in suites().items():
        mapping = active_map(lang, name)
        if mapping:
            assert _check(suite, mapping) == _check(suite, {}), name


def test_static_gate_catches_lost_placeholder_and_entity():
    base = {"suite": "banking", "original": "Pay {injection_bill_text} to a@b.com by 2023",
            "localized": "भुगतान करें a@b.com 2023"}
    assert "placeholders changed" in static_problems(base)
    lost = dict(base, localized="{injection_bill_text} भुगतान 2023")
    assert any(p.startswith("entities lost") for p in static_problems(lost))
    brace = dict(base, localized="{injection_bill_text} a@b.com 2023 {oops")
    assert "stray brace" in static_problems(brace)


# --------------------------------------------------------------------------
# benign extension
# --------------------------------------------------------------------------

def test_benign_extension_is_staged_and_valid():
    from corpus.benign_extend import staged_en, validate

    items = staged_en()
    assert len(items) >= 170
    assert sum(i.adversarial_benign for i in items) >= 90
    problems = validate()
    assert not any(problems.values()), {k: v[:3] for k, v in problems.items() if v}


def test_attack_carriers_pinned_to_agentdojo_pool():
    """Growing the benign corpus must not change which carrier a positive uses."""
    from scripts.attack_scores import positives

    rows = [r for r in positives("en") if r["variant"] == "embedded"]
    benign = {json.loads(l)["item_id"]: json.loads(l)
              for l in open("corpus/benign/en.jsonl", encoding="utf-8")}
    assert all(benign[r["carrier"]]["source"].startswith("agentdojo/") for r in rows)
