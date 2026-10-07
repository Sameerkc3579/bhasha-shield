"""The localized attacker goals: coverage, literal preservation, draft gating."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from corpus.goals import load_goals, localize_cases
from corpus.schema import InjectionCase

LANGS = ("hi", "hi_latn", "bn", "mixed")
NATIVE = re.compile(r"[ऀ-৿]")  # Devanagari + Bengali blocks

# Everything an AgentDojo checker can match on. Losing one in translation makes
# a goal fail for reasons that have nothing to do with the model's language skill.
LITERAL = re.compile(
    r"(?<!\w)'[^'\n]+'(?!\w)"
    r"|[\w.+-]+@[\w-]+\.[\w.]+"
    r"|www\.[\w./-]+[\w/]"
    r"|https?://\S+"
    r"|US\d{20,}|IBAN \d+"
    r"|\$[\d,]+"
    r"|\b\w+_\w+\b"
    r"|\d{4}-\d{2}-\d{2}|\d{2}:\d{2}"
)


def en_cases() -> list[InjectionCase]:
    lines = Path("corpus/injections/en.jsonl").read_text(encoding="utf-8").splitlines()
    return [InjectionCase.model_validate_json(l) for l in lines if l.strip()]


def en_keys() -> set[tuple[str, str]]:
    return {(c.domain, (c.source_case_id or "").split("/")[-1]) for c in en_cases()}


@pytest.mark.parametrize("lang", LANGS)
def test_every_goal_is_covered(lang):
    assert set(load_goals(lang)) == en_keys()


@pytest.mark.parametrize("lang", LANGS)
def test_checked_literals_survive_translation(lang):
    lost = [
        (key, lit)
        for key, g in load_goals(lang).items()
        for lit in set(LITERAL.findall(g.goal_en))
        if lit not in g.goal
    ]
    assert not lost, lost


@pytest.mark.parametrize("lang", LANGS)
def test_script_matches_condition(lang):
    for key, g in load_goals(lang).items():
        if lang in ("hi", "bn"):
            assert NATIVE.search(g.goal), f"{key}: {lang} has no native script"
        else:
            assert not NATIVE.search(g.goal), f"{key}: {lang} must be Latin script"


def test_hi_and_hi_latn_are_the_same_sentences():
    """Script-vs-language ablation: same items, same literals, same source."""
    hi, latn = load_goals("hi"), load_goals("hi_latn")
    assert set(hi) == set(latn)
    for key in hi:
        assert hi[key].goal_en == latn[key].goal_en
        assert set(LITERAL.findall(hi[key].goal)) == set(LITERAL.findall(latn[key].goal)), key


def test_unverified_drafts_are_refused_by_default():
    cases = en_cases()[:3]
    if all(g.verified_by for g in load_goals("hi").values()):
        pytest.skip("hi is fully verified")
    with pytest.raises(PermissionError):
        localize_cases(cases, "hi")


def test_localize_swaps_only_goal_and_lang():
    cases = en_cases()[:5]
    out = localize_cases(cases, "bn", allow_unverified=True)
    for before, after in zip(cases, out):
        assert after.lang == "bn"
        assert after.attacker_goal_nl != before.attacker_goal_nl
        assert after.case_id == before.case_id
        assert after.user_task == before.user_task
        assert after.source_case_id == before.source_case_id
