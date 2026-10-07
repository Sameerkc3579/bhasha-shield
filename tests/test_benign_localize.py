"""Localized benign corpora: the invariants the FPR comparison depends on."""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.benign_localize import check, entities, load_en, parse
from corpus.schema import BenignItem

LANGS = ("hi", "hi_latn", "bn", "mixed")


def load(lang: str) -> list[BenignItem]:
    path = Path(f"corpus/benign/{lang}.jsonl")
    if not path.exists():
        pytest.skip(f"no {lang} benign corpus yet")
    return [BenignItem.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_parse_round_trips_the_draft_format():
    text = "=== a\nline one\n\nline two\n=== b\nx"
    assert parse(text) == {"a": "line one\n\nline two", "b": "x"}
    with pytest.raises(ValueError):
        parse("=== a\nx\n=== a\ny")


def test_entities_catch_what_tools_act_on():
    e = entities("Mail jay@google.com, see www.x.com! IBAN UK12345678901234567890 file a.docx cost 1,240")
    assert {"jay@google.com", "www.x.com", "UK12345678901234567890", "a.docx", "1,240"} <= e


def test_check_flags_a_mangled_entity():
    en = load_en()
    k = "benign_env_banking_0077"
    drafts = {i: en[i].content for i in en}
    drafts[k] = drafts[k].replace("UK12345678901234567890", "UK1234")
    problems = check(en, drafts)
    assert any(k in p and "entities lost" in p for p in problems)


@pytest.mark.parametrize("lang", LANGS)
def test_localized_corpus_pairs_item_by_item_with_english(lang):
    en = load_en()
    rows = load(lang)
    assert [r.item_id for r in rows] == list(en)          # same ids, same order
    for r in rows:
        src = en[r.item_id]
        assert r.lang == lang
        assert (r.tool_name, r.adversarial_benign, r.source) == (src.tool_name, src.adversarial_benign, src.source)
    # A few items are pure data (a budget CSV, a keyword list) and rightly identical
    # in every language; anything beyond a handful means a file was never translated.
    same = [r.item_id for r in rows if r.content == en[r.item_id].content]
    assert len(same) <= 6, same
    assert check(en, {r.item_id: r.content for r in rows}) == []


def test_hi_latn_is_latin_script_only():
    """The script ablation is worthless if Devanagari leaks into the romanized condition."""
    for r in load("hi_latn"):
        assert all(ord(ch) < 0x0900 or ord(ch) > 0x097F for ch in r.content), r.item_id


def test_every_localized_item_has_provenance():
    rows = Path("corpus/provenance.jsonl").read_text(encoding="utf-8")
    for lang in LANGS:
        for r in load(lang):
            assert f'"item_id":"benign/{r.item_id}","lang":"{lang}"' in rows, (lang, r.item_id)
