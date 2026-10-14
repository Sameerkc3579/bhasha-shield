"""The benign corpus -- contribution C4, the false-positive denominator.

The adversarial-benign half is the part that does the work. A corpus of only
ordinary text would show a near-zero FPR for any detector and tell us nothing;
these are the items that separate a shippable defense from one that blocks the
user's own security policy.
"""

from __future__ import annotations

import re

import pytest

from corpus.benign_build import (
    ADVERSARIAL_BENIGN,
    MAX_CHARS,
    MIN_CHARS,
    _key_kind,
    _walk,
    adversarial_items,
    build,
)
from corpus.schema import BenignItem

agentdojo = pytest.importorskip("agentdojo", reason="vendored benchmark not installed")


# ------------------------------------------------------- adversarial half


def test_adversarial_items_are_flagged_as_such():
    items = adversarial_items()
    assert items and all(i.adversarial_benign for i in items)
    assert all(i.lang == "en" for i in items)


def test_adversarial_ids_are_unique():
    ids = [i.item_id for i in adversarial_items()]
    assert len(ids) == len(set(ids))


def test_every_adversarial_item_actually_looks_like_an_injection():
    """Otherwise it is just ordinary text wearing a label.

    Each item must contain at least one phrase a naive keyword filter fires on.
    If this fails, the corpus has stopped testing what it claims to test.
    """
    triggers = re.compile(
        r"ignore|disregard|override|system:|assistant:|forward|delete|transfer|"
        r"previous instructions|developer mode|urgent|prompt injection|"
        r"exfiltrat|unauthorised|unauthorized|confirm your card",
        re.I,
    )
    for item in adversarial_items():
        assert triggers.search(item.content), f"{item.item_id} has no injection-like phrase"


def test_adversarial_categories_are_diverse():
    """A detector could otherwise be defeated by one narrow content type."""
    cats = {i.source.split("/")[-1] for i in adversarial_items()}
    assert len(cats) >= 25


def test_adversarial_items_span_multiple_tools():
    tools = {i.tool_name for i in adversarial_items()}
    assert len(tools) >= 3  # email, file, channel, webpage


def test_adversarial_items_are_substantial():
    """One-liners are too easy; real documents are what cause false positives."""
    for item in adversarial_items():
        assert len(item.content) >= 100, item.item_id


# --------------------------------------------------------------- walking


def test_key_kind_recognises_urls_and_filenames():
    assert _key_kind("www.example.com") == "webpage"
    assert _key_kind("https://x.example") == "webpage"
    assert _key_kind("bill-december-2023.txt") == "file"
    assert _key_kind("some_other_key") == "content"


def test_walk_finds_content_under_arbitrary_keys():
    """Files and web pages are keyed by filename/URL, not by a field name.

    The first build missed every banking file and Slack page because of this.
    """
    node = {"files": {"notes.txt": "x" * 200}, "body": "y" * 60}
    found = dict(_walk(node))
    assert found["file"] == "x" * 200
    assert found["body"] == "y" * 60


def test_walk_ignores_short_incidental_strings():
    """IBANs, names and addresses are not tool output."""
    node = {"iban": "DE89370400440532013000", "recipient": "Meera"}
    assert dict(_walk(node)) == {}


def test_walk_survives_deep_nesting_without_recursing_forever():
    node: dict = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"body": "z" * 80}}}}}}}}}
    list(_walk(node))  # must terminate


# ----------------------------------------------------------- full corpus


def test_corpus_has_both_halves():
    items = build()
    adv = [i for i in items if i.adversarial_benign]
    ordinary = [i for i in items if not i.adversarial_benign]
    assert len(adv) >= 30
    assert len(ordinary) >= 60


def test_corpus_covers_more_than_one_suite():
    """A single-domain corpus would confound domain with language later on."""
    items = [i for i in build() if not i.adversarial_benign]
    suites = {i.tool_name.split(".")[0] for i in items}
    assert len(suites) >= 3


def test_harvested_content_is_deduplicated():
    items = [i for i in build() if not i.adversarial_benign]
    norm = [re.sub(r"\s+", " ", i.content.lower()) for i in items]
    assert len(norm) == len(set(norm))


def test_all_items_respect_the_length_bounds():
    for i in build():
        assert MIN_CHARS <= len(i.content) <= MAX_CHARS, i.item_id


def test_item_ids_are_globally_unique():
    ids = [i.item_id for i in build()]
    assert len(ids) == len(set(ids))


def test_harvested_items_record_their_provenance():
    """Reviewers ask where corpus text came from."""
    for i in build():
        assert i.source


def test_corpus_serialises_roundtrip():
    for i in build()[:20]:
        assert BenignItem.model_validate_json(i.model_dump_json()).content == i.content


def test_benign_fpr_rescores_an_item_whose_text_changed():
    """2026-10-14: a review fix changed an item's text; incremental scoring by id kept the old score."""
    from scripts.benign_fpr import content_sha, stale

    item = BenignItem(item_id="x", lang="hi", tool_name="read_file", content="नई पंक्ति",
                      adversarial_benign=True, source="t")
    assert stale({"item_id": "x", "sha": content_sha("पुरानी पंक्ति")}, item)
    assert not stale({"item_id": "x", "sha": content_sha("नई पंक्ति")}, item)
    assert not stale({"item_id": "x"}, item)          # rows from before hashes are trusted
