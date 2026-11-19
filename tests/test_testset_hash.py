"""The frozen v2 test set must not change (docs/v2-plan.md: "the test set is frozen by a SHA-256 hash").

EXPECTED is filled in once, at Gate G1, from `python -m corpus.testset build`. Until then this test
skips. After that, any change to a frozen file -- or a rebuild that would produce different
bytes -- fails here and in CI. Changing EXPECTED is a decision with a dated RESULTS.md entry,
never a fix for a red test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.testset import OUT, file_sha

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / OUT

# file name -> SHA-256 (CRLF folded to LF). Pinned at Gate G1, 2026-11-19 (RESULTS.md).
EXPECTED: dict[str, str] = {
    "en.jsonl": "e0c1fd8642294799feb0964f759844471ebe5a86cc51465c339d23173e4d17a9",
    "hi.jsonl": "b406bb5da51a0bbc8998909c2eb8ca2d02ac7b03b8c1186879c3a31915356d03",
    "hi_latn.jsonl": "d69e978ae10b9fa5ae7a184ee6d6966284064a3aa84f6707d280c2f8ab5951b1",
    "manifest.json": "70329a54975d4833fda981a2c85e2f1726804319fe546cbdbe75b8299382b61b",
    "mixed.jsonl": "b4f41754af36d5c38399d1f4f87e1d912771962e7397d6f05ad1ef4a39921f3e",
}


@pytest.fixture
def frozen():
    if not EXPECTED:
        pytest.skip("test set not frozen yet (Gate G1)")
    return FROZEN


def test_every_frozen_file_has_its_pinned_hash(frozen):
    assert sorted(p.name for p in frozen.iterdir()) == sorted(EXPECTED)
    for name, digest in EXPECTED.items():
        assert file_sha(frozen / name) == digest, f"{name} changed after the freeze"


def test_manifest_agrees_with_the_files(frozen):
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pilot"] is False and manifest["problems_at_build"] == 0
    for name, digest in manifest["files"].items():
        assert file_sha(frozen / name) == digest
        rows = [json.loads(l) for l in (frozen / name).read_text(encoding="utf-8").splitlines()]
        assert all(r["verified_by"] for r in rows), f"{name} has an unverified row"
        assert sum(n for k in manifest["counts"][rows[0]["lang"]].values() for n in k.values()) == len(rows)
