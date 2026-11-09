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

# file name -> SHA-256 (CRLF folded to LF). Empty until Gate G1.
EXPECTED: dict[str, str] = {}


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
