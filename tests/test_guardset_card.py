"""Gate G2: the draft dataset card is rendered from the build's own files, never typed by hand."""

from __future__ import annotations

import json
import re

import pytest

from corpus.guardset import card as C
from corpus.guardset import generate as G


@pytest.fixture
def built(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from corpus.guardset import build
    from corpus.guardset.schema import Exclusions

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(G, "WORK", tmp_path / "drafts/work")
    monkeypatch.setattr(G, "DRAFTS", tmp_path / "drafts")
    monkeypatch.setattr(G, "TOPICS", G.TOPICS[:4])
    monkeypatch.setattr(G, "ADV_SEEDS", G.ADV_SEEDS[:3])
    G.main(["all", "--fake", "--work", str(tmp_path / "drafts/work")])
    (tmp_path / "drafts/work/model.json").write_text(json.dumps({"model": "fake", "revision": "abc",
                                                                 "indictrans2": None}), encoding="utf-8")
    from pathlib import Path
    manifest = Path(__file__).resolve().parents[1] / "corpus/testset/v2/manifest.json"
    s, rep = build.run(build.load_drafts(tmp_path / "drafts"), test_texts={},
                       exclusions=Exclusions.from_manifest(manifest), augment_rate=0.1)
    out = tmp_path / "out"
    out.mkdir()
    for name, rows in (("train", s.train), ("validation", s.validation)):
        (out / f"{name}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    return tmp_path, out, rep


def test_card_numbers_come_from_the_report(built):
    root, out, rep = built
    text = C.render(out=out, work=root / "drafts/work")
    assert not re.search(r"\{[a-z_]+\}", text), "an unfilled slot"
    assert text.startswith("---\n") and "license:" in text
    hi = rep["counts"]["train"].get("hi/injection", 0)
    assert f"| hi | {hi} |" in text
    assert f"{rep['leakage']['threshold']}" in text and "Sameer" not in text
    assert "not reviewed yet" in text      # no verdicts in this build
    assert all(f"`{f}`" in text for f in rep["held_out_families"]) and rep["held_out_families"]


def test_card_refuses_a_missing_build(tmp_path):
    with pytest.raises(SystemExit, match="build"):
        C.render(out=tmp_path / "none", work=tmp_path)
