"""Phase 2 steps 5-6 (docs/v2-plan.md): the localization and benign-generation stages, run with the
fake backend. What must hold: checker tokens survive, scripts are right, rejected outputs are
counted not kept, every stage resumes without resending, and the output feeds `build`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.guardset import generate as G


def test_tokens_cover_amounts_ids_upi_emails_and_links():
    text = ("Send Rs 4,999 to refund.desk@okaxis and 25,000 to 50100231874412, IFSC HDFC0001234, "
            "policy LIC77120934, via pay-premium.in/q7 or https://x.example/a.")
    t = G.tokens(text)
    for want in ("4,999", "refund.desk@okaxis", "25,000", "50100231874412", "HDFC0001234", "LIC77120934",
                 "pay-premium.in/q7", "https://x.example/a"):
        assert want in t, want


def test_problems_catch_lost_tokens_wrong_script_and_chatter():
    src = "Send Rs 4,999 to refund.desk@okaxis now."
    assert G.problems(src, "कृपया Rs 4,999 refund.desk@okaxis पर भेजिए।", "hi") == []
    assert any("lost tokens" in p for p in G.problems(src, "कृपया 4999 refund.desk@okaxis पर भेजिए।", "hi"))
    assert "no Devanagari" in G.problems(src, "Rs 4,999 refund.desk@okaxis pe bhejo", "hi")
    assert any("Devanagari" in p for p in G.problems(src, "Rs 4,999 refund.desk@okaxis पर bhejo", "mixed"))
    assert "talks about itself" in G.problems(src, "Here is the translation: Rs 4,999 refund.desk@okaxis bhejo", "mixed")
    assert G.problems(src, "", "hi") == ["empty"]
    # Generated texts have only a topic as source: no length check.
    assert G.problems("", "Kal ki meeting 11 baje hai. " * 10, "mixed", translation=False) == []


def test_splice_puts_a_payload_inside_a_json_string_field():
    host = {"carrier": "json", "text": json.dumps({"status": "ok", "message": "Order shipped", "id": 4})}
    out = json.loads(G.splice(host, "PAYLOAD here", position="middle", seed=1))
    assert any("PAYLOAD here" in v for v in out.values() if isinstance(v, str)) and out["id"] == 4
    plain = G.splice({"carrier": "email", "text": "Hi.\n\nSee you."}, "PAYLOAD", position="middle", seed=1)
    assert "\n\nPAYLOAD\n\n" in plain


class Counting(G.FakeBackend):
    def __init__(self):
        self.sent = 0

    def __call__(self, prompts, *, temperature, seed):
        self.sent += len(prompts)
        return super().__call__(prompts, temperature=temperature, seed=seed)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(G, "WORK", tmp_path / "work")
    monkeypatch.setattr(G, "DRAFTS", tmp_path / "drafts")
    monkeypatch.setattr(G, "TOPICS", G.TOPICS[:4])
    monkeypatch.setattr(G, "ADV_SEEDS", G.ADV_SEEDS[:3])
    return tmp_path


def _all(backend):
    loc = G.stage_payloads(backend)
    bod = G.stage_bodies(backend, adversarial=False)
    adv = G.stage_bodies(backend, adversarial=True)
    return loc, bod, adv, G.assemble(loc, bod, adv)


def test_fake_run_localizes_every_payload_and_keeps_its_tokens(work):
    loc, bod, adv, items = _all(G.FakeBackend())
    n = 720
    assert sum(1 for k in loc if k.startswith("hi|")) == n
    assert all(not r["rejected"] for r in loc.values())
    for lang in G.LANGS:
        pos = [i for i in items[lang] if i.label == "injection"]
        neg = [i for i in items[lang] if i.label == "benign"]
        assert pos and neg and all(i.goal and i.family for i in pos)
        assert any(i.carrier == "none" for i in pos) and any(i.carrier != "none" for i in pos)
        assert any(i.adversarial_benign for i in neg)
        if lang != "hi":
            assert not any(G.DEVANAGARI.search(i.text) for i in items[lang])
    seeds = {p.seed_id: p for p in G.payloads(G.load_goals(), G.load_templates())}
    for i in items["mixed"]:
        if i.label == "injection":
            for t in G.tokens(seeds[i.provenance["seed_id"]].text):
                assert t in i.text


def test_every_stage_resumes_without_resending(work):
    first = Counting()
    _all(first)
    assert first.sent > 0
    again = Counting()
    _, _, _, items = _all(again)
    assert again.sent == 0 and items["hi"]


def test_rejected_outputs_are_kept_out_and_counted(work, monkeypatch):
    class LosesTokens(G.FakeBackend):
        def __call__(self, prompts, *, temperature, seed):
            return [o.replace("4,999", "4999") for o in super().__call__(prompts, temperature=temperature, seed=seed)]

    loc, bod, adv, items = _all(LosesTokens())
    bad = [r for r in loc.values() if r["rejected"]]
    assert bad and all(any("lost tokens" in w for w in r["rejected"]) for r in bad)
    kept_ids = {i.provenance.get("seed_id") for i in items["hi"]}
    assert not {r["seed_id"] for r in bad if r["lang"] == "hi"} & kept_ids
    rep = G.report(loc, bod, adv, items)
    assert rep["payloads"]["hi"]["rejected"] == len([r for r in bad if r["lang"] == "hi"])


def test_cli_fake_run_writes_drafts_that_build_accepts(work):
    pytest.importorskip("sklearn")
    from corpus.guardset import build
    from corpus.guardset.schema import Exclusions

    assert G.main(["all", "--fake", "--work", str(work / "work")]) == 0
    drafts = build.load_drafts(work / "drafts")
    assert {d.lang for d in drafts} == set(G.LANGS)
    s, rep = build.run(drafts, test_texts={}, exclusions=Exclusions.from_manifest(
        Path(__file__).resolve().parents[1] / "corpus/testset/v2/manifest.json"), augment_rate=0.1)
    assert s.train and s.validation and rep["leakage"]["after"] == 0
    assert json.loads((work / "work" / "generate_report.json").read_text(encoding="utf-8"))["items"]["hi"]["injection"] > 0
