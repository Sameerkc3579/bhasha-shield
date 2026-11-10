"""Score every classifier on the v2 test set -- Phase 3 of docs/v2-plan.md.

    python -m scripts.guard_eval --defenses protectai_deberta --langs en hi hi_latn mixed
    python -m scripts.guard_eval --set pilot --kinds v1_embedded v1_payload v1_benign --defenses ...
    python -m scripts.guard_eval --set pilot --reproduce-v1 --defenses protectai_deberta

One score per item of `corpus/testset/{set}/{lang}.jsonl`, with the item's text hash. A later run
reuses a row whose item id, text hash and model id are unchanged, so a rebuilt set (an edited
benign item, new native items) costs only the changed rows. No threshold is applied here:
`analysis/guard_tables.py` fits thresholds on the calibration half and reports the test half.
Every model goes through its `defenses/` wrapper, so chunking (1500 chars, 200 overlap, max
score) is the same for all of them, as the plan requires.

**`--reproduce-v1`** is Phase 3 step 1: before anything new is scored, the v1 items in the set
(kinds v1_*, byte-identical to what v1 scored) must give v1's numbers. It compares every fresh
score with `analysis/out/{attack_scores,benign}/` and re-runs v1's own `analysis.calibrate` on the
fresh scores, then prints both diffs. Output: `analysis/out/guard_eval/{set}/reproduce_v1__{defense}.json`.

Scores: `analysis/out/guard_eval/{set}/{lang}__{defense}.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from corpus.schema import ToolOutput
from corpus.testset import LANGS_V2, OUT as V2, PILOT, FrozenItem, sha

SETS = {"v2": V2, "pilot": PILOT}
OUT = Path("analysis/out/guard_eval")
V1_KINDS = ("v1_embedded", "v1_payload", "v1_benign")
V1_ATTACK = Path("analysis/out/attack_scores")
V1_BENIGN = Path("analysis/out/benign")
ROW_FIELDS = ("item_id", "kind", "label", "half", "goal", "adversarial_benign")


def load_items(set_name: str, lang: str) -> list[FrozenItem]:
    path = SETS[set_name] / f"{lang}.jsonl"
    if not path.exists():
        hint = " (python -m corpus.testset build --pilot)" if set_name == "pilot" else " (Gate G1)"
        raise SystemExit(f"{path} does not exist{hint}")
    return [FrozenItem.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def out_path(set_name: str, lang: str, defense: str) -> Path:
    return OUT / set_name / f"{lang}__{defense}.json"


def run(name: str, lang: str, set_name: str = "v2", *, defense=None, kinds: tuple[str, ...] | None = None,
        rerun: bool = False, device: int = -1) -> Path:
    """Score `lang` of the set with defense `name`; returns the score file."""
    from defenses.base import RunContext
    from defenses.registry import build

    defense = defense or build(name, device=device)
    model_id = getattr(defense, "model_id", "")
    items = [i for i in load_items(set_name, lang) if kinds is None or i.kind in kinds]
    path = out_path(set_name, lang, name)
    old = {}
    if path.exists() and not rerun:
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("model_id") == model_id:
            old = {r["item_id"]: r for r in prev["scores"]}

    t0, rows, fresh = time.time(), [], 0
    for it in items:
        h = sha(it.text)
        r = old.get(it.item_id)
        if r is None or r["sha"] != h:
            ctx = RunContext(case_id=it.item_id, lang=lang, model="none", user_task="")
            v = defense.inspect(ToolOutput(tool_name=it.tool_name or "tool", content=it.text, lang=lang), ctx)
            r = {"score": v.score, "latency_ms": round(v.latency_ms, 2)}
            fresh += 1
        rows.append({k: getattr(it, k) for k in ROW_FIELDS} | {"sha": h, "score": r["score"],
                                                               "latency_ms": r["latency_ms"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"set": set_name, "lang": lang, "defense": name, "model_id": model_id,
                                "scored_now": fresh, "seconds": round(time.time() - t0, 1), "scores": rows},
                               indent=1, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- reproduce v1 (Phase 3 step 1)

def v1_views(rows: list[dict]) -> tuple[dict, dict]:
    """Fresh rows in the shape of v1's benign and attack score files, for analysis.calibrate."""
    benign = {"scores": [{"item_id": r["item_id"], "adversarial_benign": r["adversarial_benign"],
                          "score": r["score"], "latency_ms": r.get("latency_ms", 0.0)}
                         for r in rows if r["kind"] == "v1_benign"]}
    attack = {"scores": [{"id": r["item_id"], "goal": r["goal"], "variant": r["kind"].removeprefix("v1_"),
                          "score": r["score"]}
                         for r in rows if r["kind"] in ("v1_embedded", "v1_payload")]}
    return benign, attack


def compare_scores(fresh: list[dict], v1_attack: dict, v1_benign: dict, *, tol: float = 1e-4) -> dict:
    """Item by item: fresh score vs the v1 score of the same id (adaptive variants are not rescored)."""
    v1 = {r["id"]: r for r in v1_attack["scores"] if r["variant"] in ("embedded", "payload")}
    v1 |= {r["item_id"]: r for r in v1_benign["scores"]}
    got = {r["item_id"]: r for r in fresh if r["kind"] in V1_KINDS}
    diffs, drifted, changed = [], [], []
    for iid, r in got.items():
        old = v1.get(iid)
        if old is None:
            continue
        if "sha" in old and old["sha"] != r["sha"]:
            changed.append(iid)
            continue
        d = abs(r["score"] - old["score"])
        diffs.append(d)
        if d > tol:
            drifted.append(iid)
    return {"matched": len(diffs), "drifted": sorted(drifted), "text_changed": sorted(changed),
            "missing_fresh": sorted(set(v1) - set(got)), "missing_v1": sorted(set(got) - set(v1)),
            "max_abs_diff": max(diffs) if diffs else None, "tol": tol}


CAL_FIELDS = ("threshold", "fpr_k", "fpr_n", "tpr_k", "tpr_n")


def compare_calibration(fresh: dict, v1: dict) -> list[dict]:
    """v1's calibrate() on fresh scores vs the committed v1 calibration, per language and point."""
    out = []
    for lang, f in fresh["langs"].items():
        o = v1["langs"].get(lang)
        if o is None:
            continue
        row = {"lang": lang, "auc_v1": o["auc"], "auc_now": f["auc"]}
        for point in ("shipped", "calibrated", "matched"):
            for k in CAL_FIELDS:
                row[f"{point}.{k}"] = (o[point][k], f[point][k])
        row["same"] = (o["auc"] == f["auc"] or abs((o["auc"] or 0) - (f["auc"] or 0)) < 1e-9) and all(
            a == b or (isinstance(a, float) and abs(a - b) < 1e-6) for k, (a, b) in row.items() if "." in k)
        out.append(row)
    return out


def reproduce_v1(name: str, langs: list[str], set_name: str) -> dict:
    from analysis.calibrate import calibrate

    v1_cal = json.loads((Path("analysis/out/calibration") / f"{name}.json").read_text(encoding="utf-8"))
    sources, items = {}, {}
    for lang in langs:
        fresh = json.loads(out_path(set_name, lang, name).read_text(encoding="utf-8"))["scores"]
        a = json.loads((V1_ATTACK / f"{lang}__{name}.json").read_text(encoding="utf-8"))
        b = json.loads((V1_BENIGN / f"{lang}__{name}.json").read_text(encoding="utf-8"))
        items[lang] = compare_scores(fresh, a, b)
        sources[lang] = v1_views(fresh)
    cal = calibrate(name, v1_cal["target_fpr"], v1_cal["variant"], sources=sources)
    report = {"defense": name, "set": set_name, "target_fpr": v1_cal["target_fpr"], "items": items,
              "english_threshold": (v1_cal["english_threshold"], cal["english_threshold"]),
              "calibration": compare_calibration(cal, v1_cal)}
    report["reproduced"] = (all(not v["drifted"] and not v["text_changed"] and not v["missing_fresh"]
                                for v in items.values())
                            and all(r["same"] for r in report["calibration"]))
    path = OUT / set_name / f"reproduce_v1__{name}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def print_reproduction(r: dict) -> None:
    print(f"\n{r['defense']}: v1 vs now ({'REPRODUCED' if r['reproduced'] else 'DIFFERS'})")
    print(f"{'lang':<9}{'items':>7}{'drifted':>9}{'changed':>9}{'missing':>9}{'max |d|':>11}")
    for lang, v in r["items"].items():
        m = v["max_abs_diff"]
        print(f"{lang:<9}{v['matched']:>7}{len(v['drifted']):>9}{len(v['text_changed']):>9}"
              f"{len(v['missing_fresh']):>9}{'-' if m is None else f'{m:.1e}':>11}")
    print(f"\ncalibration at {r['target_fpr']:.0%} FPR (v1 -> now); English threshold "
          f"{r['english_threshold'][0]:.6f} -> {r['english_threshold'][1]:.6f}")
    print(f"{'lang':<9}{'AUC':>15} | {'calibrated FPR':>15}{'TPR':>15} | {'shipped TPR':>13} | same")
    for c in r["calibration"]:
        fk, fn = c["calibrated.fpr_k"], c["calibrated.fpr_n"]
        tk, tn = c["calibrated.tpr_k"], c["calibrated.tpr_n"]
        sk = c["shipped.tpr_k"]
        print(f"{c['lang']:<9}{c['auc_v1']:>7.4f}>{c['auc_now']:<7.4f} | {fk[0]}/{fn[0]}>{fk[1]}/{fn[1]:<8}"
              f"{tk[0]}/{tn[0]}>{tk[1]}/{tn[1]:<6} | {sk[0]:>5}>{sk[1]:<6} | {'yes' if c['same'] else 'NO'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="v2", choices=sorted(SETS))
    ap.add_argument("--langs", nargs="+", default=list(LANGS_V2))
    ap.add_argument("--defenses", nargs="+", default=["protectai_deberta"])
    ap.add_argument("--kinds", nargs="+", help="score only these item kinds")
    ap.add_argument("--reproduce-v1", action="store_true",
                    help="score the v1 items and compare with v1's score files and calibration")
    ap.add_argument("--device", default="cpu", help="'cpu' or a CUDA index")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    kinds = tuple(V1_KINDS if args.reproduce_v1 else args.kinds or ()) or None
    device = -1 if args.device == "cpu" else int(args.device)
    for name in args.defenses:
        from defenses.registry import build

        defense = build(name, device=device)
        for lang in args.langs:
            p = run(name, lang, args.set, defense=defense, kinds=kinds, rerun=args.rerun)
            d = json.loads(p.read_text(encoding="utf-8"))
            print(f"{lang:<9}{name:<30}{len(d['scores']):>5} items, {d['scored_now']} scored now "
                  f"({d['seconds']:.0f}s) -> {p}", flush=True)
        if args.reproduce_v1:
            print_reproduction(reproduce_v1(name, args.langs, args.set))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
