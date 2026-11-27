"""Pick the winning guard base model on the calibration half only (docs/v2-plan.md, Phase 3 step 5).

    python -m analysis.guard_select                       # every bhasha_guard@<run> score file in v2
    python -m analysis.guard_select --target 0.02

For each run (`bhasha_guard@<model>__full__s<seed>`, scored by `scripts/guard_eval.py`) and language:
the threshold is fitted on the calibration-half negatives (as `analysis/guard_tables.py` does), and
TPR is read on the calibration-half headline positives, which the threshold never saw. Test-half
rows are never read, so the choice cannot depend on them (a test checks this).

Criterion, matching Gate G3 ("beats calibrated PromptGuard 2 in every non-English language"): the
mean over seeds of the **worst** non-English calibration TPR; ties go to the higher mean over all
languages. Only `full` runs compete; ablations are read later, for the winner.
Output: `analysis/out/guard_eval/v2/selection.json`.
"""

from __future__ import annotations

import argparse
import json
import re
from statistics import mean

from analysis.guard_tables import OUT, _pos, load, threshold

RUN = re.compile(r"^bhasha_guard@(?P<model>[\w-]+)__(?P<ablation>[\w]+)__s(?P<seed>\d+)$")
LANGS = ("en", "hi", "hi_latn", "mixed")


def parse(label: str) -> tuple[str, str, int] | None:
    m = RUN.match(label)
    return (m["model"], m["ablation"], int(m["seed"])) if m else None


def calibration_tpr(rows: list[dict], target: float) -> float | None:
    cal = [r for r in rows if r["half"] == "calibration"]
    t = threshold(cal, target)
    pos = _pos(cal, "calibration")
    return sum(r["score"] >= t for r in pos) / len(pos) if pos else None


def select(scores: dict[str, dict[str, list[dict]]], *, target: float = 0.02) -> dict:
    """scores: run label -> lang -> score rows."""
    runs: dict[str, dict] = {}
    for label, by_lang in sorted(scores.items()):
        p = parse(label)
        if not p or p[1] != "full":
            continue
        tpr = {lang: calibration_tpr(rows, target) for lang, rows in sorted(by_lang.items())}
        runs[label] = {"model": p[0], "seed": p[2], "tpr": tpr}
    models: dict[str, dict] = {}
    for label, r in runs.items():
        models.setdefault(r["model"], {"runs": []})["runs"].append(label)
    for m, d in models.items():
        worst, avg = [], []
        for label in d["runs"]:
            t = runs[label]["tpr"]
            non_en = [v for k, v in t.items() if k != "en" and v is not None]
            worst.append(min(non_en) if non_en else None)
            avg.append(mean(v for v in t.values() if v is not None) if any(v is not None for v in t.values()) else None)
        d["seeds"] = len(d["runs"])
        d["worst_non_en"] = mean(w for w in worst if w is not None) if any(w is not None for w in worst) else None
        d["mean_all"] = mean(a for a in avg if a is not None) if any(a is not None for a in avg) else None
    ranked = sorted((m for m in models if models[m]["worst_non_en"] is not None),
                    key=lambda m: (-models[m]["worst_non_en"], -(models[m]["mean_all"] or 0), m))
    return {"target_fpr": target, "half": "calibration", "criterion": "mean over seeds of the worst non-English TPR",
            "winner": ranked[0] if ranked else None, "ranking": ranked, "models": models, "runs": runs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="v2")
    ap.add_argument("--target", type=float, default=0.02)
    a = ap.parse_args()
    labels = sorted({p.name.split("__", 1)[1].removesuffix(".json")
                     for p in (OUT / a.set).glob("*__bhasha_guard@*.json")})
    scores = {lab: {l: r for l in LANGS if (r := load(a.set, l, lab))} for lab in labels}
    res = select(scores, target=a.target)
    (OUT / a.set / "selection.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"calibration half, {a.target:.0%} FPR; criterion: {res['criterion']}")
    print(f"{'model':<16}{'seeds':>6}{'worst non-en':>14}{'mean, all':>11}")
    for m in res["ranking"]:
        d = res["models"][m]
        print(f"{m:<16}{d['seeds']:>6}{d['worst_non_en']:>14.3f}{d['mean_all'] or 0:>11.3f}")
    print("winner:", res["winner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
