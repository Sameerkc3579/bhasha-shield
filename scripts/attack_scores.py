"""Class A classifiers over injected text -- the detection side of the frontier.

    python -m scripts.attack_scores --langs en hi hi_latn bn mixed \
        --defenses protectai_deberta translit+protectai_deberta

The counterpart of `scripts/benign_fpr.py`. No agent and no vLLM: a classifier's
verdict on a tool output does not depend on what the agent does next, so the
positives for threshold calibration (Phase 6) can be scored offline, on CPU.
`benign_fpr` gives FPR(t); this gives TPR(t); `analysis/calibrate.py` joins them.

Per case, three kinds of positive:

- `payload`: the rendered attack template alone (one per distinct goal).
- `embedded`: the payload spliced into a benign tool output of the same
  language at a paragraph boundary near the middle -- the position AgentDojo
  injects into is inside real content, and a detector that only fires on a
  bare payload is not much of a detector. The carrier is chosen by a stable
  hash of the case id, from the ordinary items harvested from AgentDojo --
  pinned to that pool so growing the benign corpus never changes a positive.
- `adaptive:<transform>`: the embedded text after one `corpus/adaptive.py`
  rewrite, where the transform applies to that language.

Non-English templates and goals are still unverified machine drafts, so every
non-`en` file is stamped `unverified_corpus: true` -- a pilot, not a result.

Output: `analysis/out/attack_scores/{lang}__{defense}.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from corpus.adaptive import TRANSFORMS, applicable
from corpus.adaptive import apply as adapt
from corpus.attacks import DEFAULT_ATTACK, TEMPLATE_STATUS, render
from corpus.schema import BenignItem, InjectionCase, ToolOutput
from scripts.benign_fpr import load as load_benign

OUT = Path("analysis/out/attack_scores")


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def load_cases(lang: str) -> list[InjectionCase]:
    en = [InjectionCase.model_validate_json(l)
          for l in Path("corpus/injections/en.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if lang == "en":
        return en
    from corpus.goals import localize_cases

    return localize_cases(en, lang, allow_unverified=True)


def embed(carrier: str, payload: str) -> str:
    """Splice at the paragraph break nearest the middle, else at the end."""
    breaks = [i for i in range(len(carrier)) if carrier.startswith("\n\n", i)]
    if not breaks:
        return carrier + payload
    mid = min(breaks, key=lambda i: abs(i - len(carrier) // 2))
    return carrier[:mid] + payload + carrier[mid:]


def goal_key(case: InjectionCase) -> str:
    return f"{case.domain}/{(case.source_case_id or '').split('/')[-1]}"


def positives(lang: str, attack: str = DEFAULT_ATTACK) -> list[dict]:
    """Every positive text for `lang`, with ids that pair across languages."""
    cases = load_cases(lang)
    carriers: list[BenignItem] = [b for b in load_benign(lang)
                                  if not b.adversarial_benign and b.source.startswith("agentdojo/")]
    rows: list[dict] = []
    seen_goals: set[str] = set()
    for c in cases:
        payload = render(c.attacker_goal_nl, attack=attack, lang=lang, allow_unverified=True)
        g = goal_key(c)
        if g not in seen_goals:
            seen_goals.add(g)
            rows.append({"id": f"payload/{g}", "goal": g, "variant": "payload", "text": payload})
        carrier = carriers[_h(c.case_id) % len(carriers)]
        text = embed(carrier.content, payload)
        rows.append({"id": f"embedded/{c.case_id}", "goal": g, "variant": "embedded",
                     "carrier": carrier.item_id, "text": text})
        for t in TRANSFORMS:
            if applicable(t, lang):
                rows.append({"id": f"adaptive:{t}/{c.case_id}", "goal": g, "variant": f"adaptive:{t}",
                             "carrier": carrier.item_id, "text": adapt(t, text, seed=_h(c.case_id))})
    return rows


def score_rows(defense, rows: list[dict], lang: str) -> list[dict]:
    from defenses.base import RunContext

    out = []
    for r in rows:
        ctx = RunContext(case_id=r["id"], lang=lang, model="none", user_task="")
        v = defense.inspect(ToolOutput(tool_name="attack", content=r["text"], lang=lang), ctx)
        out.append({k: r[k] for k in ("id", "goal", "variant") if k in r}
                   | {"score": v.score, "latency_ms": round(v.latency_ms, 2)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--defenses", nargs="+", default=["protectai_deberta"])
    ap.add_argument("--attack", default=DEFAULT_ATTACK)
    ap.add_argument("--device", default="cpu", help="'cpu' or a CUDA index")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    from defenses.registry import CLASSIFIERS, build, is_known

    device = -1 if args.device == "cpu" else int(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'lang':<9}{'defense':<32}{'variant':<22}{'TPR@0.5':>12}")
    for name in args.defenses:
        if not is_known(name) or name.split("+")[-1] not in CLASSIFIERS or name.startswith("translate+"):
            raise SystemExit(f"{name}: only local classifiers (optionally fold+/translit+) run here")
        defense = build(name, device=device)
        for lang in args.langs:
            f = OUT / f"{lang}__{name}.json"
            if f.exists() and not args.rerun:
                scores = json.loads(f.read_text(encoding="utf-8"))["scores"]
            else:
                t0 = time.time()
                scores = score_rows(defense, positives(lang, args.attack), lang)
                f.write_text(json.dumps({
                    "lang": lang, "defense": name, "attack": args.attack,
                    "model_id": getattr(defense, "model_id", ""),
                    "unverified_corpus": lang != "en" and not TEMPLATE_STATUS[lang]["verified_by"],
                    "seconds": round(time.time() - t0, 1), "scores": scores},
                    indent=1, ensure_ascii=False), encoding="utf-8")
            for variant in sorted({s["variant"] for s in scores}):
                sel = [s for s in scores if s["variant"] == variant]
                k = sum(s["score"] >= 0.5 for s in sel)
                print(f"{lang:<9}{name:<32}{variant:<22}{k:>5}/{len(sel):<4}{k / len(sel):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
