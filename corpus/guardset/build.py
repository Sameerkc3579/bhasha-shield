"""Assemble the guard's training set from generated drafts (docs/v2-plan.md, Phase 2 step 8).

    python -m corpus.guardset.build                          # drafts -> corpus/guardset/out/
    python -m corpus.guardset.build --test corpus/testset/pilot   # before G1: audit against the pilot

Order: refuse any draft that uses a test-set goal or template family (the frozen manifest) ->
near-duplicate removal -> train/validation split by goal and family -> augmented copies (on the side
of their source) -> TF-IDF leakage audit of both sides against every frozen test item. The report
counts what each stage removed, so the dataset card can say it.

Drafts (`corpus/guardset/drafts/generated_*.jsonl`, GuardItem rows) and the output are datasets: gitignored,
published on Hugging Face (private at G2). Only the report is evidence for RESULTS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from corpus.adaptive import TRANSFORMS, applicable
from corpus.guardset import augment, dedup, leakage, split
from corpus.guardset.schema import Exclusions, GuardItem
from corpus.testset import OUT as FROZEN, FrozenItem

DRAFTS = Path("corpus/guardset/drafts")
OUT = Path("corpus/guardset/out")


def _u(key: str) -> float:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def augmented(items: list[GuardItem], *, rate: float, salt: str) -> list[GuardItem]:
    """A stable `rate` share of items gets one applicable transform, chosen by hash."""
    out = []
    for it in items:
        if it.augmentations or _u(f"{salt}:aug:{it.item_id}") >= rate:
            continue
        options = [t for t in TRANSFORMS if applicable(t, it.lang)]
        t = options[int(_u(f"{salt}:which:{it.item_id}") * len(options)) % len(options)]
        out.append(augment.apply(it, t, seed=int(_u(f"{salt}:seed:{it.item_id}") * 2**31)))
    return out


def _counts(items: list[GuardItem]) -> dict[str, int]:
    return dict(sorted(Counter(f"{i.lang}/{i.label}{'/adv' if i.adversarial_benign else ''}" for i in items).items()))


def run(drafts: list[GuardItem], *, test_texts: dict[str, list[str]], exclusions: Exclusions,
        val_frac: float = 0.25, dedup_threshold: float = dedup.THRESHOLD,
        leak_threshold: float = leakage.THRESHOLD, augment_rate: float = 0.25,
        salt: str = split.SALT) -> tuple[split.Split, dict]:
    problems = exclusions.violations(drafts)
    if problems:
        raise SystemExit("drafts use the frozen test set's goals or families (fix the generator):\n  "
                         + "\n  ".join(problems[:50]))
    kept, removed = dedup.dedup(drafts, threshold=dedup_threshold)
    s = split.group_split(kept, val_frac=val_frac, salt=salt)
    aug_train, aug_val = augmented(s.train, rate=augment_rate, salt=salt), augmented(s.validation, rate=augment_rate, salt=salt)
    s.train += aug_train
    s.validation += aug_val
    s.train, leak_train = leakage.audit(s.train, test_texts, threshold=leak_threshold)
    s.validation, leak_val = leakage.audit(s.validation, test_texts, threshold=leak_threshold)
    after = leakage.audit(s.train + s.validation, test_texts, threshold=leak_threshold)[1]["dropped"]
    report = {
        "params": {"val_frac": val_frac, "dedup_threshold": dedup_threshold, "leak_threshold": leak_threshold,
                   "augment_rate": augment_rate, "salt": salt, "split_by": ["family", "group"]},
        "held_out_families": [k.split(":", 1)[1] for k in s.held_out.get("family", [])],
        "stages": {"drafts": len(drafts), "dedup_removed": len(removed),
                   "split_dropped_cross": len(s.dropped), "augmented": len(aug_train) + len(aug_val),
                   "leakage_dropped": leak_train["dropped"] + leak_val["dropped"]},
        "counts": {"train": _counts(s.train), "validation": _counts(s.validation)},
        "goals": {"train": len({i.goal for i in s.train if i.goal}), "validation": len({i.goal for i in s.validation if i.goal})},
        "families": {"train": sorted({i.family for i in s.train if i.label == "injection"}),
                     "validation": sorted({i.family for i in s.validation if i.label == "injection"})},
        "dedup_removed": removed,
        "leakage": {"threshold": leak_threshold, "after": after,
                    "histogram_train": leak_train["histogram"], "histogram_validation": leak_val["histogram"],
                    "dropped": leak_train["dropped_rows"] + leak_val["dropped_rows"]},
    }
    return s, report


def load_drafts(d: Path) -> list[GuardItem]:
    """The generator's `generated_{lang}.jsonl` files only: `seed_en.jsonl` shares the folder."""
    return [GuardItem.model_validate_json(l) for f in sorted(d.glob("generated_*.jsonl"))
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_test_texts(d: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(d.glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                it = FrozenItem.model_validate_json(l)
                out.setdefault(it.lang, []).append(it.text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=Path, default=DRAFTS)
    ap.add_argument("--test", type=Path, default=FROZEN, help="frozen test set dir (manifest.json + {lang}.jsonl)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--augment-rate", type=float, default=0.25)
    ap.add_argument("--leak-threshold", type=float, default=leakage.THRESHOLD)
    args = ap.parse_args()

    if not (args.test / "manifest.json").exists():
        raise SystemExit(f"{args.test}/manifest.json not found: build the test set first "
                         "(python -m corpus.testset build, or build --pilot and pass --test corpus/testset/pilot)")
    manifest = json.loads((args.test / "manifest.json").read_text(encoding="utf-8"))
    s, report = run(load_drafts(args.drafts), test_texts=load_test_texts(args.test),
                    exclusions=Exclusions.from_manifest(args.test / "manifest.json"),
                    val_frac=args.val_frac, augment_rate=args.augment_rate, leak_threshold=args.leak_threshold)
    report["test_set"] = {"dir": str(args.test), "pilot": manifest.get("pilot"), "files": manifest.get("files")}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", s.train), ("validation", s.validation)):
        (args.out / f"{name}.jsonl").write_bytes("".join(r.model_dump_json() + "\n" for r in rows).encode("utf-8"))
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["stages"]), "\n", json.dumps(report["counts"], indent=1))
    if manifest.get("pilot"):
        print("audited against the PILOT test set: Gate G2 needs the frozen one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
