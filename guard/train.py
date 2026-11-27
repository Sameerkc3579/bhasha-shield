"""Train one guard run: one base model x one seed x one ablation (docs/v2-plan.md, Phase 3 steps 3-4).

    python -m guard.train --config guard/configs/mdeberta.yaml --seed 1                 # GPU
    python -m guard.train --config guard/configs/muril.yaml --seed 2 --ablation en_only
    python -m guard.train --config guard/configs/mdeberta.yaml --fake                   # CI / laptop

Input: `corpus/guardset/out/{train,validation}.jsonl` from `python -m corpus.guardset.build`. A build
whose report is missing, or whose leakage audit left any item above the threshold, is refused.
The frozen test set is never read here: thresholds and the test-half numbers are
`scripts/guard_eval.py` + `analysis/guard_tables.py`, after training.

Settings come from the config (the plan's defaults: lr 2e-5, 3 epochs, max length 512, batch 16,
fp16); the loss is class-weighted, and the checkpoint kept is the epoch with the best validation
PR-AUC (early stopping with patience 1). Labels: 0 = BENIGN, 1 = INJECTION.

Ablations (step 4) change the training data only, except `en_only`, which also limits validation
to English so that no other language steers early stopping:
`full`, `en_only`, `no_adv` (no adversarial-benign items), `no_aug` (no augmented copies),
`frac25`, `frac50` (a stable hashed share of the training items; frac25 is inside frac50).

Output: `guard/runs/{name}__{ablation}__s{seed}/` with `train_log.json` (config, seed, data counts,
per-epoch validation PR-AUC, library versions, the build report's hash) and the model (`model/`,
or `fake_model.pkl` for `--fake`, a char n-gram logistic regression that runs the same plumbing).
Weights are gitignored; they go to Hugging Face (private) from the notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from corpus.guardset.schema import GuardItem

CONFIGS = Path(__file__).parent / "configs"
DATA = Path("corpus/guardset/out")
RUNS = Path("guard/runs")
ABLATIONS = ("full", "en_only", "no_adv", "no_aug", "frac25", "frac50")
LABELS = {0: "BENIGN", 1: "INJECTION"}
SALT = "guard-2026"


@dataclass(frozen=True)
class Config:
    name: str
    base_model: str
    lr: float = 2e-5
    epochs: int = 3
    max_length: int = 512
    batch_size: int = 16
    fp16: bool = True
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    patience: int = 1
    revision: str | None = None


def load_config(path: Path) -> Config:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown config keys {unknown}; known: {sorted(known)}")
    c = Config(**raw)
    return Config(**{**asdict(c), "lr": float(c.lr)})   # YAML reads 2e-5 as a string


def _u(key: str) -> float:
    return int(hashlib.sha256(f"{SALT}:{key}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def ablate(train: list[GuardItem], val: list[GuardItem], ablation: str) -> tuple[list[GuardItem], list[GuardItem]]:
    if ablation not in ABLATIONS:
        raise ValueError(f"ablation {ablation!r} not in {ABLATIONS}")
    if ablation == "en_only":
        return [i for i in train if i.lang == "en"], [i for i in val if i.lang == "en"]
    if ablation == "no_adv":
        return [i for i in train if not i.adversarial_benign], val
    if ablation == "no_aug":
        return [i for i in train if not i.augmentations], val
    if ablation.startswith("frac"):
        share = int(ablation[4:]) / 100
        return [i for i in train if _u(i.item_id) < share], val
    return train, val


def class_weights(labels: list[int]) -> dict[int, float]:
    """n / (2 * count): each label carries half of the total loss weight."""
    c = Counter(labels)
    return {y: len(labels) / (2 * c[y]) for y in (0, 1) if c[y]}


def pr_auc(labels: list[int], scores: list[float]) -> float | None:
    """Average precision; None without positives."""
    pos = sum(labels)
    if not pos:
        return None
    hits, total = 0, 0.0
    for k, (_, y) in enumerate(sorted(zip(scores, labels), key=lambda t: -t[0]), 1):
        if y:
            hits += 1
            total += hits / k
    return total / pos


def _read(p: Path) -> list[GuardItem]:
    return [GuardItem.model_validate_json(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_data(d: Path) -> tuple[list[GuardItem], list[GuardItem], str]:
    rp = d / "report.json"
    if not rp.exists():
        raise SystemExit(f"{rp} missing: train only on a build (python -m corpus.guardset.build)")
    report = json.loads(rp.read_text(encoding="utf-8"))
    after = report.get("leakage", {}).get("after")
    if after != 0:
        raise SystemExit(f"{rp}: the leakage audit left {after} items above the threshold; refusing to train")
    return _read(d / "train.jsonl"), _read(d / "validation.jsonl"), hashlib.sha256(rp.read_bytes()).hexdigest()


def _counts(items: list[GuardItem]) -> dict:
    return {"injection": sum(i.label == "injection" for i in items), "benign": sum(i.label == "benign" for i in items),
            "by_lang": dict(sorted(Counter(f"{i.lang}/{i.label}" for i in items).items()))}


# ------------------------------------------------------------------ trainers

def train_fake(train: list[GuardItem], val: list[GuardItem], run: Path, seed: int) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1),
                          LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    model.fit([i.text for i in train], [int(i.label == "injection") for i in train])
    (run / "fake_model.pkl").write_bytes(pickle.dumps(model))
    scores = model.predict_proba([i.text for i in val])[:, 1].tolist() if val else []
    return {"trainer": "fake", "epochs": [{"epoch": 1, "val_pr_auc": pr_auc([int(i.label == "injection") for i in val], scores)}],
            "val_scores": scores, "versions": {"sklearn": __import__("sklearn").__version__}}


def train_hf(cfg: Config, train: list[GuardItem], val: list[GuardItem], run: Path, seed: int) -> dict:
    import numpy as np
    import torch
    import transformers
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer,
                              TrainingArguments, set_seed)

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, revision=cfg.revision, num_labels=2, id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()}, ignore_mismatched_sizes=True)

    class DS(torch.utils.data.Dataset):
        def __init__(self, items):
            self.enc = tok([i.text for i in items], truncation=True, max_length=cfg.max_length)
            self.y = [int(i.label == "injection") for i in items]

        def __len__(self):
            return len(self.y)

        def __getitem__(self, k):
            return {**{n: v[k] for n, v in self.enc.items()}, "labels": self.y[k]}

    ds_train, ds_val = DS(train), DS(val)
    w = class_weights(ds_train.y)
    weight = torch.tensor([w.get(0, 1.0), w.get(1, 1.0)], dtype=torch.float)

    class Weighted(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=weight.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(p):
        logits, y = p
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float), -1)[:, 1].tolist()
        return {"pr_auc": pr_auc(list(map(int, y)), probs) or 0.0}

    args = TrainingArguments(
        output_dir=str(run / "ckpt"), learning_rate=cfg.lr, num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size * 2,
        weight_decay=cfg.weight_decay, warmup_ratio=cfg.warmup_ratio, fp16=cfg.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="pr_auc", greater_is_better=True, logging_steps=50, report_to=[], seed=seed,
        data_seed=seed)
    trainer = Weighted(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                       processing_class=tok, data_collator=transformers.DataCollatorWithPadding(tok),
                       compute_metrics=metrics, callbacks=[EarlyStoppingCallback(cfg.patience)])
    trainer.train()
    trainer.save_model(str(run / "model"))
    tok.save_pretrained(str(run / "model"))
    pred = trainer.predict(ds_val)
    scores = torch.softmax(torch.tensor(pred.predictions, dtype=torch.float), -1)[:, 1].tolist()
    epochs = [{"epoch": round(h["epoch"], 2), "val_pr_auc": h["eval_pr_auc"]}
              for h in trainer.state.log_history if "eval_pr_auc" in h]
    import shutil
    shutil.rmtree(run / "ckpt", ignore_errors=True)   # the best checkpoint is in model/
    return {"trainer": "hf", "epochs": epochs, "best_checkpoint": trainer.state.best_model_checkpoint,
            "val_scores": scores, "base_model_commit": getattr(model.config, "_commit_hash", None),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "numpy": np.__version__, "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=RUNS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ablation", default="full", choices=ABLATIONS)
    ap.add_argument("--fake", action="store_true", help="no GPU: a char n-gram model through the same plumbing")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    run = a.out / f"{cfg.name}__{a.ablation}__s{a.seed}"
    if (run / "train_log.json").exists():
        raise SystemExit(f"{run} exists: a finished run is never overwritten (delete it to retrain)")
    train, val, report_sha = load_data(a.data)
    train, val = ablate(train, val, a.ablation)
    run.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = train_fake(train, val, run, a.seed) if a.fake else train_hf(cfg, train, val, run, a.seed)
    best = max((e["val_pr_auc"] for e in res["epochs"] if e["val_pr_auc"] is not None), default=None)
    log = {"run": run.name, "config": asdict(cfg), "seed": a.seed, "ablation": a.ablation,
           "trainer": res.pop("trainer"), "counts": {"train": _counts(train), "validation": _counts(val)},
           "val_pr_auc": best, "epochs": res.pop("epochs"), "data_report_sha256": report_sha,
           "seconds": round(time.time() - t0, 1), "python": platform.python_version(), **res}
    val_scores = log.pop("val_scores")
    (run / "val_scores.jsonl").write_text("".join(
        json.dumps({"item_id": i.item_id, "lang": i.lang, "label": i.label, "score": s}) + "\n"
        for i, s in zip(val, val_scores)), encoding="utf-8")
    (run / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"{run.name}: validation PR-AUC {best if best is None else round(best, 4)} "
          f"({log['counts']['train']['injection']}+{log['counts']['train']['benign']} train items, {log['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
