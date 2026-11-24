"""Write a guard run's model card and upload the run to a private Hugging Face repo (Phase 6 step 1).

    python -m guard.push_hub --run guard/runs/mdeberta__full__s1 --card-only     # -> hf/model_card.md
    python -m guard.push_hub --run guard/runs/mdeberta__full__s1 --repo <user>/bhasha-guard --yes

The card is rendered from files: the run's `train_log.json`, the selection on the calibration half
(`analysis/out/guard_eval/v2/selection.json`), the classifier table (`tables.json`, test half, read
once) and the latency file (`analysis/out/latency/<run>.json`). A section whose file is missing says
so instead of showing a number. No person's name appears (double-blind review).

Uploading publishes outside this machine, so it needs `--yes`; the repo is created **private**.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CARD = Path("hf/model_card.md")
EVAL = Path("analysis/out/guard_eval/v2")
LATENCY = Path("analysis/out/latency")
LANG_ORDER = ("en", "hi", "hi_latn", "mixed")


def _json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render(run: Path, *, eval_dir: Path = EVAL, latency_dir: Path = LATENCY, target: float = 0.02) -> str:
    log = _json(run / "train_log.json")
    if not log:
        raise SystemExit(f"{run}/train_log.json missing: not a finished guard run")
    label = f"bhasha_guard@{log['run']}"
    sel, tables, lat = _json(eval_dir / "selection.json"), _json(eval_dir / "tables.json"), _json(latency_dir / f"{log['run']}.json")
    cfg, ct = log["config"], log["counts"]

    lines = ["---", "license: other   # set from docs/licences.md (base model and data terms) before release",
             "language: [en, hi]", "pipeline_tag: text-classification",
             "tags: [prompt-injection, indirect-prompt-injection, security, hinglish]",
             f"base_model: {cfg['base_model']}", "---", "",
             f"# bhasha-shield guard ({log['run']})", "",
             "Flags **indirect prompt injection in tool output** (emails, web pages, chats, documents, API JSON) in "
             "English, Hindi, Romanized Hindi and Hindi-English code-mixed text. Labels: `BENIGN`, `INJECTION`. "
             "Long inputs are split into 1500-character chunks with 200 overlap and the maximum score is used.", "",
             "## Training", "",
             f"- Base model: `{cfg['base_model']}`; seed {log['seed']}; ablation `{log['ablation']}`.",
             f"- Learning rate {cfg['lr']}, up to {cfg['epochs']} epochs, max length {cfg['max_length']}, batch "
             f"{cfg['batch_size']}, class-weighted loss, best epoch by validation PR-AUC "
             f"({log['val_pr_auc'] if log['val_pr_auc'] is None else round(log['val_pr_auc'], 4)}).",
             f"- Data: {ct['train']['injection']} injection and {ct['train']['benign']} benign training items "
             f"(synthetic; see the dataset card), build report sha256 `{log['data_report_sha256'][:12]}…`.",
             f"- Libraries: {', '.join(f'{k} {v}' for k, v in log.get('versions', {}).items() if v)}.", ""]

    lines += ["## How it was chosen", ""]
    if sel and sel.get("winner"):
        lines.append(f"Picked on the **calibration half** of the frozen test set only ({sel['criterion']}, at "
                     f"{sel['target_fpr']:.0%} FPR): ranking {', '.join(sel['ranking'])}; winner `{sel['winner']}`.")
    else:
        lines.append("No selection file yet (`python -m analysis.guard_select`).")
    lines += ["", f"## Evaluation (frozen human-verified test set, test half, {target:.0%} FPR)", ""]
    t = (tables or {}).get("defenses", {}).get(label)
    if t:
        lines += ["Thresholds fitted per language on the calibration half; TPR 95% CI is a bootstrap over attacker goals.", "",
                  "| language | TPR [95% CI] | FPR ordinary | FPR adversarial-benign | AUC |", "|---|---|---|---|---|"]
        for lang in [l for l in LANG_ORDER if l in t]:
            c = t[lang]["at"][str(target)]["calibrated"]
            ci = "" if c["tpr_ci"][0] is None else f" [{_pct(c['tpr_ci'][0])}-{_pct(c['tpr_ci'][1])}]"
            auc = "-" if c["auc"] is None else f"{c['auc']:.3f}"
            lines.append(f"| {lang} | {_pct(c['tpr'])}{ci} | {c['fpr_ordinary_k']}/{c['fpr_ordinary_n']} | "
                         f"{c['fpr_adversarial_k']}/{c['fpr_adversarial_n']} | {auc} |")
    else:
        lines.append(f"Not evaluated yet (`python -m analysis.guard_tables --defenses {label} ...`).")
    lines += ["", "## CPU latency", ""]
    if lat:
        lines.append(f"One check (chunking + model) on {lat['texts']} test-set texts, {lat['threads']} threads, "
                     f"{lat['machine']}: fp32 median {lat['fp32']['median_ms']:.1f} ms (p95 {lat['fp32']['p95_ms']:.1f}); "
                     f"int8 ONNX median {lat['int8']['median_ms']:.1f} ms (p95 {lat['int8']['p95_ms']:.1f}).")
    else:
        lines.append("Not measured yet (`python -m guard.export_onnx --bench`).")
    lines += ["", "## Limitations", "",
              "- Trained only on machine-generated text; the test-set numbers above are the ones to trust.",
              "- Calibrated thresholds are per language; a language not listed falls back to the English one.",
              "- A bounded adaptive attacker lowers detection (see the paper); it is not robust to a determined one.",
              "- Not a complete defense: use it with the proxy's provenance tagging and sink rule.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--repo", help="Hugging Face repo id to create (private) and upload to")
    ap.add_argument("--card-only", action="store_true")
    ap.add_argument("--yes", action="store_true", help="really upload (publishes outside this machine)")
    a = ap.parse_args(argv)
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text(render(a.run), encoding="utf-8", newline="\n")
    print("wrote", CARD)
    if a.card_only:
        return 0
    if not (a.repo and a.yes):
        raise SystemExit("upload needs --repo <id> and --yes")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, private=True, exist_ok=True)
    folder = a.run / "model"
    api.upload_folder(repo_id=a.repo, folder_path=str(folder), commit_message=f"guard run {a.run.name}")
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md", repo_id=a.repo)
    print(f"uploaded {folder} to {a.repo} (private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
