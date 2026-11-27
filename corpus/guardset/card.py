"""Render the draft dataset card from the build's own files (docs/v2-plan.md, Gate G2).

    python -m corpus.guardset.card            # -> hf/dataset_card.md

Every number in the card is read from `corpus/guardset/out/report.json` (the build),
`corpus/guardset/drafts/work/generate_report.json` and `model.json` (the generation run), the seed
files, and the G2 sample verdicts. Nothing is typed in by hand, so the card regenerates after any
rebuild. No person's name appears: the venue is double-blind, and reviewers are codes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from corpus.guardset import review
from corpus.guardset.build import OUT
from corpus.guardset.generate import LANGS, WORK
from corpus.guardset.seed import load_goals, load_templates

CARD = Path("hf/dataset_card.md")
LANG_NAME = {"en": "English", "hi": "Hindi (Devanagari)", "hi_latn": "Hindi in Latin script",
             "mixed": "Hindi-English code-mixed (Latin script)"}


def _json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _review_lines(out: Path) -> list[str]:
    old = review.OUT
    try:
        review.OUT = out
        st = review.status()
    finally:
        review.OUT = old
    if not any(s["reviewed"] for s in st.values()):
        return ["The fixed-seed 5% sample check (Gate G2) is **not reviewed yet**."]
    lines = ["| language | sample | reviewed | wrong | error rate [95% CI] | reviewers |", "|---|---|---|---|---|---|"]
    for lang, s in st.items():
        e = "-" if s["error_rate"] is None else f"{s['error_rate']:.1%} [{s['error_ci'][0]:.1%}-{s['error_ci'][1]:.1%}]"
        lines.append(f"| {lang} | {s['sample']} | {s['reviewed']} | {s['wrong']} | {e} | {', '.join(s['reviewers'])} |")
    return lines


def render(*, out: Path = OUT, work: Path = WORK) -> str:
    rep = _json(out / "report.json")
    if not rep:
        raise SystemExit(f"{out}/report.json missing: build first (python -m corpus.guardset.build)")
    gen, model = _json(work / "generate_report.json"), _json(work / "model.json")
    goals, templates = load_goals(), load_templates()
    c_tr, c_va = rep["counts"]["train"], rep["counts"]["validation"]
    langs = [l for l in LANGS if any(k.startswith(f"{l}/") for k in c_tr | c_va)]
    total = sum(c_tr.values()) + sum(c_va.values())
    st, lk = rep["stages"], rep["leakage"]

    rows = ["| language | train: injection | train: benign | train: adversarial-benign | validation: injection "
            "| validation: benign | validation: adversarial-benign |", "|---|---|---|---|---|---|---|"]
    for l in langs:
        rows.append(f"| {l} | {c_tr.get(f'{l}/injection', 0)} | {c_tr.get(f'{l}/benign', 0)} | {c_tr.get(f'{l}/benign/adv', 0)} "
                    f"| {c_va.get(f'{l}/injection', 0)} | {c_va.get(f'{l}/benign', 0)} | {c_va.get(f'{l}/benign/adv', 0)} |")
    gen_rows = []
    for stage in ("payloads", "bodies", "advbenign"):
        for l, s in sorted(gen.get(stage, {}).items()):
            reasons = ", ".join(f"{k} {v}" for k, v in s.get("reasons", {}).items()) or "-"
            gen_rows.append(f"| {stage} | {l} | {s['kept']} | {s['rejected']} | {reasons} |")
    domains = Counter(g.domain for g in goals)

    return "\n".join([
        "---",
        "license: other   # set at Gate G0 from docs/licences.md before the dataset is public",
        "language: [en, hi]",
        "task_categories: [text-classification]",
        "tags: [prompt-injection, indirect-prompt-injection, security, code-mixing, hinglish]",
        "pretty_name: bhasha-shield guard training set (draft)",
        "---",
        "",
        "# bhasha-shield guard training set (draft card)",
        "",
        "Labelled training data for a classifier that flags **indirect prompt injection in tool output** "
        "(emails, web pages, chats, documents, API JSON) in English, Hindi, Romanized Hindi and Hindi-English "
        f"code-mixed text. {total} items in this build. **Every text is machine-generated**; none is human-written.",
        "",
        "## Languages",
        "",
        *[f"- `{l}`: {LANG_NAME.get(l, l)}" for l in langs],
        "",
        "## Size",
        "",
        *rows,
        "",
        f"Goals: {rep['goals']['train']} in train, {rep['goals']['validation']} in validation. "
        f"Template families: {len(rep['families']['train'])} in train, {len(rep['families']['validation'])} in validation.",
        "",
        "## Fields",
        "",
        "`item_id`, `text`, `label` (`injection` / `benign`), `lang`, `family` (template family, or `ordinary` / "
        "`adversarial_benign`), `goal` (attacker goal id), `group` (benign source group), `carrier` (where the text "
        "sits: email, web, document, chat, sms, calendar, review, json, or none for a bare payload), "
        "`adversarial_benign`, `source` (generator and model id), `augmentations`, `provenance`.",
        "",
        "## How it was made",
        "",
        f"1. **Seeds (English, written for this dataset with an LLM assistant, not by hand):** {len(goals)} attacker goals over "
        f"{len(domains)} domains ({', '.join(f'{d} {n}' for d, n in sorted(domains.items()))}) and "
        f"{len(templates)} templates in {len({t.family for t in templates})} families, none shared with the test set.",
        f"2. **Localization:** English payloads -> Hindi"
        f"{' (an IndicTrans2 draft `' + model['indictrans2'] + '`, then' if model.get('indictrans2') else ' (by'} "
        f"`{model.get('model', 'unknown')}` at revision `{model.get('revision', 'unknown')}`), then rewritten as "
        "Romanized Hindi and code-mixed text by the same model.",
        "3. **Benign text:** ordinary carrier texts and adversarial-benign texts (they sound like instructions but are "
        "harmless) generated by the same model from topic lists, in all four languages. No external corpus is used.",
        "4. **Checks on every output:** amounts, ids, e-mail and UPI addresses and links survive unchanged and are not "
        "repeated; the script is right for the language; Romanized and code-mixed text contains enough common Hindi "
        "words not to be plain English; JSON carriers parse; no prompt echo or chatter. Failures are retried twice, "
        "then dropped and counted:",
        "",
        "| stage | language | kept | rejected | reasons |",
        "|---|---|---|---|---|",
        *gen_rows,
        "",
        f"5. **Assembly and augmentation:** payloads spliced into carriers or kept bare; augmented copies "
        f"(homoglyphs, zero-width characters, script mixing) at rate {rep['params']['augment_rate']}: {st['augmented']} items.",
        f"6. **Clean-up:** near-duplicate removal (threshold {rep['params']['dedup_threshold']}): {st['dedup_removed']} removed; "
        f"train/validation split by template family and benign group: {round(rep['params']['val_frac'] * 100)}% of each "
        f"go to validation (families {', '.join(f'`{f}`' for f in rep.get('held_out_families', [])) or 'none'}), so validation "
        "measures unseen templates; attacker goals may appear on both sides. None of them is a test-set goal.",
        "",
        "## Separation from the test set",
        "",
        "The frozen, human-verified test set is separate and is not part of this dataset. No training item uses a "
        "test-set attacker goal or template family (checked against the test set's manifest, by base family name). "
        f"A TF-IDF leakage audit against every test item dropped {st['leakage_dropped']} items at threshold "
        f"{lk['threshold']}; after it, {lk['after']} training items are above the threshold.",
        "",
        "## Quality check",
        "",
        *_review_lines(out),
        "",
        "## Intended use and limitations",
        "",
        "- For training and validating prompt-injection detectors for tool-using agents. Not for building attacks.",
        "- All text is synthetic. A model trained on it can learn generator habits; its real performance is measured "
        "on the separate human-verified test set, not on this data.",
        "- The language checks are heuristics (script and common-word share); the sample check above estimates what "
        "they miss.",
        "- Hindi is the only Indian language here; Romanized and code-mixed text follow one model's idea of them.",
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--work", type=Path, default=WORK)
    a = ap.parse_args(argv)
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text(render(out=a.out, work=a.work), encoding="utf-8", newline="\n")
    print(f"wrote {CARD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
