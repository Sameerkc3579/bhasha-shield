# bhasha-shield v2: Build Plan

Written 9 Nov 2026. Live version with diagrams: https://claude.ai/artifact/KcZpPVy4XLFLgTS5Nid1qS

<!-- Target workshops and deadlines (fill in during Phase 0):
     1.
     2.
     3.
-->

## Goal and finish line

v2 turns bhasha-shield from "we measured the problem" into "we measured it **and fixed it**". It stays in the same repo, under the same name, as one resume entry.

**The four deliverables:**

| # | Deliverable | What it is | Where it lives |
| --- | --- | --- | --- |
| D1 | **Benchmark v2** | Two parts. (a) The existing agent-level benchmark (AgentDojo cases in each language), extended and frozen. (b) A new **classifier dataset**: tool-output snippets labelled *injection* or *benign*, in every language, including harmless text that looks like an instruction. | GitHub + Hugging Face Datasets |
| D2 | **Guard model** | A small multilingual classifier that reads tool output and flags prompt injection. It is trained on D1(b) and must beat PromptGuard 2 and ProtectAI on Indian languages at low false-positive rates. | GitHub (code) + Hugging Face Hub (weights) |
| D3 | **Agent-level proof** | The guard plugged into the agent harness as a defense, run on the full grid. This shows it cuts attack success without killing task completion. | `RESULTS.md` + paper tables |
| D4 | **Paper** | A 4–8 page workshop paper, plus a public preprint. | arXiv + a workshop submission |

**v2 is done when all of these are true:**

- [ ] At least 4 languages are human-verified end to end (en, hi, hi_latn, mixed). Each further language is added only when a fluent reviewer signs off.
- [ ] The guard model's detection at 2% FPR beats the best baseline in **every** non-English language, with a 95% confidence interval that doesn't overlap the baseline's.
- [ ] As an agent defense, the guard lowers utility-normalized attack success (uASR) with a significant paired test (McNemar p < 0.05). Task completion drops by no more than 5 points vs undefended.
- [ ] Dataset and model are public on Hugging Face with cards. The GitHub release `v2.0` reproduces every number in the paper with one command per table.
- [ ] The paper is submitted and the preprint is public.

Only finished, reproducible numbers go on the resume. Until then, the v1 bullets stay as they are.

## Starting point: what v1 already gives us

v1 is finished and well built. The README says "the experiments are complete" as of 9 Nov 2026, with 285 tests. v2 reuses almost all of it, and three v1 findings shape the plan directly.

**Reusable parts of the `agentfw` package:**

| Part | What it does | How v2 uses it |
| --- | --- | --- |
| `harness/` (runner, scoring, capability, cache, trace, budget) | Agent loop with a defense hook at the tool-output boundary; uASR scoring; response cache; JSONL traces | Runs the agent-level grid with the new guard as a defense |
| `corpus/` (schema, attacks, goals, benign, environments, adaptive, provenance, review) | Verified goals and templates for en, hi, hi_latn, mixed; 300 benign items per language; adaptive attacks; provenance log | Becomes the **frozen human-verified test set**; generators extend it |
| `defenses/` (9 defenses + `registry.py`; `classifiers/xlmr_finetuned.py` stub) | Classifier, prompting and architectural defenses | The guard plugs in as a new classifier defense, replacing the dropped `xlmr_finetuned` |
| `analysis/calibrate.py` | Per-language thresholds on a stable 50/50 split (positives split by attacker goal) | The same split and method for every model, so the comparison is fair |
| `analysis/stats.py`, `figures.py` | Wilson CIs, exact McNemar, every table from disk | Reused for all v2 tables |
| `service/proxy.py` | HTTP firewall with per-language thresholds and a sink rule | Ships the guard as a deployable component |
| `notebooks/grid_colab.ipynb` | Resumable free-T4 GPU runs, gated on verification | Pattern for the training and grid notebooks |

**Three v1 findings that shape v2:**

1. **There's no labelled Indic training set.** `xlmr_finetuned` was dropped because no free, verifiable labelled Hindi set exists. *So v2's real work is building one.* That dataset is a contribution in its own right.
2. **PromptGuard 2 ranks well but uses the wrong threshold.** It has AUC 0.94–0.99 in every language, but its English threshold catches 0%. Calibrated per language, it gets 51% in Hindi at 2% FPR and 79% in Romanized Hindi at 0.7%. *So the baseline to beat is calibrated PromptGuard 2, not the out-of-the-box version.* Beating the weak version proves nothing.
3. **Hindi attacks barely succeed against the agent.** uASR is 1/40 with the translated template, but 15/49 when the English template wraps the Hindi goal. *So the agent-level test needs attacks that actually work in Hindi.* v2 uses the English-template version plus new natively written Hindi templates, otherwise there's nothing for the guard to block.

**Still open from v1:** the paper isn't written, the H5 user tasks need human verification, and Bengali stays a pilot (no fluent reviewer). v2 writes **one paper covering v1 and v2 together**: v1 diagnoses the problem, v2 fixes it.

## System architecture

The test set never touches training; it is used once, at evaluation.

```text
DATA      [Generators]  ---->  [Training set]  - - leakage audit: no overlap - -  [FROZEN TEST SET]
          IndicTrans2 +        dedup + group                                      human-verified, hashed
          Qwen 14B; new        split by goal                                      calibration / test half
          goals, carriers      and family                                           |            |
                                   |                                                | calibration| test half
                                   v                                                v   half     |
TRAIN                        [Guard training]  ----------------------------->  [Thresholds]      |
                             4 base models x                                  one per language   |
                             3 seeds, Kaggle T4                               fit on calib. half |
                                                                               |         |       |
                                   +-------------------------------------------+         v       v
                                   v                                                [Classifier eval]
EVALUATE                     [Agent grid]                                           TPR at 2% FPR on
                             AgentDojo + Qwen 14B                                   test half, vs baselines
                             uASR, utility, McNemar                                         |
                                   |                                                        |
                                   v                                                        v
SHIP      [Release: Hugging Face dataset + model . ONNX firewall + Gradio Space . paper on arXiv]
```

There are two data paths. Generated training data feeds the guard. The frozen, human-verified test set feeds only the thresholds (calibration half) and the final scores (test half). The leakage audit keeps the two paths apart. The calibrated guard is then tested twice: on its own as a classifier, and inside the agent as a defense. Everything that passes ships together.

## Tech stack

Everything runs free: open-weight models, a free Colab or Kaggle T4 for GPU work, a laptop CPU for classifier scoring, and free Hugging Face hosting. The v1 stack stays. New pieces are marked **new**. Check every model's licence in Phase 0 before you build on it.

| Layer | Tool | Used for | Why this one |
| --- | --- | --- | --- |
| Language + package | Python 3.10+, the existing `agentfw` package, pydantic schemas | All code | Already there, with frozen data types |
| Agent benchmark | AgentDojo (pinned SHA, in `vendor/`) | Agent-level attack cases and environments | v1 results are built on it; pinning keeps numbers stable |
| Agent model serving | vLLM + `Qwen2.5-14B-Instruct-AWQ` | The agent under attack; drafting training items | Same model as v1, so results compare |
| Translation drafts | **new** IndicTrans2 (AI4Bharat, 200M distilled or 1B) | First drafts of training items in Indian languages | Built for Indian languages; fast; free |
| Romanization + augmentation | **new** `indic-transliteration`; existing `corpus/adaptive.py` | Romanized Hindi, homoglyphs, zero-width characters, code-mixing | Programmatic variety without extra review cost |
| Dedup + leakage checks | **new** `datasketch` (MinHash), scikit-learn TF-IDF | Near-duplicate removal; train/test leakage audit | Same leakage check that caught memorisation in the Kangri project |
| Guard model training | **new** PyTorch, Hugging Face Transformers, `datasets`, `accelerate` | Fine-tuning the classifier | Standard and well documented |
| Guard base models (candidates) | **new** `mDeBERTa-v3-base`, `MuRIL-base`, `IndicBERT v2`, and PromptGuard 2 86M (continued training) | The model being trained | Multilingual; MuRIL and IndicBERT were trained on Indian and romanized text |
| Baselines | ProtectAI `deberta-v3-base-prompt-injection-v2`; PromptGuard 2 (86M and 22M), default and calibrated; v1's transliterate-first variants | What the guard must beat | Already scored in v1, same harness |
| Statistics | existing `analysis/stats.py` (Wilson, exact McNemar); **new** bootstrap CIs for detection at fixed FPR | Every reported number | Paired tests on the same cases |
| Experiment log | JSONL traces + `RESULTS.md` (v1 rule: every number regenerates from a trace) | Reproducibility | Already the house rule |
| Fast CPU inference | **new** ONNX Runtime + Hugging Face Optimum (int8) | Guard latency in the proxy | ONNX export is already a known skill |
| Deployment | existing `service/proxy.py`; **new** Gradio demo on a Hugging Face Space | A firewall people can run; a demo recruiters can click | Free CPU hosting |
| Release | **new** Hugging Face Hub (dataset + model + Space), GitHub release `v2.0` | Public artifacts | Downloads become outside proof |
| Quality gates | pytest, **new** GitHub Actions CI running the offline (`--fake`) tests | Nothing merges red | Cheap insurance |
| Paper | LaTeX on Overleaf (ACL template), arXiv | The write-up | Standard for NLP workshops |
| Build tool | Claude Code | Writing code, tests, notebooks, analysis | See "Working with Claude Code" |

## Repository layout

Everything new goes inside the existing `bhasha-shield` repo. v1 folders stay where they are; v2 adds new folders next to them and changes a few files. Large files (datasets, model weights) go to Hugging Face, never into git.

```text
bhasha-shield/
├── CLAUDE.md                      # NEW: rules Claude Code reads every session
├── README.md                      # updated: v2 section, links to HF dataset, model, Space
├── RESULTS.md                     # keep appending dated entries
├── docs/
│   ├── plan.md, implementation-plan.md   # v1, unchanged
│   └── v2-plan.md                 # NEW: this plan
├── corpus/                        # v1, unchanged except new templates
│   ├── attacks.py                 # + native Hindi / Hinglish templates (human-written)
│   └── guardset/                  # NEW: training data factory
│       ├── schema.py              # GuardItem: text, label, lang, family, carrier, source, provenance
│       ├── carriers.py            # emails, web pages, docs, chats, calendar entries, API JSON
│       ├── generate.py            # English seeds -> drafts via Qwen / IndicTrans2
│       ├── augment.py             # romanize, code-mix, homoglyph, zero-width (reuses adaptive.py)
│       ├── benign.py              # ordinary + adversarial-benign items
│       ├── dedup.py               # MinHash near-duplicate removal
│       ├── split.py               # group split by goal and template family
│       ├── leakage.py             # TF-IDF audit: train vs test
│       └── review/                # review packs + reviewer sign-offs
├── guard/                         # NEW: the model
│   ├── configs/*.yaml             # one file per run: base model, lr, epochs, seed
│   ├── train.py                   # HF Trainer, class weights, fp16, early stopping
│   ├── infer.py                   # chunking (1500 chars, 200 overlap, max score) as in v1
│   ├── export_onnx.py             # ONNX + int8, CPU latency benchmark
│   └── push_hub.py                # upload with model card
├── defenses/classifiers/
│   └── bhasha_guard.py            # NEW: replaces the xlmr_finetuned stub; registered in registry.py
├── scripts/
│   ├── guard_eval.py              # NEW: score every classifier on the frozen test set
│   └── phase5_grid.py             # v1 grid runner, now with --defense bhasha_guard
├── analysis/
│   └── guard_tables.py            # NEW: TPR@FPR tables, CIs, per-language plots
├── notebooks/
│   ├── guard_train_kaggle.ipynb   # NEW: training on a free T4, resumable
│   └── grid_colab.ipynb           # v1, extended with the guard cells
├── service/proxy.py               # updated: guard as the default detector
├── hf/                            # NEW: Hugging Face cards + Space
│   ├── dataset_card.md, model_card.md
│   └── space/app.py               # Gradio demo
├── paper/                         # NEW: LaTeX source (or a link to Overleaf)
└── tests/                         # NEW tests: split, dedup, leakage, guard defense, test-set hash
```

**One rule protects the whole project:** the test set is frozen by a SHA-256 hash in `tests/`. If anything changes it, CI fails.

## Step-by-step build: Phase 0 to the end

There are seven phases, each ending in a **gate**: a check that must pass before the next phase starts. If a gate fails, write the failure in `RESULTS.md` and use the fallback in "Risks and fallbacks". Don't quietly move on. Sizes and training settings below are starting values; adjust them from what you measure.

### Phase 0 — Set up and freeze v1 (week 1, 2–3 days)

1. Tag the current commit `v1.0` and make a GitHub release. Every number on the current resumes then has a permanent source.
2. Create a `v2` branch. Commit this plan (`docs/v2-plan.md`) and write `CLAUDE.md` (see "Working with Claude Code").
3. Add GitHub Actions CI: install `.[dev]` and run the offline test suite on every push.
4. Licence check, written to `docs/licences.md`: PromptGuard 2 (Llama licence), mDeBERTa, MuRIL, IndicBERT v2, IndicTrans2, ProtectAI, AgentDojo, and every source of benign text.
5. Re-run the prior-art search (a v1 standing rule): has anyone released a multilingual or Indic prompt-injection guard since September? Note it in the plan appendix.
6. Line up reviewers. Sameer covers hi, hi_latn and mixed. Ask classmates for one fluent reviewer each for Bengali, Marathi, Tamil or Telugu. **A language is in only if a reviewer says yes.**
7. Pick 2–3 target workshops and write their deadlines at the top of this file.

**Gate G0:** `v1.0` is tagged, CI is green, the licence table is complete, and the language list is fixed.

### Phase 1 — Freeze the test set (weeks 1–2)

The test set is what every claim stands on. It must be human-verified and must never be trained on.

1. Collect the v1 pool per language: 150 verified attack payloads and 300 benign items (167 ordinary + 133 adversarial-benign).
2. **Human-review all non-English benign test items.** In v1 they were machine-drafted, which made the false-positive side a pilot. This step fixes that.
3. Hand-write **native** attacks: 30–50 per language, phrased the way a native speaker would try to persuade, not translated from English. Include Hinglish.
4. Hand-write native adversarial-benign items, about 50 per language: Hindi security notices that quote an attack, harmless instructions, recipes, and how-tos.
5. Use v1's stable 50/50 split by attacker goal (`analysis/calibrate.py`): the **calibration half** is for thresholds, the **test half** is for reported numbers.
6. Hash the frozen files and add a test that fails if the hash changes.

**Gate G1:** every test item has `verified_by` set, the hash is committed, and the hash test passes.

### Phase 2 — Training data factory (weeks 2–4)

The goal is a large, varied, labelled training set that **shares nothing** with the test set.

1. Define `GuardItem` in `corpus/guardset/schema.py`: text, label, language, template family, attacker goal, carrier, source, augmentations, provenance.
2. Build **carriers** (where an injection hides): emails, web pages, documents, WhatsApp/Slack messages, calendar entries, product reviews, API JSON. The payload is inserted at a random position.
3. Write **new** attacker goals (100+, across banking/UPI, travel, email, workspace, shopping, government services) and **new** template families. Nothing may overlap with AgentDojo test goals or test template families.
4. Create positives as template family × goal × carrier × language × augmentation.
5. Draft the languages: English seed → IndicTrans2 draft → Qwen rewrite for natural style. Romanized Hindi and code-mixed text come from an LLM rewrite plus `indic-transliteration`.
6. Create negatives:
    - ordinary text from openly licensed Indic sources, wrapped in the same carriers
    - generated adversarial-benign text that sounds like an instruction but is harmless
7. Augment with homoglyphs, zero-width characters and script mixing (reusing `corpus/adaptive.py`).
8. Clean up:
    - MinHash dedup
    - group split into train/validation by goal and template family
    - **TF-IDF leakage audit** against the test set: drop anything too close and record how many were dropped
9. Quality check: reviewers check a fixed-seed 5% sample per language. If more than 10% are wrong, fix the generator and regenerate.
10. Start at about 2,000 items per language. Grow only while the learning curve (Phase 3) is still rising.

**Gate G2:** the leakage audit leaves zero test items above the threshold, the sample error rate is 10% or less, and a draft dataset card is written. Push to Hugging Face as **private** for now.

### Phase 3 — Baselines and guard training (weeks 4–6)

1. **Reproduce v1 first.** Score ProtectAI (raw and transliterate-first) and PromptGuard 2 (86M and 22M, default and calibrated) with `scripts/guard_eval.py` on the v1 items. They must match `RESULTS.md`. If they don't, fix that before anything else.
2. Score those baselines on the new frozen test set.
3. Train the candidates: mDeBERTa-v3-base, MuRIL, IndicBERT v2, and PromptGuard 2 86M with continued training. Use **3 seeds each**. Starting settings: learning rate 2e-5, 3 epochs, max length 512, batch 16, fp16, class-weighted loss, early stopping on validation PR-AUC.
4. Run the ablations a reviewer will ask for:
    - English-only training vs multilingual training (this proves the dataset matters)
    - with and without adversarial-benign items
    - with and without augmentation
    - a learning curve at 25/50/100% of the data
5. Fit thresholds per language on the calibration half. Pick the winning model on the **calibration half only**, then report it on the test half once.

**Gate G3:** the guard's detection at 2% FPR beats calibrated PromptGuard 2 in every non-English language with non-overlapping 95% CIs. If not, record the negative result and take the fallback.

### Phase 4 — Agent-level grid (weeks 6–8)

1. Add `defenses/classifiers/bhasha_guard.py`, register it, and test it with the fake model.
2. Attacks per language:
    - English: `important_instructions`
    - Hindi: the English-template-around-Hindi-goal attack (`enframe`) plus the native Hindi templates from Phase 1
    - hi_latn and mixed: native templates
3. Run on a Colab T4: undefended, `bhasha_guard`, calibrated PromptGuard 2, sandwich, and guard + sandwich. Defended cells run on the scorable set only (v1 rule D3).
4. Report uASR, efficacy and utility with paired exact McNemar tests.

**Gate G4:** in at least English and Hindi, the guard cuts uASR significantly (p < 0.05) with a utility drop of 5 points or less.

### Phase 5 — Robustness and deployment (weeks 8–9)

1. Run the adaptive attacks against the guard and report how much detection drops. Call it a bounded adaptive attacker, not "robust", as v1 did.
2. Export to ONNX with int8 quantization. Measure median and p95 CPU latency per check on a laptop.
3. Make the guard the default detector in `service/proxy.py`, with per-language thresholds.
4. Build the Gradio Space: paste a tool output and see the score, detected language and allow/block decision.

**Gate G5:** the Space is public and the latency table is in `RESULTS.md`.

### Phase 6 — Release and paper (weeks 9–11)

1. Make the dataset and model public on Hugging Face with full cards: intended use, languages, licence, evaluation, limitations.
2. Publish GitHub release `v2.0`. The README gets one command per paper table.
3. Write the paper:
    - **Introduction:** the gap
    - **Method:** uASR and the capability probe (v1)
    - **Benchmark v2 and the training set**
    - **The guard**
    - **Results:** v1 diagnosis → classifier results → agent-level results → robustness
    - **Limitations:** synthetic training data, one strong agent model, the language list
    - **Ethics:** attack templates are dual-use; explain what you release and why
4. Post the preprint on arXiv, submit to the workshop, then update the resume and post on LinkedIn.

**Gate G6:** the preprint is public, the submission is confirmed, and v2.0 is tagged.

## Timeline

Eleven weeks from setup to preprint; the guard result (G3) is the turning point. Phases overlap; each phase ends at its gate.

| Phase | Weeks | Dates | Gate |
| --- | --- | --- | --- |
| P0 Set up, freeze v1 | 1 | 10 Nov – 12 Nov 2026 | G0 |
| P1 Freeze test set | 1–2 | 10 Nov – 23 Nov 2026 | G1 |
| P2 Training data | 2–4 | 17 Nov – 7 Dec 2026 | G2 |
| **P3 Train the guard** | 4–6 | 1 Dec – 21 Dec 2026 | **G3** |
| P4 Agent-level grid | 6–8 | 15 Dec 2026 – 4 Jan 2027 | G4 |
| P5 Robustness, demo | 8–9 | 29 Dec 2026 – 11 Jan 2027 | G5 |
| P6 Release, paper | 9–11 | 5 Jan – 25 Jan 2027 | G6 |

The schedule assumes about 15–20 hours a week alongside classes. The test set (P1) is frozen before any training data exists, and G3 decides whether the paper leads with a positive or a negative guard result. If exams or deadlines take a week, move every later phase right; don't overlap phases further. Using the cut order under "Risks and fallbacks" keeps the preprint date.

## Evaluation and statistics rules

These rules are what make the results defensible in an interview or a paper review. They are also in `CLAUDE.md` so every session follows them.

**Classifier level (guard vs baselines):**

| Metric | Definition | Reported as |
| --- | --- | --- |
| TPR at 1% and 2% FPR | Share of attacks caught when the threshold allows 1% or 2% false positives on benign text | **Headline number.** Per language, with bootstrap 95% CI |
| FPR, ordinary vs adversarial-benign | False positives split by benign type | Per language; this is what decides whether it can ship |
| AUC | Ranking quality, independent of the threshold | Per language, as supporting evidence |
| Shipped vs calibrated threshold | One global threshold vs one per language | Shows the v1 calibration finding still holds |
| Latency | Median and p95 ms per check on a laptop CPU (ONNX int8) | One table |

**Agent level (inside the harness):** uASR, efficacy E(L,D) and utility, exactly as v1 defines them. Defended cells run on the scorable set only. Paired exact McNemar on the same cases. **Undefined stays undefined:** if nothing is scorable, the result is `None`, never 0.

**Rules that are never broken:**

1. The test half is used **once**, for the final table. All model selection, thresholds and tuning use the calibration half or the validation split.
2. Every model and baseline uses the same split, the same chunking (1500 characters, 200 overlap, max score) and the same per-language calibration.
3. 3 seeds for every trained model. Report the mean and the spread, not the best run.
4. Report results per language. An average across languages can hide a failure in one of them.
5. Every number in the paper regenerates from a trace on disk with one command (v1's standing rule).
6. Bad results go into `RESULTS.md` too, with the date.

## Compute and budget

The target is the same as v1: **Rs 0**. Every step has a free place to run.

| Work | Where it runs | Notes |
| --- | --- | --- |
| Code, tests, data cleaning, dedup, leakage audit, analysis | Laptop CPU | Claude Code works here |
| Baseline and guard scoring on the test set | Laptop CPU | v1 already did this for ProtectAI and PromptGuard 2 |
| Translation drafts (IndicTrans2) and LLM rewrites (Qwen 14B AWQ via vLLM) | Kaggle or Colab T4 | Batch it; cache every response, as v1 does |
| Guard training (4 base models × 3 seeds + ablations) | Kaggle T4 (about 30 GPU hours a week on the free tier; check the current quota) | The biggest GPU job. Run the cheapest candidate first and drop losers early |
| Agent-level grid | Colab T4 with the existing resumable notebook | Defended cells on the scorable set only, which v1 found about 4× cheaper |
| ONNX export + latency | Laptop CPU | |
| Hosting | Hugging Face (dataset, model, CPU Space), GitHub, arXiv | All free |

**Two ways to keep GPU time down:**

- Use `guard_eval.py` on the CPU to rule out weak candidates before training full seeds.
- Use the response cache everywhere, so a crashed session never pays twice.

If a free quota runs out mid-phase, the fallback is waiting a week for the reset. Plan the GPU-heavy weeks (3–5 and 6–8) with that in mind.

## Risks and fallbacks

| Risk | Early warning | Fallback |
| --- | --- | --- |
| No reviewer for extra languages | Nobody agrees by the end of Phase 0 | Ship with en, hi, hi_latn, mixed, which Sameer can verify. Bengali stays a labelled pilot. Four verified languages beat eight unverified ones. |
| The guard learns generator quirks instead of attacks (synthetic overfitting) | High validation score but a weak score on the human-written test set | More variety in carriers and families; add natural benign text; train on the calibration half of the native items with a nested split; report the gap honestly |
| The guard doesn't beat calibrated PromptGuard 2 (G3 fails) | The calibration-half comparison in Phase 3 | Switch the main model to PromptGuard 2 with continued training on our data, or an ensemble. The dataset and benchmark are still a publishable contribution. |
| Hindi attacks too rare to test defenses (as in v1) | Fewer than 10 undefended successes in a language | Use the `enframe` attack and the native templates. If still too rare, report that language at classifier level only and say why. |
| Licence blocks a base model or data source | The Phase 0 licence check | Train from mDeBERTa (MIT) on self-generated data; release under CC BY 4.0 like the v1 corpus |
| Someone publishes a similar Indic guard | The prior-art search at the start of each phase | Position on what v1 has that others don't: uASR, per-language calibration, agent-level testing. Compare against them directly. |
| Free GPU quota or sessions run out | Kaggle quota under 10 hours mid-week | The response cache and resumable notebooks; do CPU work while waiting; drop the weakest base-model candidate |
| Exams, internship deadlines, and running out of time | Two phases slip by more than a week | Cut in this order: extra languages → Phase 5 Space polish → ablation (c) → the guard + sandwich combination. Never cut the test-set freeze or the paper. |
| Releasing attack text is dual-use | — | Agent attack templates like these are already public in AgentDojo. Release with a clear intended-use statement and an ethics section. Don't release any goal that targets a real service. |

## Working with Claude Code

Claude Code writes the code, tests, notebooks and analysis. **Sameer** runs the GPU notebooks, reviews the Indian-language text, and decides at every gate. Claude can't judge whether Hindi sounds natural or a label is right; that's the job of Sameer and the reviewers.

### Step 1: put this `CLAUDE.md` in the repo root

```markdown
# bhasha-shield: rules for Claude Code

## Project
- Python package `agentfw`. v1 is frozen at tag `v1.0`; v2 work happens on branch `v2`.
- The plan is docs/v2-plan.md. Read the current phase before starting a task.
- Running log of every number: RESULTS.md.

## Never
- Never edit the frozen test set; tests/test_testset_hash.py guards it.
- Never train, tune or pick a model on the test half. Use the calibration half or validation split.
- Never write a number into RESULTS.md, README or the paper unless it regenerates from a trace on disk.
- Never invent results, citations or dataset sizes. If unknown, say so.
- Never commit datasets, model weights, caches, .env or tokens.
- Never call a paid hosted API. Models are open-weight on free GPUs.

## Always
- Write or update tests first. Run `python -m pytest` before every commit.
- Keep the `--fake` (no-GPU) paths working; CI runs only those.
- Fix seeds and log model IDs and versions in every trace.
- GPU work goes into resumable, cached notebook cells. The user runs them and brings back the cache archive.
- Every new number gets a dated RESULTS.md entry, including bad ones.
- Small commits, one task each; the message says what changed and why.
- Report results per language. Undefined stays None, never 0.
```

### Step 2: the working loop for every task

1. **One session per task**, not per phase. Use `/clear` between tasks so context stays focused.
2. **Start in plan mode.** Ask for a design before code, read it, push back, then approve.
3. **Tests first**, then the code, then `pytest`, then a commit.
4. **GPU loop** (the same as v1): Claude writes the notebook cell → Sameer runs it on Kaggle or Colab → downloads the cache and results archive → Claude merges it, regenerates tables, and writes the RESULTS.md entry.
5. **Before each gate**, ask Claude to check the gate's criteria against files on disk, and show the evidence.
6. Run `/code-review` on the branch at the end of each phase.

### Step 3: starter prompts for each phase

| Phase | Prompt to start with |
| --- | --- |
| 0 | "Read README.md, RESULTS.md and docs/v2-plan.md. Tag v1.0, create branch v2, add a GitHub Actions workflow that runs the offline tests, and draft docs/licences.md with one row per model and data source. Leave the licence column for me to fill in." |
| 1 | "Plan mode. Read corpus/schema.py, corpus/attacks.py and analysis/calibrate.py. Design how the frozen v2 test set is assembled from the verified v1 items plus native items I'll write, split 50/50 by attacker goal as calibrate.py does, and locked by a SHA-256 hash test." |
| 2 | "Plan mode. Design corpus/guardset/ as in docs/v2-plan.md Phase 2: schema, carriers, generation, augmentation, dedup, group split, leakage audit. Nothing may share a goal or template family with the test set; write the tests that prove it first." |
| 2 (GPU) | "Write a resumable Kaggle notebook cell that drafts training items with IndicTrans2 and rewrites them with Qwen2.5-14B-AWQ via vLLM, using harness/cache.py, so a crashed session resumes for free." |
| 3 | "Write scripts/guard_eval.py and first reproduce the v1 ProtectAI and PromptGuard 2 numbers from RESULTS.md exactly. Show me a diff table before scoring anything new." |
| 3 (GPU) | "Write guard/train.py and notebooks/guard_train_kaggle.ipynb: config-driven, 3 seeds, fp16, early stopping on validation PR-AUC, checkpoints to the Kaggle working directory, metrics to JSONL." |
| 4 | "Add defenses/classifiers/bhasha_guard.py replacing the xlmr_finetuned stub, register it, test it with the fake model, then add the grid cells to grid_colab.ipynb for the defenses and attacks in Phase 4." |
| 5 | "Export the winning guard to ONNX with int8 via Optimum, benchmark median and p95 CPU latency, make it the default in service/proxy.py, and build hf/space/app.py as a Gradio demo." |
| 6 | "Using only numbers in RESULTS.md and analysis/out, draft the Hugging Face dataset and model cards, and generate every paper table with one command each. Flag any number you can't trace." |

## Release, paper and resume

**What goes where:**

| Artifact | Where | When |
| --- | --- | --- |
| Code, configs, tests, notebooks | GitHub `bhasha-shield`, release `v2.0` | Gate G6 |
| Training set + frozen test set, with dataset card | Hugging Face Datasets (private at G2, public at G6) | G2 → G6 |
| Guard model (PyTorch + ONNX int8), with model card | Hugging Face Hub | After G3 |
| Live demo | Hugging Face Space (Gradio, free CPU) | G5 |
| Paper | arXiv preprint + a workshop submission | G6 |

**Where to submit:** workshops on multilingual or Indic NLP, trustworthy and safe NLP, or LLM agent security. These usually sit alongside ACL, EMNLP, NAACL, NeurIPS or ICLR. Check the current calls in Phase 0 and pick the nearest one whose deadline is after week 11.

**Resume bullets to aim for.** Fill in the X's only with real numbers from `RESULTS.md`. Each bullet goes on the resume as soon as its gate passes, even before the paper:

- After G2: "Built a human-verified multilingual prompt-injection dataset for Indian languages (X items, Y languages, released on Hugging Face)"
- After G3: "Trained a multilingual injection guard that catches **X%** of Hindi attacks at 2% FPR vs **51%** for calibrated PromptGuard 2 (95% CI, 3 seeds)"
- After G4: "As an agent defense, the guard cut Hindi attack success from X% to Y% (McNemar p = Z) with under 5 points of task-completion loss"
- After G5: "Shipped it as an ONNX firewall at X ms per check on CPU, with a live demo on Hugging Face Spaces"
- After G6: "First-author paper submitted to [workshop]; preprint on arXiv"

When these are in, the bhasha-shield entry gets the strongest 3–4 of them and drops the weakest v1 bullets, so the resume stays one page.

**Tell people about it:** a LinkedIn post at G3 (the result) and at G6 (the release). Tag the PromptGuard and ProtectAI teams politely with the comparison. Cold-email Indian AI labs and startups with the Space link.
