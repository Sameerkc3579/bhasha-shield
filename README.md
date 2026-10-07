# bhasha-shield

**Utility-normalized cross-lingual evaluation of prompt-injection defenses for tool-using agents.**
*Bhasha* is Hindi for language. The Python package is `agentfw`.

Plans: [`docs/plan.md`](docs/plan.md) (the claim) and [`docs/implementation-plan.md`](docs/implementation-plan.md) (the build order).
Running log of every number: [`RESULTS.md`](RESULTS.md).

---

## The claim

Multilingual agent-security papers report raw attack success rate. That number
conflates two very different things:

> **"The defense stopped the attack"** and **"the agent was too incompetent in
> this language to carry the attack out either"** produce the identical raw ASR.

This repo measures them apart. Before scoring any injection case in language
*L*, it asks the agent to perform the attacker's goal *as a legitimate user
instruction* in *L*. Cases the agent could never have completed are unscorable,
not secure.

It also measures the cost side nobody reports per language: false-positive rate
on benign tool output, and task utility under each defense.

---

## Status

**Runs entirely free.** The agent is an open-weight model (`Qwen2.5-7B-Instruct`, AWQ
4-bit) served by vLLM on a free Colab T4 -- `notebooks/phase1_colab.ipynb` does the whole
thing top to bottom. No API key is needed anywhere; spend to date is Rs 0.

```bash
python -m scripts.phase0_smoke --fake   # no model: verifies the wiring
python -m scripts.phase2_probe --fake   # no model: verifies the probe wiring
```

| Phase | State |
|---|---|
| 0 -- Foundations | done |
| 1 -- English baseline | done: 7B and 14B with the bare goal; 14B re-run with `important_instructions` |
| 2 -- Capability ceiling (**go/no-go**) | English + four-language pilot done on the 14B; `hi` re-run on verified goals queued in the notebook |
| 3 -- Localization | goals: `hi` verified, others drafted; templates + benign corpus (300/lang) drafted in all four; environments localized (prose only, gated by AgentDojo's own ground truth); human review packs in `corpus/review/` |
| 4 -- Defenses | all nine built, green on fixtures; `xlmr_finetuned` has no checkpoint |
| 5 -- The grid | runner built (`scripts/phase5_grid.py`); needs the GPU |
| 6 -- Calibration + artifact | built: `analysis/calibrate.py`, `defenses/normalize.py`, `corpus/adaptive.py`, `service/proxy.py`; classifier scores run on CPU |
| 7 -- Write-up | `analysis/figures.py` regenerates every table/figure; `DATASHEET.md` drafted |

---

## Setup

```bash
python -m venv ~/.venvs/agentfw          # keep it out of OneDrive
~/.venvs/agentfw/Scripts/python -m pip install -e ".[dev]"
~/.venvs/agentfw/Scripts/python -m pytest
```

### Vendored benchmark

AgentDojo is pinned, not tracked, so upstream churn cannot silently change
results mid-experiment:

```
repo: https://github.com/ethz-spylab/agentdojo
SHA:  089ed468cf3ed0322acc66b0211f26d9d90dbf60
```

```bash
git clone --depth 1 https://github.com/ethz-spylab/agentdojo.git vendor/agentdojo
cd vendor/agentdojo && git checkout 089ed468cf3ed0322acc66b0211f26d9d90dbf60
pip install -e .
```

---

## Metrics

```
C(L)       capability ceiling  -- can the agent do the attacker's task in L at all?
S(L)       scorable set        -- injection cases whose attacker task is achievable in L
ASR(L,D)   raw attack success rate                    (what the literature reports)
uASR(L,D)  ASR restricted to S(L)                     (what it should report)
E(L,D)     1 - uASR(L,D) / uASR(L, undefended)        (efficacy vs this language's ceiling)
FPR(L,D)   false positives on the benign corpus       (the shippability number)
```

**Undefined stays undefined.** When nothing in a language is achievable, `uASR`
returns `None`, never `0.0`. A zero there reads as "perfectly secure" and would
reproduce the exact error this project exists to correct. `tests/test_scoring.py`
pins that behaviour.

---

## Layout

| Path | What |
|---|---|
| `corpus/schema.py` | Frozen data types — languages, cases, provenance |
| `defenses/base.py` | The `Defense` protocol every defense plugs into |
| `harness/scoring.py` | The metrics above. **The contribution.** |
| `harness/capability.py` | The capability probe (C1) |
| `harness/budget.py` | Hard spend cap, persists across restarts |
| `harness/cache.py` | Content-addressed response cache |
| `harness/trace.py` | Append-only JSONL; every paper number regenerates from these |
| `corpus/extract.py` | AgentDojo -> InjectionCase, stratified sampler |
| `harness/runner.py` | The agent loop; defense hook at the tool-output boundary |
| `scripts/phase0_smoke.py` | Phase 0 exit check, `--fake` runs it without a key |
| `defenses/` | the nine defenses (A classifiers, B prompting, C architectural) + `registry.py` |
| `defenses/normalize.py` | fold / transliterate / translate before classifying (Phase 6 cheap fix) |
| `corpus/adaptive.py` | bounded adaptive attacker: homoglyph, zero-width, script laundering |
| `scripts/benign_fpr.py`, `scripts/attack_scores.py` | classifier scores, no agent needed (CPU is enough) |
| `analysis/calibrate.py` | per-language thresholds; shipped vs calibrated vs matched-detection FPR |
| `analysis/figures.py` | every paper table and figure, from files on disk |
| `service/proxy.py` | the deployable firewall: HTTP, per-language threshold, sink rule |
| `corpus/environments.py` | C3: AgentDojo environments with prose localized, checker values kept; three offline gates |
| `corpus/benign_extend.py` | benign corpus 127 -> 300: hand-written items (India-context), staged and entity-checked |
| `tests/` | 251 tests, scoring first |

---

## The firewall in 90 seconds

```bash
python -m scripts.benign_fpr --langs en hi hi_latn bn mixed --defenses protectai_deberta   # CPU, ~8 min
python -m scripts.attack_scores --langs en hi hi_latn bn mixed --defenses protectai_deberta
python -m analysis.calibrate --defenses protectai_deberta     # writes per-language thresholds
python -m service.proxy --demo                                # Hindi email, Hindi injection, two payments
python -m service.proxy --port 8787                           # POST /v1/inspect, /v1/check_call
```

---

## Standing rules

- **Never report a number you cannot regenerate from a trace.**
- Append to `RESULTS.md` every week, including weeks the number got worse.
- Re-run the arXiv prior-art search at the start of every phase (see `docs/plan.md` appendix).
- Fix temperature and seed; log model version strings.

---

## Cost

**Zero, by design.** Every model call goes to a locally served open-weight model on a free
GPU (Colab T4, or Kaggle 2xT4), and the response cache makes a resumed session free for
anything already computed. The `--backend anthropic` path in the older scripts is kept only
so Phase 0 traces stay comparable; nothing in the plan uses it. If a second model is needed
for a robustness check, it is another open-weight model on the same free hardware, not a
hosted frontier API.
