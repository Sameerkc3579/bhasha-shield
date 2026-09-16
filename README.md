# agentfw

**Utility-normalized cross-lingual evaluation of prompt-injection defenses for tool-using agents.**

Plans: [`../plan.md`](../plan.md) (the claim) and [`../implementation-plan.md`](../implementation-plan.md) (the build order).
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

**Phase 0 built and verified offline — 77 tests green.** Agent loop, scoring,
budget, cache, traces; 949 AgentDojo security cases enumerated, 150-case
stratified subset drawn. The one remaining exit step is a live model call:

```bash
ant auth login                      # or: cp .env.example .env
python -m scripts.phase0_smoke      # ~$0.005
python -m scripts.phase0_smoke --fake   # no key, no spend: verifies the wiring
```

| Phase | State |
|---|---|
| 0 — Foundations | built + verified offline; live run needs credentials |
| 1 — English baseline | case extraction + stratified subset done; baseline run blocked on `ANTHROPIC_API_KEY` |
| 2 — Capability ceiling (**go/no-go**) | not started |
| 3 — Localization | not started |
| 4 — Defenses | not started |
| 5 — The grid | not started |
| 6 — Calibration + artifact | not started |
| 7 — Write-up | not started |

---

## Setup

```bash
python -m venv ~/.venvs/agentfw          # keep it out of OneDrive
~/.venvs/agentfw/Scripts/python -m pip install -e ".[dev]"
cp .env.example .env                      # then fill in ANTHROPIC_API_KEY
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
| `tests/` | 77 tests, scoring first |

---

## Standing rules

- **Never report a number you cannot regenerate from a trace.**
- Append to `RESULTS.md` every week, including weeks the number got worse.
- Re-run the arXiv prior-art search at the start of every phase (see `../plan.md` appendix).
- Fix temperature and seed; log model version strings.

---

## Cost

Measured estimate from `python -m harness.budget`, at 150 cases × 5 languages ×
9 defense configs:

| | Runs | Cost |
|---|---|---|
| Primary (`claude-haiku-4-5`), full grid | 6,750 | ~$39 |
| Spot-check (`claude-opus-5`), 20% slice | 1,350 | ~$39 |

Assumes an 80% prompt-cache hit rate. **If a dry run shows
`cache_read_input_tokens` near zero, stop and fix it before running the grid** —
it is roughly a 3× cost difference, and the system prompt plus tool definitions
are identical across every run.
