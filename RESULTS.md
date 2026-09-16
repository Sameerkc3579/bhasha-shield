# RESULTS

Append-only. Every week gets an entry, **including weeks where the number got
worse or nothing worked**. This is the file to read from in an interview.

Rule: no number appears here that cannot be regenerated from a trace in
`traces/`.

---

## 2026-09-16 — Phase 0: Foundations

**Done.**

- Repo scaffolded. Core types frozen: `corpus/schema.py` (5 language
  conditions, `InjectionCase`, `BenignItem`, `ProvenanceRecord`) and
  `defenses/base.py` (`Defense` protocol, `Verdict`, `RunContext`).
- `harness/scoring.py` implements C(L), ASR, uASR, efficacy, FPR, benign
  utility, with Wilson intervals for proportions and a seeded percentile
  bootstrap for the efficacy ratio.
- `harness/capability.py` implements the C1 probe and its join onto grid
  results, plus a probe-disagreement check.
- `harness/budget.py` — persistent per-phase spend cap. `harness/cache.py` —
  content-addressed response cache. `harness/trace.py` — append-only JSONL with
  git SHA recorded per session.
- **44 tests green** (`pytest`, 0.65s). Scoring was written and tested before
  any experiment code, per the standing rule.

- `corpus/extract.py` turns the vendored suites into `InjectionCase` records and
  draws the stratified subset. **Phase 1 step 1 is therefore already done.**

**Numbers.**

| Quantity | Value | Source |
|---|---|---|
| Tests passing | 59 / 59 | `pytest` |
| Security cases available (AgentDojo v1.2.1) | **949** | `python -m corpus.extract` |
| Stratified subset drawn | 150 over 37 strata | `corpus/injections/en.jsonl` |
| Grid cost, primary (`claude-haiku-4-5`, 6,750 runs) | $38.75 | `python -m harness.budget` |
| Grid cost, spot-check (`claude-opus-5`, 20% slice) | $38.74 | `python -m harness.budget` |
| Total projected API spend | ~$77.50 | sum of above |

Projected spend lands inside the $60–110 band estimated in `plan.md` §6, at an
assumed 80% prompt-cache hit rate. That assumption is the single biggest cost
lever and gets verified in Phase 1, not assumed.

**Correction to the plan.** `plan.md` §0 cites AgentDojo at 629 security cases,
which is the figure from the original paper. The pinned v1.2.1 suites actually
yield **949** (4 domains, 97 user tasks × 35 injection tasks, version-filtered).
The 150-case subset and the cost model are unaffected; the sampling frame is
larger than planned, which is good for stratification.

**Case distribution** (attacker goal type, full 949):

| Goal type | n |
|---|---|
| exfiltration-via-message | 523 |
| unauthorised-transaction | 208 |
| state-tampering | 76 |
| destructive-action | 61 |
| information-disclosure | 41 |
| content-manipulation | 20 |
| other | 20 |

Goal type is derived from the attacker's ground-truth function calls where they
exist. About a third of AgentDojo's injection tasks have no ground-truth calls —
their success is judged on model output — so those fall back to English prose
keywords. The label is fixed on the English case and **inherited** by
translations rather than recomputed, or the stratification would drift between
language columns.

**Vendored.** AgentDojo pinned at `089ed468cf3ed0322acc66b0211f26d9d90dbf60`
(`github.com/ethz-spylab/agentdojo`). Its `BaseInjectionTask.GOAL` field is the
attacker objective already written as a plain, LLM-understandable instruction —
which is exactly the input the capability probe needs, so C1 requires no
hand-authoring of attacker goals for the upstream cases.

**Decisions.**

- Virtualenv lives at `~/.venvs/agentfw`, outside OneDrive. A venv inside a
  synced folder means thousands of files churning through sync and occasional
  file locks mid-run.
- Primary model is a cheap fast model with one frontier model on a stratified
  slice, per `plan.md` §6 — the cost table above is why.
- Unprobed cases default to *achievable*. A missing probe therefore inflates the
  apparent attack surface rather than quietly shrinking it into a
  better-looking defense number, which would recreate the bias being corrected.

**Not done / next.**

- Phase 0's live end-to-end run of one AgentDojo case is blocked on
  `ANTHROPIC_API_KEY`. Everything else in Phase 0 is verified offline.
- Phase 1: stratified 150-case subset, English undefended baseline, benign
  corpus v1, and a cache-hit-rate check on the first real runs.

---

## 2026-09-16 — Phase 0: the agent loop

**Done.**

- `harness/runner.py` — the agent loop, with the `Defense` hook sitting exactly
  where it has to: between tool execution and the tool result reaching the
  model. Plus `AgentDojoRunner` (which also satisfies the Phase 2 capability
  probe's `AgentRunner` protocol, so probe and grid share one execution path)
  and `AnthropicClient` (cached, budgeted).
- `scripts/phase0_smoke.py` — the exit-criterion script, with a `--fake` mode
  that replays the attacker's own ground-truth calls through the real loop, real
  environment and real scorers. No key, no spend.
- **77 tests green**, up from 59.

**Decision: we do not use AgentDojo's agent pipeline.** Its pinned model
registry stops at Claude 3.7 Sonnet and has no entry for the models this project
runs on. Patching their enum would put us back here at the next model change.
We take their environments, tools (`FunctionsRuntime.run_function`) and their
`utility()` / `security()` scorers, and own the loop. Needed anyway: a defense
must inspect tool output before the model sees it, and that boundary is inside
the loop.

**Verified end to end, offline.** `--fake` run on
`banking__user_task_0__injection_task_7`: the loop executed `update_password`,
and AgentDojo's own `security()` scorer confirmed the attack succeeded, while
`utility()` correctly reported the user's task was not done. Trace written, 8
events. So the loop, runtime, environment mutation and scoring wiring are all
connected.

**Two bugs found and fixed.**

1. *Sampling parameters do not exist any more.* The installed SDK (`anthropic`
   1.6.0) has removed `temperature` from `messages.create()` outright — passing
   it raises `TypeError` before a request is made. The obvious
   "set temperature=0 for determinism" instinct is now a crash.
   **Consequence: runs cannot be made bit-deterministic.** Two things carry that
   weight instead — the response cache (makes re-analysis exactly reproducible)
   and the 5% re-run variance measurement in Phase 5, which stops being
   optional. Report the variance in the paper; most work in this area does not.
2. *The smoke script printed success over a failed run.* It reported
   "PHASE 0 EXIT: ..." and returned 0 even when the run errored. Caught on the
   first live attempt. Now exits 1 and says the criterion was not met. This is
   the class of bug that lets a broken grid look fine for a week.

**Numbers.**

| Quantity | Value |
|---|---|
| Tests passing | 77 / 77 |
| Offline smoke run (`--fake`) | exit 0, attack detected by `security()`, 8 trace events |
| Live smoke run | exit 1 — no credentials |
| Spend to date | **$0.00** |

**Phase 0 exit criterion: not yet met.** Everything except the live model call
is built and verified. The remaining step needs credentials:

```
ant auth login          # or: cp .env.example .env, fill in ANTHROPIC_API_KEY
python -m scripts.phase0_smoke
```

Expected cost of that run: under a cent.

---

<!-- Next entry goes here. Date it. Include the bad weeks. -->
