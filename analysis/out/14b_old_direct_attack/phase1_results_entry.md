
## 2026-10-02 — Phase 1: undefended English baseline

Ran on a free Colab T4, `Qwen/Qwen2.5-14B-Instruct-AWQ`, repo at `n/a`, AgentDojo
`089ed46`. Spend: **Rs 0 / $0.00**.

**Deviation from `docs/KAGGLE.md`:** that runbook serves fp16 on Kaggle's 2xT4. A single
free Colab T4 has 16 GB, which cannot hold these weights in fp16, so this ran the
**AWQ 4-bit** build instead (`max_model_len` 12288, extra vLLM args ['--enforce-eager']). Quantization can move a capability ceiling, and `C(L)` is the headline
number, so Phase 2 must run against this same served artifact — not the fp16 one.

| Quantity | Value |
|---|---|
| Cases | 150 |
| Errored | 2 (1.3%) |
| `ASR(en, undefended)` | 0.033783783783783786 (5/150) |
| `uASR(en, undefended)` | 0.033783783783783786 |
| `Utility(en, undefended)`, clean | 69/150 |
| Utility under attack | 57/150 |
| Mean latency | 258169.91440256755 ms |
| Tokens | 3,507,882 |

`uASR` equals `ASR` here by construction: no capability probe has run yet, so every case
defaults to achievable. That default is deliberate — a missing probe inflates the
apparent attack surface rather than quietly shrinking it into a better-looking defense
number, which is the bias this project exists to correct. Phase 2 replaces it with a
measurement.

**Phase 0 exit criterion met** — first live model call, one AgentDojo case, traced.

**Next:** Phase 2, the capability probe, and the go/no-go it decides. It also measures
the McNemar discordance that the Phase 1 power analysis left open, which is what sizes
n=150 vs 250 vs 300.
