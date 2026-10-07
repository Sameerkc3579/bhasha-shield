# R4 — Project Plan

**Working title:** *Does Your Agent Firewall Speak Hindi? A utility-normalized cross-lingual audit of prompt-injection defenses for tool-using agents.*

**Created:** 2026-09-16
**Parent doc:** `LLM-Projects-Resume-and-Research.md` → R4 (resume) / S10 (research) bridge entry
**Status:** planned, not started

---

## 0. The honest verdict on R4 as written

R4 in the parent file says: *build injection-detection middleware, build an adversarial corpus, compare architectural vs classifier defenses.*

**All three are now saturated.** Prior-art checking (2026-09-16) finds:

| R4 component | Already done by |
|---|---|
| Firewall / sanitizing middleware | CommandSans, AgentWatcher, LlamaFirewall, Sentra-Guard |
| Adversarial corpus / benchmark | BIPIA, InjecAgent, AgentDojo, ASB, WASP, LivePI, PIArena |
| Architectural defenses (IFC, capabilities, reference monitors) | CaMeL, FIDES, Progent, RTBAS, FORGE, IPIGuard |
| Architectural *vs* classifier comparison | AgentShield (does this explicitly) |
| "Defenses break under adaptive attack" | Zhan et al. (NAACL'25), AutoDojo, PISmith |
| Multilingual injection detection | MIPIAD, IndicGuard, MrGuard, X-Guard, SEALGuard, TWGuard |
| Hindi/Hinglish injection dataset | Sharma et al. (PMC13201763) — 4,000 prompts |
| Multilingual agent perf + security benchmark | MAPS — 12 languages incl. Hindi, 805 tasks |

The obvious "novel" pivot — *multilingual injection defense* — is also taken. **AgentShield (arXiv 2605.11026)** already evaluated AgentDojo across English / Kurdish / Arabic / code-switched and already reported the headline that classifier detectors (ProtectAI DeBERTa, Meta Prompt-Guard-2) collapse off-English while behavior-based defense stays language-agnostic.

Building R4 as specified produces a competent project with no claim attached to it. The plan below keeps the *artifact* and replaces the *question*.

---

## 1. The gap that is actually open

Every paper in the table above reports **attack success rate**. Almost none report the cost side, and none report it *as a function of language*:

1. **Nobody publishes per-language false-positive rate on benign tool output.** AgentShield's FP testing was 485 normal-use cases on a single model. MIPIAD is text classification with no per-language FPR. IndicGuard is content moderation, not agentic injection. A detector that flags "nearly every Kurdish or Arabic input as malicious" has an ASR number that looks great and an FPR that makes it unshippable — and the FPR is not in the table.

2. **Nobody controls for the agent's own competence in the language.** This is the serious one, and it invalidates a chunk of existing numbers:

   > When an agent is measured at low ASR in Hindi, we cannot distinguish *"the defense stopped the attack"* from *"the agent was too incompetent in Hindi to execute the attacker's task either."*

   MAPS independently shows agent task performance degrades substantially in non-English. That degradation applies to the attacker's injected task exactly as it applies to the user's task. **Raw ASR in a low-resource language is deflated by agent incompetence, and every multilingual agent-security result published so far inherits this confound.**

3. **Nobody reports the security–utility frontier per language.** Thresholds are tuned on English and shipped globally. Whether an English-calibrated operating point is even near-optimal for Devanagari or romanized Hinglish input is unmeasured.

**The thesis:** *Multilingual agent-security results are measured wrong. Once you normalize for what the agent can actually do in each language and report the false-positive cost, the ranking of defenses changes, and "language-agnostic" defenses turn out to be language-agnostic in ASR but not in usability.*

A null result — "normalization changes nothing, thresholds transfer fine" — is still publishable, because it retroactively validates a literature that currently has no right to assume it. That is the property the parent file demands (§ *Before committing*, rule 2).

---

## 2. Contributions

**C1 — The capability-ceiling protocol (methodological).**
Before scoring any injection case in language *L*, measure whether the agent can perform the *attacker's* task in *L* when asked directly, as a legitimate user instruction. That is the capability ceiling `C(L)`. Cases above the ceiling are **unscorable, not secure**.

**C2 — Utility-normalized ASR (metric).**

```
Capability ceiling  C(L)    = success rate of attacker-task-as-direct-user-instruction in L
Scorable set        S(L)    = injection cases whose attacker task is achievable in L
uASR(L,D)                   = ASR(L,D) restricted to S(L)
Defense efficacy    E(L,D)  = 1 − uASR(L,D) / uASR_undefended(L)
```

Report `E` and `uASR` alongside raw ASR. The claim under test: **the gap between raw ASR and uASR grows as language resource level falls.**

**C3 — A localized multilingual agent-security suite.**
AgentDojo-derived, but *localized* rather than translated — MAPS's stated limitation is that it kept English environments and translated only the inputs. Localizing the environment (names, emails, currency, dates, addresses, document bodies) is the part a solo student in India can do better than a lab, and it is the data moat the parent file says to look for.

**C4 — A matched benign corpus per language.**
The FPR denominator nobody built. Benign tool outputs — real-shaped emails, web pages, documents, tool responses — in each language condition.

**C5 — A calibrated firewall (the deployable artifact).**
Per-language threshold calibration plus a script-normalization pre-pass, with the recovered frontier and the added latency/token cost measured honestly.

---

## 3. Language conditions

Five conditions, chosen so that *script* and *language* can be separated — the same ablation trick S1 in the parent file uses:

| ID | Condition | Purpose |
|---|---|---|
| `en` | English | Baseline; where every defense was tuned |
| `hi` | Hindi, Devanagari | Non-Latin script, mid-resource |
| `hi-latn` | Hinglish, romanized Hindi | **Same language, Latin script** — isolates script from language |
| `bn` | Bengali, Bengali script | Second non-Latin script, lower resource |
| `mixed` | Code-switched EN↔HI within a single payload | The attack style that actually appears in the wild |

`hi` vs `hi-latn` is the sharp experiment. If detectors recover on `hi-latn`, the failure is tokenization/script. If they do not, the failure is semantic coverage. Those are two different conclusions and the experiment costs nothing extra.

Stretch: `ta` (Tamil) for a third script, only if Phases 0–4 land on time.

---

## 4. Defense suite

Three classes, because the cross-class interaction is where the interesting result lives:

**Class A — text classifiers** (expected: collapse off-English)
- Meta Prompt-Guard-2
- ProtectAI DeBERTa-v3 injection detector
- An XLM-R baseline fine-tuned on the Hindi/Hinglish corpus — the "someone already tried multilingual" control

**Class B — prompting defenses** (expected: partial degradation)
- Spotlighting / delimiter marking
- Sandwich defense
- Explicit instruction-hierarchy system prompt

**Class C — architectural / policy defenses** (expected: ASR-invariant, utility-variable)
- Tool-call allowlist + parameter validation
- A CaMeL-lite data/control-flow separation with provenance tags on untrusted spans
- Honeytool detection (AgentShield-style), reimplemented as the language-agnostic reference point

**The hypothesis grid:**

| | ASR off-English | FPR off-English | Utility off-English |
|---|---|---|---|
| A — classifiers | H1: degrades | **H2: degrades severely** | collapses via over-block |
| B — prompting | H3: degrades | mild | mild |
| C — architectural | H4: invariant | invariant | **H5: degrades** (agent flails → more policy blocks) |

H2 and H5 are the two cells no published work fills. H5 is the counter-intuitive one: architectural defenses are advertised as language-agnostic, and in *security* they are — but if the agent is worse in Hindi it takes more wrong-but-benign actions, so a strict policy blocks more legitimate work. **Language-agnostic security, language-dependent usability.**

---

## 5. Metrics reported

Per (language × defense × model):

- `ASR` raw, and `uASR` restricted to the scorable set
- `E` defense efficacy relative to that language's undefended ceiling
- `FPR` on the matched benign corpus
- `BenignUtility` — user task completion under the defense
- `C(L)` capability ceiling
- **Frontier plot:** BenignUtility vs (1 − uASR), one curve per language, threshold swept
- Cost: added p50/p95 latency and tokens per tool call

The headline figure is the frontier plot with five curves that do not lie on top of each other.

---

## 6. Feasibility and budget

| Resource | Need |
|---|---|
| Compute | T4-class session for Class A classifier inference + XLM-R fine-tune. Everything else is API. |
| Primary model | One cheap fast model for the full grid |
| Spot-check model | One frontier model on a 20% stratified slice, to show the finding is not model-specific |
| Scope | ~150 stratified security cases × 5 languages × ~8 defense configs × 2 models ≈ 7.5k agent runs |
| Est. API spend | $60–110 at cheap-model pricing; hard-capped per phase |
| Human labor | Localization QA is the bottleneck — see risk R1 |

---

## 7. Risks

**R1 — Translation quality is the whole project.** Machine-translated payloads produce "translationese" artifacts that a classifier learns instead of the attack — a stated limitation of the Sharma et al. Hindi work. *Mitigation:* human-verify 100% of injection payloads and a 20% sample of benign content. You are a native speaker of a target language; this is the moat, so spend the time here. Budget a full week.

**R2 — AgentDojo churn.** The upstream benchmark moves. *Mitigation:* pin a commit, vendor the subset, record the SHA in the repo.

**R3 — Scope.** This is an 8–10 week project, not R4's 5–7. *Mitigation:* the de-scoping ladder in the implementation plan — drop `bn`, then `mixed`, then Class B, in that order. The `en` / `hi` / `hi-latn` trio alone still supports the paper.

**R4 — Someone publishes this first.** The area moves in weeks. *Mitigation:* the capability-ceiling protocol (C1/C2) is the durable contribution; even if the empirical numbers get scooped, a metric correction that invalidates prior numbers survives. Re-run the arXiv search at the start of every phase.

**R5 — Framing collision with rejected ideas.** Your notes reject *MCP safety gateway / MCP provenance / MCP trust scoring* as crowded. This project ships middleware that can sit on an MCP transport, and that is fine — but **do not frame it as an MCP gateway.** MCP is a deployment detail in one paragraph of the README. The claim is about cross-lingual measurement, not about MCP.

---

## 8. Output shape

**Resume face.** Deployed firewall middleware with a real frontier behind it.
> *"Cross-lingual injection firewall: per-language calibration cut false-positive rate on benign Hindi tool output from 34% to 4% at equal attack-success reduction; audited 8 defenses across 5 language conditions over 7.5k agent runs."*

**Research face.** Workshop paper.
> *"Utility-normalized cross-lingual evaluation of prompt-injection defenses"* — venue: an ACL/EMNLP security or multilingual workshop, or a NeurIPS/ICLR safety workshop. The benchmark + metric artifact outlives the paper, which is the property S3 in the parent file identifies as what labs actually cite.

**Both faces come from one codebase.** That is the bridge property R4 was selected for in the first place.

---

## 9. Kill criteria

Stop and re-plan if, by end of Phase 2:

- The capability ceiling `C(L)` is ≈1.0 for every language on the chosen task subset — the confound you are correcting does not exist at this scale, so the metric contribution evaporates. *Fallback:* pivot to the FPR/frontier contribution alone, which still stands.
- A paper appears doing exactly C1+C2. *Fallback:* pivot to the localization axis (C3) — full environment localization is still unclaimed, and MAPS names it as their limitation.

---

## 10. Pre-commitment (parent file § *Before committing*)

- [x] **arXiv/GitHub searched.** Near-misses found and catalogued in §0. Re-check each phase.
- [x] **Abstract written both ways.** Positive: *"raw ASR overstates multilingual defense efficacy by N points."* Null: *"utility normalization does not change defense ranking, so existing multilingual security numbers can be trusted — here is the evidence they previously lacked."* Both are worth reading.
- [x] **Results table named before starting.** §5.
- [ ] **Data access confirmed in week 1.** AgentDojo is open; the localization corpus is self-produced. Confirm in Phase 0, not Phase 3.

---

## Appendix — prior art, with links

**Benchmarks**
- AgentDojo — 97 tasks / 629 security cases, stateful environments · arXiv 2406.13352
- InjecAgent — 1,054 cases, 17 user tools, 62 attacker tools
- BIPIA — first IPI-specific benchmark, 250 attacker objectives
- WASP — web-agent injection, built on WebArena · arXiv 2504.18575
- LivePI — realistic IPI benchmarking · arXiv 2605.17986
- PIArena — evaluation platform · arXiv 2604.08499
- MAPS — 12-language agent perf + security, 805 tasks · arXiv 2505.15935

**Defenses**
- CaMeL — dual-LLM + capabilities + IFC interpreter · arXiv 2503.18813
- IPIGuard — tool-dependency-graph defense · arXiv 2508.15310
- CommandSans — surgical prompt sanitization · arXiv 2510.08829
- AgentWatcher — rule-based monitor · arXiv 2604.01194
- LlamaFirewall — open-source guardrail system · arXiv 2505.03574
- Meta SecAlign — injection-resistant foundation model · arXiv 2507.02735
- Securing AI Agents with Information-Flow Control · arXiv 2505.23643
- Tool-result parsing defense · arXiv 2601.04795

**Attacks / adaptive evaluation**
- Zhan et al. — adaptive attacks break 8 IPI defenses · arXiv 2503.00061
- AutoDojo — adaptive black-box attacks on IPI defenses · arXiv 2606.15057
- Adaptive evaluation of out-of-band defenses · arXiv 2606.26479
- PISmith — RL-based red teaming · arXiv 2603.13026
- Sandwich attack — multi-language mixture adaptive attack · arXiv 2404.07242

**Multilingual safety / the closest neighbours**
- **AgentShield** — deception-based detection, 4 languages on AgentDojo, classifier-collapse finding · arXiv 2605.11026 · *closest competitor, read first*
- MIPIAD — multilingual IPI detection, text classification only · arXiv 2605.07269
- IndicGuard — Indic safety guard model + dataset · arXiv 2606.22841
- Sharma et al. — Hindi/Hinglish prompt-injection detection, direct attacks only · PMC13201763
- MrGuard · arXiv 2504.15241 — X-Guard · arXiv 2504.08848 — SEALGuard · arXiv 2507.08898 — TWGuard · arXiv 2604.16542
