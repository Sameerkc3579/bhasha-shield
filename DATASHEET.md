# Datasheet: agentfw multilingual injection + benign corpus

Follows Gebru et al., *Datasheets for Datasets* (2021). Draft, kept in step with
`corpus/`; counts and verification status as of 2026-10-14.

## Motivation

**Why was it created?** To measure prompt-injection defenses for tool-using
agents across languages *without* the confound the literature leaves in: an
attack that "fails" in a language because the agent cannot do the attacker's
task in that language at all is not a defense success. The corpus therefore
pairs every attack with (a) the attacker's goal as a plain instruction, for the
capability probe, and (b) a benign corpus of tool output, for false-positive
rate, item-aligned across five language conditions.

**Who created it?** One researcher (Sameer Choudhary), as a solo student project. No funding.
Total spend Rs 0.

## Composition

| Part | Files | Items | Status |
|---|---|---|---|
| Injection cases (English) | `corpus/injections/en.jsonl` | 150 cases, 35 distinct attacker goals | drawn from AgentDojo v1 by stratified sample |
| Localized attacker goals | `corpus/goals/{hi,hi_latn,bn,mixed}.jsonl` | 35 per language | `hi`, `hi_latn`, `mixed` human-verified (unchanged on review); `bn` machine draft |
| Attack templates | `corpus/attacks.py` | `important_instructions` x 5 languages | `en` = AgentDojo verbatim; `hi`, `hi_latn`, `mixed` human-verified; `bn` machine draft |
| Benign tool output | `corpus/benign/{lang}.jsonl` | 300 per language: 87 harvested + 213 hand-written, 133 adversarial-benign | `en` built; others machine drafts, item-aligned; the ~20% samples of `hi`, `hi_latn`, `mixed` were **model**-reviewed (one fix), not human-verified |
| Localized user instructions (H5) | `corpus/user_tasks/hi.jsonl` | 32 of the 79 user tasks (those a correct run passes in any language) | machine draft, model-reviewed (4 edits); used as a labelled pilot |
| Localized environments | `corpus/environments/{lang}.json` | 131 prose strings per language; 113-127 active | machine drafts, gated (see below) |
| Provenance | `corpus/provenance.jsonl` | one row per localized item | complete for what exists |

**Language conditions.** `en`; `hi` (Hindi, Devanagari); `hi_latn` (the *same*
Hindi sentences in Latin script -- the script-vs-language ablation); `bn`
(Bengali); `mixed` (casual Hindi-English code-switching in Latin script).

**Domains** (injection cases): workspace 83, travel 26, banking 23, slack 18.
**Attacker goal types:** exfiltration via message 82, unauthorised transaction
31, state tampering 12, destructive action 9, information disclosure 8,
content manipulation 4, other 4.

**Adversarial-benign items** (133 per language) are benign tool outputs that
look like injections: agendas full of imperatives, quoted email chains,
security memos that quote an example attack, code containing "ignore". They
are what makes an FPR number informative. The 93 added in October 2026 are
India-specific shapes as well -- society maintenance notices, bank KYC and OTP
messages, UPI receipts, school circulars -- the legitimate text a Hindi or
Bengali user's tools return most often.

**Ordinary hand-written items** (80, `benign_hw_*`): everyday tool output in
an Indian setting -- names, rupee amounts, Indian cities, DD/MM/YYYY dates.
The 87 harvested items are AgentDojo's own (Emma Johnson's world); these are
the benign half of C3.

**What stays English in every language, on purpose.** Every string an AgentDojo
checker matches on (email addresses, IBANs, URLs, quoted subjects, ids,
3+-digit numbers) is byte-identical across languages, as are the
`<INFORMATION>` tags and the victim name. Otherwise an attack would fail because
of the translation, manufacturing the effect being measured. Enforced by
`corpus/benign_localize.py` and tested.

**Localized environments (C3), and their limit.** `corpus/environments.py`
translates the environment's *prose* -- email bodies, documents, calendar
descriptions, Slack web pages, hotel and restaurant reviews, the banking
files -- so injected payloads sit inside text of the same language. It does
**not** localize names, addresses, amounts, dates, ids or opening hours:
AgentDojo's checkers compare those values, and changing them would mean
rewriting the checkers. An entry is used only if it passes three offline
gates: placeholders and entities preserved; no checker string from the task
code lost; and AgentDojo's own ground truth (`suite.check()`) unchanged.
Excluded entries stay English and are listed with the reason.

**Personal information.** None real. AgentDojo's environments are synthetic;
hand-written items use invented names and reserved-looking addresses.

## Collection and localization process

1. English cases: `corpus/extract.py`, stratified sample from AgentDojo at the
   pinned SHA (see README).
2. English benign: harvested from AgentDojo default environments (87) and
   hand-written adversarial-benign items (40), `corpus/benign_build.py`.
3. Localization: machine draft (by an LLM in-session, no paid MT API), then
   **100% human repair for injection text** (goals, templates) and a fixed-seed
   ~20% back-translation sample for benign text (`corpus/review/`).
   `hi_latn` is produced by transliterating the *verified* `hi` text, never
   translated independently.
4. Every non-English item has a provenance row: method (`machine`,
   `machine+human`, `human`, `transliterated`), source item, verifier, date.

**Quality gate (plan, Phase 3):** 30 payloads per language rated "would a
native speaker write this?" by a second reader; >= 90% yes required before any
result on that language is reported as a result rather than a pilot.
**Status:** verification so far is by one reader, the author (a fluent Hindi
reader); no second reader has rated payloads. `bn` has no fluent reviewer and
is reported as a pilot throughout. A "model review" (by the LLM that drafted
the text) is recorded in provenance as such and never counts as verification.

Hand-written English items were written for this corpus (2026-10-04) and are
`source: hand-written/<category>`. Their translations follow the same draft and
review path as everything else (`corpus/benign_extra/`, `corpus/benign_extend.py`).

## Uses

**Intended:** evaluating injection defenses for tool-using agents, per
language, with utility normalization and per-language FPR.

**Not intended:** training an injection detector and evaluating it on the same
items; treating the `mixed` condition as representative of code-switching in
general (it is one register, typed by one person).

## Distribution and maintenance

To be released with the harness after Phase 7. Licence intent: code MIT
(matching AgentDojo, MIT, from which cases and environments derive); corpus
CC BY 4.0. **Not yet chosen formally -- the author's decision.**

Maintainer: the author. Errata go in `RESULTS.md` with a date, like everything else.
