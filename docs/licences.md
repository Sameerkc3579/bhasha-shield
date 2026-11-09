# Licences (v2 Phase 0)

One row per model, dataset and text source that v1 used or v2 plans to use. **The licence columns
are for Sameer to fill in from each source's own page** (model card, LICENSE file, PyPI page),
not from memory. Gate G0 needs this table complete.

What decides the outcome, for each row:

- **Use**: may we run it for research at all?
- **Derive**: may we train on it or on its outputs, and release what we train?
- **Redistribute**: may we put it (or text from it) in the public Hugging Face dataset?

If a row blocks something, the plan's fallback applies: train from mDeBERTa on self-generated data and
release under CC BY 4.0 like the v1 corpus (docs/v2-plan.md, Risks and fallbacks).

## Models

| Model | Used for | Where | Licence | Use / derive / redistribute | Notes and questions to check |
|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct (AWQ) | the agent under attack (v1, v2); drafting and rewriting training items (v2) | `Qwen/Qwen2.5-14B-Instruct-AWQ` | | | Qwen's licence differs between model sizes: check the 14B card itself. May its outputs be used as training data and released? |
| Llama 3.1 8B Instruct (AWQ) | v1 second-model spot-check | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (from `meta-llama/Llama-3.1-8B-Instruct`) | | | Gated. v1 only; no v2 plan uses it. |
| Prompt Guard 2, 86M and 22M | baseline (v1, v2); candidate base for continued training (v2) | `meta-llama/Llama-Prompt-Guard-2-86M`, `meta-llama/Llama-Prompt-Guard-2-22M` | | | Gated. If the guard is trained from it, the released guard is a derivative: what name, notice and licence terms must it carry? |
| ProtectAI DeBERTa v3 prompt-injection v2 | baseline (v1, v2) | `protectai/deberta-v3-base-prompt-injection-v2` | | | |
| mDeBERTa-v3-base | candidate guard base model | `microsoft/mdeberta-v3-base` | | | The plan's fallback base model. |
| MuRIL base | candidate guard base model | `google/muril-base-cased` | | | |
| IndicBERT v2 | candidate guard base model | `ai4bharat/IndicBERTv2-MLM-only` | | | Check the exact model id and which checkpoint the plan means. |
| IndicTrans2 | first drafts of training items in Indian languages | `ai4bharat/indictrans2-en-indic-1B`, `ai4bharat/indictrans2-en-indic-dist-200M` | | | May its translations be released in a dataset? |

## Benchmark, data and text sources

| Source | Used for | Where | Licence | Use / derive / redistribute | Notes and questions to check |
|---|---|---|---|---|---|
| AgentDojo | agent environments, the 150 English cases, the `important_instructions` template, 87 harvested benign items | `github.com/ethz-spylab/agentdojo` at `089ed468` | | | `corpus/LICENSE` already carries AgentDojo's copyright notice for the derived v1 items. |
| v1 corpus, written in this project | localized goals and templates, 213 hand-written benign items, user tasks | this repo, `corpus/` | | | Released in v1 under the licence in `corpus/LICENSE`. |
| Openly licensed Indic text, for natural benign items (v2 Phase 2) | ordinary negatives wrapped in carriers | not chosen yet | | | Pick only sources whose licence allows redistribution in a public dataset; record each one here before use. |

## Libraries

Installed from PyPI; their licences matter less (we use them, we do not redistribute them), but list
any that would restrict a release.

| Library | Used for | Licence | Notes |
|---|---|---|---|
| vLLM | serving the agent model | | |
| Hugging Face Transformers, `datasets`, `accelerate`, Optimum | training, export | | |
| ONNX Runtime | CPU inference in the proxy | | |
| `indic-transliteration` | romanized Hindi | | |
| `datasketch` | MinHash dedup | | |
| Gradio | the Hugging Face Space | | |

## What v2 will release, and under what

| Artifact | Planned licence | Depends on rows above |
|---|---|---|
| Code (GitHub) | MIT, as v1 (`LICENSE`) | libraries |
| Dataset: training set + frozen test set (Hugging Face) | CC BY 4.0, as the v1 corpus, unless a source requires otherwise | AgentDojo, Qwen outputs, IndicTrans2 outputs, Indic text sources |
| Guard model (Hugging Face) | depends on the base model chosen at Gate G3 | the base model's row |
