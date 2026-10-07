"""Meta Llama Prompt Guard 2 (86M) -- the multilingual off-the-shelf detector.

mDeBERTa-based, and Meta lists Hindi among its training languages, which makes
it the fair "a multilingual detector already exists" comparison for `hi`. It
says nothing about romanized Hindi or Bengali; whether it transfers there is
exactly what the grid measures.

The weights are **gated** on Hugging Face: accept the Llama licence on the
model page and log in (`huggingface-cli login`, or HF_TOKEN in Colab secrets)
before the first run. Free, but it is a manual step.

Labels: LABEL_0 = benign, LABEL_1 = malicious.
"""

from __future__ import annotations

from defenses.classifiers.hf import HFClassifier


class PromptGuard2(HFClassifier):
    name = "promptguard2"
    model_id = "meta-llama/Llama-Prompt-Guard-2-86M"
    positive_label = "LABEL_1"
