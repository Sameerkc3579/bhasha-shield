"""ProtectAI DeBERTa-v3 prompt-injection classifier (v2) -- the English-only baseline.

Trained on English data only (per its model card), so it is the clean test of
the paper's FPR question: an English-calibrated detector shipped unchanged to
Hindi, romanized Hindi and Bengali tool output. Ungated; Apache-2.0.

Labels: SAFE / INJECTION.
"""

from __future__ import annotations

from defenses.classifiers.hf import HFClassifier


class ProtectAIDeberta(HFClassifier):
    name = "protectai_deberta"
    model_id = "protectai/deberta-v3-base-prompt-injection-v2"
    positive_label = "INJECTION"
