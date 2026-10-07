"""XLM-R fine-tuned on Hindi/Hinglish injection data -- the "just train one" control.

The plan's question for this defense: does a detector fine-tuned *for* Hindi
close the gap that calibration (C5) is meant to close? It is a control, not a
contribution, so it only needs to be a reasonable fine-tune, reported plainly.

The checkpoint does not exist yet. Training needs a GPU and a labelled
Hindi/Hinglish injection set; until then this wrapper refuses to construct
without a model path, rather than silently falling back to an untrained head.
Point it at a local directory or Hub id via `model_id=` or AGENTFW_XLMR_MODEL.

Labels: whatever the fine-tune writes; LABEL_1 = injection by default.
"""

from __future__ import annotations

import os

from defenses.classifiers.hf import HFClassifier, Scorer


class XLMRFinetuned(HFClassifier):
    name = "xlmr_finetuned"
    positive_label = "LABEL_1"

    def __init__(self, *, model_id: str | None = None, scorer: Scorer | None = None,
                 positive_label: str | None = None, **kw):
        model_id = model_id or os.environ.get("AGENTFW_XLMR_MODEL")
        if not model_id and scorer is None:
            raise RuntimeError(
                "xlmr_finetuned has no checkpoint yet: pass model_id= or set AGENTFW_XLMR_MODEL "
                "to a fine-tuned XLM-R directory or Hub id."
            )
        if positive_label:
            self.positive_label = positive_label
        super().__init__(model_id=model_id or "unset", scorer=scorer, **kw)
