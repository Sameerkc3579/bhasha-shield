"""bhasha_guard: the v2 guard as a Class A defense (docs/v2-plan.md, Phase 4 step 1).

Points at a `guard.train` run directory, a saved model directory or a Hugging Face id, via
`model_id=` or AGENTFW_GUARD_MODEL; without one it refuses to construct rather than fall back to
an untrained head. Chunking and the always-recorded score are HFClassifier's, as for the
baselines. Thresholds are per language and come from the calibration half (analysis/guard_tables.py);
the 0.5 default here is only the constructor's.
"""

from __future__ import annotations

import os

from defenses.classifiers.hf import HFClassifier, Scorer


class BhashaGuard(HFClassifier):
    name = "bhasha_guard"
    positive_label = "INJECTION"

    def __init__(self, *, model_id: str | None = None, scorer: Scorer | None = None,
                 device: int | str | None = None, revision: str | None = None, **kw):
        model_id = model_id or os.environ.get("AGENTFW_GUARD_MODEL")
        if not model_id and scorer is None:
            raise RuntimeError("bhasha_guard has no model: pass model_id= or set AGENTFW_GUARD_MODEL to a "
                               "guard run directory or Hugging Face id")
        if scorer is None:
            from guard.infer import load_scorer

            scorer = load_scorer(model_id, device=device, revision=revision)
        super().__init__(model_id=model_id or "unset", scorer=scorer, revision=revision, **kw)
