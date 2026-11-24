"""Load a trained guard run as a scorer: texts -> P(injection) (docs/v2-plan.md, Phase 3-4).

A run directory from `guard.train` (its `model/`, or `fake_model.pkl` from `--fake`), a saved model
directory, or a Hugging Face id. Chunking (1500 chars, 200 overlap, max score) is not done here: it
is `defenses.classifiers.hf.HFClassifier`'s, the same for every classifier, as v1 did it.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

from defenses.classifiers.hf import Scorer, hf_scorer

POSITIVE = "INJECTION"


def load_scorer(path: str | Path, *, device: int | str | None = None, revision: str | None = None) -> Scorer:
    p = Path(path)
    if (p / "fake_model.pkl").exists():
        model = pickle.loads((p / "fake_model.pkl").read_bytes())   # our own file, from guard.train --fake

        def score(texts: Sequence[str]) -> list[float]:
            return model.predict_proba(list(texts))[:, 1].tolist()

        return score
    model_dir = p / "model" if (p / "model").is_dir() else p
    return hf_scorer(str(model_dir) if model_dir.exists() else str(path), POSITIVE, revision=revision, device=device)
