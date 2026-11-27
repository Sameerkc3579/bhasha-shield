"""Load a trained guard run as a scorer: texts -> P(injection) (docs/v2-plan.md, Phase 3-4).

A run directory from `guard.train` (its `model/`, or `fake_model.pkl` from `--fake`), an int8 ONNX
export (`guard.export_onnx`: a directory with `model_quantized.onnx`), a saved model directory, or
a Hugging Face id. Chunking (1500 chars, 200 overlap, max score) is not done here: it
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
    if (p / "model_quantized.onnx").exists():
        return onnx_scorer(p)
    model_dir = p / "model" if (p / "model").is_dir() else p
    return hf_scorer(str(model_dir) if model_dir.exists() else str(path), POSITIVE, revision=revision, device=device)


def onnx_scorer(path: Path, *, batch_size: int = 16) -> Scorer:
    """P(INJECTION) from an ONNX Runtime export (optimum), on the CPU."""
    state: dict = {}

    def score(texts: Sequence[str]) -> list[float]:
        if "model" not in state:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            state["model"] = ORTModelForSequenceClassification.from_pretrained(path, file_name="model_quantized.onnx")
            state["tok"] = AutoTokenizer.from_pretrained(path)
            state["pos"] = {v: k for k, v in state["model"].config.id2label.items()}[POSITIVE]
        import torch

        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            enc = state["tok"](list(texts[i:i + batch_size]), truncation=True, max_length=512, padding=True,
                               return_tensors="pt")
            logits = state["model"](**enc).logits
            out += torch.softmax(logits.float(), -1)[:, state["pos"]].tolist()
        return out
