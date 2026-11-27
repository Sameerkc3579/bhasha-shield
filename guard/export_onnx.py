"""Export a guard run to ONNX with int8 weights and time it on the CPU (docs/v2-plan.md, Phase 5 step 2).

    python -m guard.export_onnx --run guard/runs/mdeberta__full__s1             # -> <run>/onnx/
    python -m guard.export_onnx --run guard/runs/mdeberta__full__s1 --bench     # + latency table

Export: Hugging Face Optimum (`ORTModelForSequenceClassification`, then dynamic int8 quantization
with `ORTQuantizer`). `guard.infer.load_scorer` then loads `<run>/onnx/` like any other model, so
`scripts/guard_eval.py` can score the frozen set with the int8 model.

**Quantization moves scores.** The thresholds the proxy uses must be fitted on the scores of the
model it runs: score the calibration half with the int8 model (`guard_eval --label
bhasha_guard@<run>-int8`) and fit from those, never reuse the fp32 thresholds.

`--bench` times one check (chunking + model, as the proxy does it) per text on a fixed sample of
the frozen set's texts, for the fp32 model and the int8 one, and reports the median and p95 in
ms with the machine, thread count and library versions: `analysis/out/latency/<run>.json`.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

OUT = Path("analysis/out/latency")


def percentile(xs: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(1, min(len(s), int(-(-q * len(s) // 1))))   # ceil(q * n), at least 1
    return s[k - 1]


def bench(check: Callable[[str], object], texts: Sequence[str], *, warmup: int = 5) -> dict:
    """Wall time of one check per text, after `warmup` untimed checks."""
    for t in texts[:warmup]:
        check(t)
    ms = []
    for t in texts:
        t0 = time.perf_counter()
        check(t)
        ms.append((time.perf_counter() - t0) * 1000)
    return {"n": len(ms), "median_ms": median(ms) if ms else None, "p95_ms": percentile(ms, 0.95),
            "max_ms": max(ms) if ms else None}


def export(run: Path) -> Path:
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    src, out = run / "model", run / "onnx"
    model = ORTModelForSequenceClassification.from_pretrained(src, export=True)
    model.save_pretrained(out / "fp32")
    AutoTokenizer.from_pretrained(src).save_pretrained(out / "fp32")
    q = ORTQuantizer.from_pretrained(out / "fp32")
    q.quantize(save_dir=out, quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=False))
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    return out


def sample_texts(n: int = 200, seed: int = 7) -> list[str]:
    from corpus.testset import LANGS_V2, OUT as V2, FrozenItem

    items = [FrozenItem.model_validate_json(l) for lang in LANGS_V2 for l in
             (V2 / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [i.text for i in random.Random(seed).sample(items, min(n, len(items)))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args(argv)

    out = a.run / "onnx"
    if not (out / "model_quantized.onnx").exists():
        print("exported ->", export(a.run))
    if a.bench:
        import torch

        from corpus.schema import ToolOutput
        from defenses.base import RunContext
        from defenses.classifiers.bhasha_guard import BhashaGuard

        texts = sample_texts(a.n)
        ctx = RunContext(case_id="bench", lang="en", model="none", user_task="")
        res = {"run": a.run.name, "texts": len(texts), "threads": torch.get_num_threads(),
               "machine": platform.processor() or platform.machine(), "python": platform.python_version()}
        for name, path in (("fp32", a.run), ("int8", out)):
            g = BhashaGuard(model_id=str(path))
            res[name] = bench(lambda t: g.inspect(ToolOutput(tool_name="t", content=t, lang="en"), ctx), texts)
            print(f"{name}: median {res[name]['median_ms']:.1f} ms, p95 {res[name]['p95_ms']:.1f} ms")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{a.run.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
