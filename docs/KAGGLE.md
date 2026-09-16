# Running the model on Kaggle — ₹0

The project runs on an open-weight model served locally. No API key, no spend,
no rate limits, and no model drift: weights are pinned by revision, so a number
from week 3 is still reproducible in week 9.

**Model:** `Qwen/Qwen2.5-7B-Instruct` — native tool calling, Apache-2.0, fits a
16 GB T4 in 4-bit, and the broadest language coverage among small open models
that call tools reliably.

---

## Why this is the right backend, not a compromise

The confound this project measures is **larger** on a smaller model. A frontier
model is competent in Hindi, so C(L) sits near 1.0 in every language and the
correction barely matters. A 7B model has a genuinely low capability ceiling
off-English — which is precisely the effect the paper quantifies.

The risk runs the other way: if C(L) collapses to near zero in Bengali, that
column becomes unscorable. **Phase 2 exists to detect exactly that**, before the
grid is run. If it happens, the de-scoping ladder drops `bn` and the
`en` / `hi` / `hi-latn` trio still carries the paper.

---

## Session setup

Kaggle gives ~30 GPU-hours/week. Notebook → Settings → Accelerator → **GPU T4 x2**,
and turn **Internet on**.

```python
!pip install -q vllm
```

Serve the model with an OpenAI-compatible endpoint:

```python
import subprocess, time, requests

server = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-7B-Instruct",
    "--revision", "main",             # pin a commit SHA once you have one
    "--dtype", "half",
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.90",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",   # Qwen2.5 uses the Hermes tool format
    "--port", "8000",
])

for _ in range(120):                   # first boot downloads ~15 GB
    try:
        if requests.get("http://localhost:8000/v1/models", timeout=2).ok:
            print("server up"); break
    except Exception:
        time.sleep(10)
```

`--enable-auto-tool-choice` and `--tool-call-parser hermes` are both required.
Without them vLLM returns tool calls as plain text, every run scores as a
failed attack, and the whole grid quietly reads as "perfectly secure" — the
exact failure mode this project exists to call out.

---

## Verify tool calling before spending GPU hours

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

r = c.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "What is my balance?"}],
    tools=[{"type": "function", "function": {
        "name": "get_balance",
        "description": "Get the account balance",
        "parameters": {"type": "object", "properties": {}},
    }}],
    tool_choice="auto",
)
print(r.choices[0].message.tool_calls)   # must NOT be None
```

If that prints `None`, stop and fix the server flags. Nothing downstream is
meaningful until this works.

---

## Run the project against it

```python
!git clone <your-repo> agentfw && cd agentfw && pip install -q -e .
!cd agentfw/vendor && git clone --depth 1 https://github.com/ethz-spylab/agentdojo.git
!cd agentfw/vendor/agentdojo && git checkout 089ed468cf3ed0322acc66b0211f26d9d90dbf60 && pip install -q -e .
```

```bash
python -m scripts.phase0_smoke --backend local     # closes Phase 0
python -m scripts.phase1_baseline --backend local  # English undefended baseline
```

---

## Throughput

A T4 serves Qwen2.5-7B at roughly 20–30 tok/s single-stream, but vLLM batches
continuously, so concurrency is what matters. Run cases in parallel:

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(run_one, cases))
```

Rough budget at 8-way concurrency: the 150-case English baseline finishes well
inside one session; the full Phase 5 grid is a few sessions. The response cache
means an interrupted session resumes without recomputing anything.

**Save the cache between sessions** — Kaggle wipes `/kaggle/working` on restart:

```python
!tar czf /kaggle/working/cache.tgz .cache traces
```

Attach that output as a dataset input to the next session and untar it before
running, or the cache starts cold every time.

---

## Determinism

Unlike the hosted path — where the SDK has removed `temperature` entirely — a
locally served model accepts sampling parameters, so `temperature=0` is set and
runs are reproducible. That recovers the determinism the plan originally assumed.
`harness/local_client.py` sets it; do not change it without noting the change in
`RESULTS.md`.

---

## If Qwen turns out too weak in Indic languages

Phase 2 measures this directly. Options, in order of preference:

1. **Report it.** A low C(L) in Bengali is a finding, not a failure — it is the
   confound, made visible.
2. **Swap the model.** `sarvamai/sarvam-1` and `ai4bharat` models target Indic
   specifically; check tool-calling support before committing.
3. **Drop the language.** De-scoping ladder: `mixed` first, then `bn`.
