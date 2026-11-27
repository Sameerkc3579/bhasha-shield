# bhasha-shield: rules for Claude Code

## Project
- Python package `agentfw`. v1 is frozen at tag `v1.0`; v2 work happens on branch `v2`.
- The plan is docs/v2-plan.md. Read the current phase before starting a task.
- Running log of every number: RESULTS.md.

## Never
- Never edit the frozen test set; tests/test_testset_hash.py guards it.
- Never train, tune or pick a model on the test half. Use the calibration half or validation split.
- Never write a number into RESULTS.md, README or the paper unless it regenerates from a trace on disk.
- Never invent results, citations or dataset sizes. If unknown, say so.
- Never commit datasets, model weights, caches, .env or tokens.
- Never call a paid hosted API. Models are open-weight on free GPUs.
- Never put a person's name in a frozen or released artifact (test set, dataset, model card, paper
  links): the target venue, USENIX SAIS '27, is double-blind. Reviewers appear as codes from
  corpus/reviewers.json.

## Always
- Write or update tests first. Run `python -m pytest` before every commit.
- Keep the `--fake` (no-GPU) paths working; CI runs only those.
- Fix seeds and log model IDs and versions in every trace.
- GPU work goes into resumable, cached notebook cells. The user runs them and brings back the cache archive.
- Every new number gets a dated RESULTS.md entry, including bad ones.
- Small commits, one task each; the message says what changed and why.
- Report results per language. Undefined stays None, never 0.

## Local machine (Windows laptop, 8 GB RAM)
- Python: `~/.venvs/agentfw/Scripts/python` (outside OneDrive; quote the path, it has a space).
- CPU classifier jobs: `OMP_NUM_THREADS=2`, one language per process.
- Windows console: set `PYTHONIOENCODING=utf-8` before printing Devanagari.
- A Colab archive arrives as `cache (N).tgz` or an extracted `cache (N)/`. Merge rule: compare first,
  never overwrite `.cache/` entries, replace a trace only if the new one extends the local copy.
