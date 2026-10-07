"""Append-only JSONL traces.

Standing rule from implementation-plan.md: never report a number you cannot
regenerate from a trace. Every table and figure in the paper is rebuilt from
these files, so they record the inputs as well as the outcome -- including the
git SHA and model version string, because "the model changed mid-experiment" is
a real and common way to lose a month of work.
"""

from __future__ import annotations

import json
import re
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_ROOT = Path("traces")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    """Make a string safe as a filename component.

    Open-weight model ids carry a namespace -- `Qwen/Qwen2.5-7B-Instruct` -- and
    the slash silently turns into a directory that does not exist. The model name
    is part of the trace identity, so it has to survive, just flattened.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


class TraceEvent(BaseModel):
    ts: str = Field(default_factory=_now)
    kind: str                       # run_start | tool_call | call_check | verdict | run_end | note
    run_id: str
    case_id: str | None = None
    lang: str | None = None
    defense: str | None = None
    model: str | None = None
    payload: dict[str, Any] = {}


class RunTrace:
    """One JSONL file per (phase, language, defense) cell.

    Splitting by cell means a crashed run loses at most one cell, and a partial
    grid is still a set of complete, analysable cells.
    """

    def __init__(
        self,
        *,
        phase: str,
        lang: str,
        defense: str,
        model: str,
        root: Path | str = DEFAULT_ROOT,
    ):
        self.phase = phase
        self.lang = lang
        self.defense = defense
        self.model = model
        self.root = Path(root) / phase
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{_slug(lang)}__{_slug(defense)}__{_slug(model)}.jsonl"
        self._fh = None

    def __enter__(self) -> "RunTrace":
        self._fh = self.path.open("a", encoding="utf-8")
        self.write(
            "note",
            run_id="session",
            payload={
                "git_sha": _git_sha(),
                "started": _now(),
                "python": os.sys.version.split()[0],
                "phase": self.phase,
            },
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write(self, kind: str, *, run_id: str, case_id: str | None = None, payload: dict | None = None) -> None:
        ev = TraceEvent(
            kind=kind,
            run_id=run_id,
            case_id=case_id,
            lang=self.lang,
            defense=self.defense,
            model=self.model,
            payload=payload or {},
        )
        line = ev.model_dump_json() + "\n"
        if self._fh:
            self._fh.write(line)
            self._fh.flush()  # flush per event: a killed run keeps its history
        else:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    @staticmethod
    def read(path: Path | str) -> list[TraceEvent]:
        out: list[TraceEvent] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(TraceEvent.model_validate_json(line))
        return out
