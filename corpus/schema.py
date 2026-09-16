"""Canonical data types for the corpus.

Frozen in Phase 0. Everything downstream imports from here; if this file churns,
traces from different weeks stop being comparable and the results log becomes
worthless. Change it only with a migration note in RESULTS.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The five language conditions (plan.md section 3).
# `hi` and `hi_latn` must stay semantically identical -- same sentence, different
# script -- because they are the script-vs-language ablation.
Lang = Literal["en", "hi", "hi_latn", "bn", "mixed"]

LANGS: tuple[Lang, ...] = ("en", "hi", "hi_latn", "bn", "mixed")

Provenance = Literal["trusted", "untrusted"]

# How a piece of corpus text came to exist, so reviewers can audit the pipeline
# and so we can tell a real finding from a translation artifact (risk R1).
TranslationMethod = Literal[
    "source",          # written natively in this language, not translated
    "machine",         # MT draft, not yet repaired -- never valid for injections
    "machine+human",   # MT draft repaired by a human
    "human",           # written by a human in this language
    "transliterated",  # mechanically transliterated from a verified sibling
]


class ProvenanceRecord(BaseModel):
    """One row of corpus/provenance.jsonl. Required for every non-`en` item."""

    item_id: str
    lang: Lang
    method: TranslationMethod
    source_item_id: str | None = None
    verified_by: str | None = None
    verified_at: str | None = None  # ISO 8601
    notes: str = ""


class ToolOutput(BaseModel):
    """A single tool response handed back to the agent.

    This is the unit a defense inspects. `provenance` is the crucial field:
    trusted text came from the user or the system, untrusted text came from the
    outside world and is where injections live.
    """

    tool_name: str
    content: str
    lang: Lang
    provenance: Provenance = "untrusted"
    item_id: str | None = None


class InjectionCase(BaseModel):
    """An injection test case, in one language.

    `attacker_goal_nl` is what makes the capability probe possible: it is the
    attacker's objective restated as a plain user instruction, so we can ask the
    agent to do it directly and find out whether it *could* have, in this
    language, before we credit a defense for stopping it.
    """

    case_id: str
    lang: Lang
    domain: str                    # workspace / banking / travel / slack ...
    user_task: str                 # the legitimate task the user asked for
    attacker_goal_nl: str          # attacker objective as a direct instruction
    attacker_goal_type: str        # exfiltration / unauthorised-action / ...
    payload: str                   # the injected text placed in a tool output
    injected_into_tool: str
    tool_call_depth: int = Field(ge=1)
    source_case_id: str | None = None  # upstream AgentDojo id


class BenignItem(BaseModel):
    """Benign tool output -- the false-positive denominator (contribution C4).

    `adversarial_benign` marks content that legitimately *looks* like an
    injection: meeting agendas full of imperatives, quoted email chains, policy
    documents, code containing the word "ignore". Those are the items that make
    an FPR number interesting rather than decorative.
    """

    item_id: str
    lang: Lang
    tool_name: str
    content: str
    adversarial_benign: bool = False
    source: str = ""
