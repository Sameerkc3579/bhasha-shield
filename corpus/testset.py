"""The frozen v2 test set (docs/v2-plan.md, Phase 1).

    python -m corpus.testset status                      # what is reviewed, what is left
    python -m corpus.testset review export hi            # -> corpus/review/testset_hi.csv (Excel)
    python -m corpus.testset review import hi            # signed rows back in, provenance stamped
    python -m corpus.testset native check hi             # validate corpus/native/hi_*.txt
    python -m corpus.testset native import hi --author "Sameer Choudhary"
    python -m corpus.testset build --pilot               # the shape, with unverified items allowed
    python -m corpus.testset build                       # the frozen set: refuses anything unverified

Every v2 claim -- the guard against calibrated PromptGuard 2 (Gate G3), the guard as an agent
defense (G4) -- is measured on this set. It is assembled from:

- **v1 positives**, exactly as `scripts/attack_scores.py` scored them in v1: each verified goal in
  the verified `important_instructions` template (`payload`), and spliced into a carrier tool
  output (`embedded`). Byte-identical to v1, so Phase 3 can reproduce v1's numbers first.
- **enframe positives** (non-English): AgentDojo's English template around the localized goal,
  the v1 diagnostic attack, in the same carriers.
- **native positives** (hi, hi_latn, mixed): templates written in the language by a person, with a
  `{goal}` slot filled by one verified goal (chosen by stable hash), bare and embedded.
- **v1 benign items** (300 per language), human-reviewed here, and **native benign items**:
  adversarial-benign text written in the language.

**Halves.** `analysis.calibrate.is_fit`, unchanged: positives by attacker goal, negatives by item
id. Every v1 item keeps its v1 side and no goal is on both sides. The calibration half is for
thresholds and model selection; the test half is read once, for the reported numbers.

**Verification** lives in `corpus/provenance.jsonl` (append-only), with the SHA-256 of the text
that was signed, so an edit after review is caught. A `hi_latn` item also records the hash of its
`hi` sibling: when the Hindi is edited, the Romanized copy goes stale until it is re-read.

**Reviewers appear as codes** (`R1`, from `corpus/reviewers.json`), never as names: the paper goes
to double-blind review (USENIX SAIS '27) and the set ships behind anonymous links, while the hash
makes the frozen bytes permanent. Provenance keeps the names; a reviewer without a code blocks
the freeze.

`build` refuses anything unverified; `build --pilot` writes to `corpus/testset/pilot/`, which is
never hashed. The frozen files are `corpus/testset/v2/{lang}.jsonl` + `manifest.json`;
`tests/test_testset_hash.py` pins their hashes from Gate G1 on. The manifest's goals and template
families are what Phase 2's training data may not use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from analysis.calibrate import is_fit
from corpus.attacks import DEFAULT_ATTACK, ENFRAME, TEMPLATE_STATUS, USER_NAME, render
from corpus.benign_localize import entities
from corpus.provenance import now, record
from corpus.schema import BenignItem, InjectionCase, ProvenanceRecord

BENIGN = Path("corpus/benign")
INJECTIONS = Path("corpus/injections/en.jsonl")
PROVENANCE = Path("corpus/provenance.jsonl")
REVIEWERS = Path("corpus/reviewers.json")     # reviewer name -> code; frozen rows carry only the code
NATIVE = Path("corpus/native")
REVIEW = Path("corpus/review")
OUT = Path("corpus/testset/v2")
PILOT = Path("corpus/testset/pilot")

LANGS_V2 = ("en", "hi", "hi_latn", "mixed")   # bn joins only with a fluent reviewer (Gate G0)
NATIVE_LANGS = ("hi", "hi_latn", "mixed")
TOOLS = ("email", "web", "document", "chat", "sms", "calendar", "review", "json")
SALT = "calib-2026"                            # analysis.calibrate.is_fit's default, kept for v1 parity
FAMILY_V1 = "important_instructions"           # also the family of the enframe attack (English text)

Kind = Literal["v1_payload", "v1_embedded", "enframe_embedded", "native_payload", "native_embedded",
               "v1_benign", "native_benign"]

DEVANAGARI = re.compile("[ऀ-ॿ]")
SLOT = re.compile(r"\{[^{}\s]*\}")
HEADER = re.compile(r"^=== (\S+)(.*)$")


class FrozenItem(BaseModel):
    """One row of the frozen test set: the text a classifier scores, and why it is there."""

    item_id: str
    lang: str
    label: Literal["injection", "benign"]
    kind: Kind
    half: Literal["calibration", "test"]
    text: str
    goal: str | None = None            # attacker goal key, e.g. "banking/injection_task_3"
    family: str | None = None          # template family (Phase 2 may not train on it)
    carrier: str | None = None         # benign item the payload was spliced into
    tool_name: str = ""
    adversarial_benign: bool = False
    source: str = ""
    verified_by: str | None = None
    verified_at: str | None = None


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_sha(path: Path) -> str:
    """SHA-256 of a file with CRLF folded to LF: a Windows checkout must hash like CI's."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def half(key: str) -> Literal["calibration", "test"]:
    return "calibration" if is_fit(key, SALT) else "test"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))   # LF on every OS


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def provenance_index() -> dict[tuple[str, str], list[ProvenanceRecord]]:
    """Every provenance row, grouped by (item_id, lang), in file order (last = current)."""
    idx: dict[tuple[str, str], list[ProvenanceRecord]] = defaultdict(list)
    if PROVENANCE.exists():
        for line in PROVENANCE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = ProvenanceRecord.model_validate_json(line)
                idx[(r.item_id, r.lang)].append(r)
    return idx


def note_fields(notes: str) -> dict[str, str]:
    """'review: edited; sha=ab12; hi_sha=cd34' -> {'sha': 'ab12', 'hi_sha': 'cd34'}."""
    return dict(p.strip().split("=", 1) for p in notes.split(";") if "=" in p)


@dataclass
class Status:
    state: Literal["verified", "unverified", "changed", "stale", "not imported", "invalid"]
    verified_by: str | None = None
    verified_at: str | None = None


# --------------------------------------------------------------------------
# benign items and their review
# --------------------------------------------------------------------------

def load_benign(lang: str) -> list[BenignItem]:
    path = BENIGN / f"{lang}.jsonl"
    return [BenignItem.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def benign_status(lang: str, idx: dict | None = None) -> dict[str, Status]:
    idx = provenance_index() if idx is None else idx
    hi = {b.item_id: b for b in load_benign("hi")} if lang == "hi_latn" and (BENIGN / "hi.jsonl").exists() else {}
    out: dict[str, Status] = {}
    for b in load_benign(lang):
        if lang == "en" and b.source.startswith("agentdojo/"):
            out[b.item_id] = Status("verified", "AgentDojo")   # AgentDojo's own text, as its template
            continue
        rows = idx.get((f"benign/{b.item_id}", lang), [])
        last = rows[-1] if rows else None
        if last is None or not last.verified_by:
            out[b.item_id] = Status("unverified")
            continue
        f = note_fields(last.notes)
        if f.get("sha") != sha(b.content):
            out[b.item_id] = Status("changed")
        elif b.item_id in hi and f.get("hi_sha") != sha(hi[b.item_id].content):
            out[b.item_id] = Status("stale")
        else:
            out[b.item_id] = Status("verified", last.verified_by, last.verified_at)
    return out


REVIEW_FIELDS = ["item_id", "adversarial", "english", "text", "verified_by", "notes"]
SAVE_AS = "in Excel: File > Save As > 'CSV UTF-8 (Comma delimited)'"


def _to_sheet(text: str) -> str:
    """Excel reads a cell starting with = + - @ as a formula and saves '#NAME?' back. A leading
    apostrophe makes it plain text in the sheet; `_from_sheet` takes it off again."""
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def _from_sheet(text: str, old: str) -> str:
    return text[1:] if text.startswith("'") and not old.startswith("'") else text


def review_path(lang: str) -> Path:
    return REVIEW / f"testset_{lang}.csv"


def review_export(lang: str, *, everything: bool = False) -> Path:
    """The review sheet: by default only items not yet verified (unverified, changed, stale)."""
    status = benign_status(lang)
    en = {b.item_id: b.content for b in load_benign("en")} if lang != "en" else {}
    path = review_path(lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig: Excel shows Devanagari
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for b in load_benign(lang):
            st = status[b.item_id]
            if everything or st.state != "verified":
                w.writerow({"item_id": b.item_id, "adversarial": "yes" if b.adversarial_benign else "",
                            "english": _to_sheet(en.get(b.item_id, "")), "text": _to_sheet(b.content),
                            "verified_by": "",
                            "notes": "" if st.state in ("verified", "unverified") else st.state})
    return path


def review_import(lang: str) -> tuple[int, int]:
    """Signed rows back into corpus/benign/{lang}.jsonl, one provenance row each. All or nothing."""
    items = load_benign(lang)
    by_id = {b.item_id: b for b in items}
    en = {b.item_id: b for b in load_benign("en")}
    idx = provenance_index()
    hi = {b.item_id: b for b in load_benign("hi")} if lang == "hi_latn" else {}
    try:
        with review_path(lang).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        raise SystemExit(f"{review_path(lang)} was not saved as UTF-8; nothing imported. Save it again, "
                         f"{SAVE_AS}, and re-run the import.") from None

    problems, unsigned, updates = [], [], []
    for r in rows:
        iid, who = r["item_id"], (r.get("verified_by") or "").strip()
        if iid not in by_id:
            problems.append(f"{iid}: unknown item id")
            continue
        old = by_id[iid].content
        text = _from_sheet(r["text"].replace("\r\n", "\n").strip("\n"), old)
        if not who:
            if text != old:
                unsigned.append(iid)
            continue
        if not text.strip():
            problems.append(f"{iid}: empty text")
            continue
        if text != old:
            lost_chars = {c for c in old if ord(c) > 127} - set(text)
            if lost_chars and text.count("?") > old.count("?"):
                problems.append(f"{iid}: non-English characters came back as '?': the sheet was saved "
                                f"without UTF-8 ({SAVE_AS})")
                continue
            ref = en[iid].content if iid in en else old
            lost = sorted(e for e in entities(ref) if e not in text)
            if lost:
                problems.append(f"{iid}: entities lost in the edit {lost}")
                continue
        updates.append((iid, text, who, r.get("notes", "").strip()))
    if problems:
        raise SystemExit(f"refusing to import {review_path(lang)}:\n  " + "\n  ".join(problems))

    edited = 0
    stamp = now()
    for iid, text, who, note in updates:
        changed = text != by_id[iid].content
        edited += changed
        by_id[iid] = by_id[iid].model_copy(update={"content": text})
        prev = idx.get((f"benign/{iid}", lang), [])
        if changed:
            method = "human" if lang == "en" else "machine+human"
        else:
            method = prev[-1].method if prev else ("source" if lang == "en" else
                                                   "transliterated" if lang == "hi_latn" else "machine")
        fields = f"review: {'edited' if changed else 'unchanged'}; sha={sha(text)}"
        if iid in hi:
            fields += f"; hi_sha={sha(hi[iid].content)}"
        record(f"benign/{iid}", lang, method, source_item_id=f"benign/{iid}@en", verified_by=who,
               verified_at=stamp, notes=fields + (f"; note: {note}" if note else ""), path=PROVENANCE)
    _write(BENIGN / f"{lang}.jsonl", "".join(by_id[b.item_id].model_dump_json() + "\n" for b in items))
    if unsigned:
        print(f"note: {len(unsigned)} rows had edited text but no verified_by; left unchanged: "
              + ", ".join(unsigned[:10]) + (" ..." if len(unsigned) > 10 else ""))
    if lang == "hi" and edited:
        print(f"{edited} Hindi items edited: their hi_latn copies are now stale; re-read them "
              "(python -m corpus.testset review export hi_latn).")
    return edited, len(updates)


# --------------------------------------------------------------------------
# native items, written by a person in the language
# --------------------------------------------------------------------------

@dataclass
class NativeEntry:
    id: str
    kind: Literal["attack", "benign"]
    text: str
    line: int
    meta: dict[str, str] = field(default_factory=dict)


def native_paths(lang: str) -> dict[str, Path]:
    return {"attack": NATIVE / f"{lang}_attacks.txt", "benign": NATIVE / f"{lang}_benign.txt"}


def parse_native(text: str, *, kind: Literal["attack", "benign"], lang: str) -> tuple[list[NativeEntry], list[str]]:
    """`=== native_hi_t01 | family: bank_alert` + text lines, repeated. Returns entries and problems."""
    entries: list[NativeEntry] = []
    problems: list[str] = []
    cur: NativeEntry | None = None
    buf: list[str] = []

    def close() -> None:
        if cur is not None:
            cur.text = "\n".join(buf).strip("\n")
            entries.append(cur)

    for n, line in enumerate(text.splitlines(), 1):
        m = HEADER.match(line)
        if m:
            close()
            meta = {}
            for part in m.group(2).split("|"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    meta[k.strip()] = v.strip()
            cur, buf = NativeEntry(m.group(1), kind, "", n, meta), []
        elif cur is None:
            if line.strip():
                problems.append(f"line {n}: text before the first '=== id' header")
        else:
            buf.append(line)
    close()

    id_re = re.compile(rf"^native_{re.escape(lang)}_[A-Za-z0-9-]+$")
    seen: dict[str, int] = {}
    texts: dict[str, str] = {}
    for e in entries:
        where = f"{e.id} (line {e.line})"
        if not id_re.match(e.id):
            problems.append(f"{where}: id must look like native_{lang}_t01 (letters/digits after native_{lang}_)")
        if e.id in seen:
            problems.append(f"{where}: id appears twice (first at line {seen[e.id]})")
        seen.setdefault(e.id, e.line)
        if not e.text.strip():
            problems.append(f"{where}: no text")
            continue
        if e.text in texts:
            problems.append(f"{where}: same text as {texts[e.text]}")
        texts.setdefault(e.text, e.id)
        if kind == "attack":
            if not e.meta.get("family"):
                problems.append(f"{where}: header needs '| family: <name>'")
            slots = SLOT.findall(e.text)
            if slots.count("{goal}") != 1:
                problems.append(f"{where}: needs exactly one {{goal}} slot (found {slots.count('{goal}')})")
            unknown = sorted({s for s in slots if s not in ("{goal}", "{user}")})
            if unknown:
                problems.append(f"{where}: unknown placeholder {', '.join(unknown)}; only {{goal}} and {{user}}")
        elif e.meta.get("tool") not in TOOLS:
            problems.append(f"{where}: header needs '| tool: <one of {', '.join(TOOLS)}>'")
        has_dev = bool(DEVANAGARI.search(e.text))
        if lang == "hi" and not has_dev:
            problems.append(f"{where}: no Devanagari; hi items are written in Devanagari")
        if lang in ("hi_latn", "mixed") and has_dev:
            problems.append(f"{where}: contains Devanagari; {lang} items are written in Latin script")
    return entries, problems


def native_entries(lang: str) -> tuple[list[NativeEntry], list[NativeEntry], list[str]]:
    paths = native_paths(lang)
    out: dict[str, list[NativeEntry]] = {}
    problems: list[str] = []
    for kind, p in paths.items():
        if not p.exists():
            out[kind] = []
            problems.append(f"{p} does not exist yet")
            continue
        entries, probs = parse_native(p.read_text(encoding="utf-8"), kind=kind, lang=lang)
        out[kind] = entries
        problems += [f"{p.name}: {x}" for x in probs]
    clash = {e.id for e in out["attack"]} & {e.id for e in out["benign"]}
    problems += [f"{i}: used in both the attacks and the benign file" for i in sorted(clash)]
    return out["attack"], out["benign"], problems


def native_status(lang: str, entries: list[NativeEntry], idx: dict | None = None) -> dict[str, Status]:
    idx = provenance_index() if idx is None else idx
    out: dict[str, Status] = {}
    for e in entries:
        rows = idx.get((f"native/{e.id}", lang), [])
        last = rows[-1] if rows else None
        if last is None or not last.verified_by:
            out[e.id] = Status("not imported")
        elif note_fields(last.notes).get("sha") != sha(e.text):
            out[e.id] = Status("changed")
        else:
            out[e.id] = Status("verified", last.verified_by, last.verified_at)
    return out


def native_import(lang: str, *, author: str) -> int:
    """Stamp the author as verifier of every native item whose text is new or changed."""
    attacks, benign, problems = native_entries(lang)
    if problems:
        raise SystemExit(f"{lang} native files have problems; nothing imported:\n  " + "\n  ".join(problems))
    status = native_status(lang, attacks + benign)
    stamp = now()
    n = 0
    for e in attacks + benign:
        if status[e.id].state == "verified":
            continue
        what = f"family={e.meta['family']}" if e.kind == "attack" else f"tool={e.meta['tool']}"
        record(f"native/{e.id}", lang, "human", verified_by=author, verified_at=stamp,
               notes=f"native {e.kind}: written in {lang}; sha={sha(e.text)}; {what}", path=PROVENANCE)
        n += 1
    return n


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def load_cases() -> list[InjectionCase]:
    return [InjectionCase.model_validate_json(l)
            for l in INJECTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]


def _who(*parts: tuple[str | None, str | None]) -> tuple[str | None, str | None]:
    """Combine (verified_by, verified_at) of the parts of an item: unverified if any part is."""
    if any(by is None for by, _ in parts):
        return None, None
    names = ", ".join(sorted({by for by, _ in parts}))
    stamps = [at for _, at in parts if at]
    return names, max(stamps) if stamps else None


def collect(lang: str, *, pilot: bool = False) -> tuple[list[FrozenItem], list[str]]:
    """Every test item for `lang`, and every reason it could not be frozen yet."""
    from corpus.goals import load_goals, localize_cases
    from scripts.attack_scores import _h, embed, goal_key

    problems: list[str] = []
    idx = provenance_index()
    items: list[FrozenItem] = []

    benign = load_benign(lang)
    status = benign_status(lang, idx)
    bstate = {}
    for b in benign:
        st = status[b.item_id]
        bstate[b.item_id] = (st.verified_by, st.verified_at)
        if st.state != "verified":
            problems.append(f"{lang} benign {b.item_id}: {st.state}")
        items.append(FrozenItem(item_id=b.item_id, lang=lang, label="benign", kind="v1_benign",
                                half=half(b.item_id), text=b.content, tool_name=b.tool_name,
                                adversarial_benign=b.adversarial_benign, source=b.source,
                                verified_by=st.verified_by, verified_at=st.verified_at))
    # v1's carrier rule (scripts/attack_scores.positives): ordinary AgentDojo items, file order
    carriers = [b for b in benign if not b.adversarial_benign and b.source.startswith("agentdojo/")]

    en_cases = load_cases()
    if lang == "en":
        cases, goal_who = en_cases, defaultdict(lambda: ("AgentDojo", None))
        template_who = ("AgentDojo", None)
    else:
        cases = localize_cases(en_cases, lang, allow_unverified=True)
        goal_who = {f"{d}/{t}": (g.verified_by, g.verified_at) for (d, t), g in load_goals(lang).items()}
        template_who = (TEMPLATE_STATUS.get(lang, {}).get("verified_by"), None)
        if template_who[0] is None:
            problems.append(f"{lang}: the {DEFAULT_ATTACK} template is not verified")
        for k, (by, _) in sorted(goal_who.items()):
            if by is None:
                problems.append(f"{lang} goal {k}: unverified")

    goal_text: dict[str, str] = {}
    seen: set[str] = set()
    for c in cases:
        g = goal_key(c)
        goal_text[g] = c.attacker_goal_nl
        payload = render(c.attacker_goal_nl, attack=DEFAULT_ATTACK, lang=lang, allow_unverified=True)
        carrier = carriers[_h(c.case_id) % len(carriers)]
        if g not in seen:
            seen.add(g)
            by, at = _who(goal_who[g], template_who)
            items.append(FrozenItem(item_id=f"payload/{g}", lang=lang, label="injection", kind="v1_payload",
                                    half=half(g), text=payload, goal=g, family=FAMILY_V1,
                                    source=f"agentdojo/{c.source_case_id}", verified_by=by, verified_at=at))
        by, at = _who(goal_who[g], template_who, bstate[carrier.item_id])
        items.append(FrozenItem(item_id=f"embedded/{c.case_id}", lang=lang, label="injection",
                                kind="v1_embedded", half=half(g), text=embed(carrier.content, payload),
                                goal=g, family=FAMILY_V1, carrier=carrier.item_id, tool_name=carrier.tool_name,
                                source=f"agentdojo/{c.source_case_id}", verified_by=by, verified_at=at))
        if lang != "en":
            framed = render(c.attacker_goal_nl, attack=ENFRAME, lang=lang)
            by, at = _who(goal_who[g], ("AgentDojo", None), bstate[carrier.item_id])
            items.append(FrozenItem(item_id=f"enframe/{c.case_id}", lang=lang, label="injection",
                                    kind="enframe_embedded", half=half(g), text=embed(carrier.content, framed),
                                    goal=g, family=FAMILY_V1, carrier=carrier.item_id,
                                    tool_name=carrier.tool_name, source=f"agentdojo/{c.source_case_id}",
                                    verified_by=by, verified_at=at))

    if lang in NATIVE_LANGS:
        attacks, nbenign, probs = native_entries(lang)
        problems += [f"{lang} native: {p}" for p in probs]
        nstatus = native_status(lang, attacks + nbenign, idx)
        for e in attacks + nbenign:
            if nstatus[e.id].state != "verified":
                problems.append(f"{lang} native {e.id}: {nstatus[e.id].state}")
        keys = sorted(k for k in goal_text if pilot or goal_who[k][0] is not None)
        for e in attacks:
            if not keys or "family" not in e.meta:
                continue
            g = keys[_h(e.id) % len(keys)]
            text = e.text.replace("{goal}", goal_text[g]).replace("{user}", USER_NAME)
            carrier = carriers[_h(e.id) % len(carriers)]
            author = (nstatus[e.id].verified_by, nstatus[e.id].verified_at)
            family = f"native/{lang}/{e.meta['family']}"
            by, at = _who(author, goal_who[g])
            items.append(FrozenItem(item_id=f"native_payload/{e.id}", lang=lang, label="injection",
                                    kind="native_payload", half=half(g), text=text, goal=g, family=family,
                                    source="native", verified_by=by, verified_at=at))
            by, at = _who(author, goal_who[g], bstate[carrier.item_id])
            items.append(FrozenItem(item_id=f"native_embedded/{e.id}", lang=lang, label="injection",
                                    kind="native_embedded", half=half(g), text=embed(carrier.content, text),
                                    goal=g, family=family, carrier=carrier.item_id, tool_name=carrier.tool_name,
                                    source="native", verified_by=by, verified_at=at))
        for e in nbenign:
            st = nstatus[e.id]
            items.append(FrozenItem(item_id=e.id, lang=lang, label="benign", kind="native_benign",
                                    half=half(e.id), text=e.text, tool_name=f"native.{e.meta.get('tool', '')}",
                                    adversarial_benign=True, source="native",
                                    verified_by=st.verified_by, verified_at=st.verified_at))

    items, missing = anonymize(items)
    problems += [f"reviewer {n!r}: no code in {REVIEWERS} (add one, e.g. \"R2\")" for n in sorted(missing)]
    return sorted(items, key=lambda i: i.item_id), problems


def reviewer_codes() -> dict[str, str]:
    codes = {"AgentDojo": "AgentDojo"}   # the benchmark's own text: an institution, not a reviewer
    if REVIEWERS.exists():
        codes.update(json.loads(REVIEWERS.read_text(encoding="utf-8")))
    return codes


def anonymize(items: list[FrozenItem]) -> tuple[list[FrozenItem], set[str]]:
    """Replace reviewer names with their codes; return the names that have none."""
    codes, missing, out = reviewer_codes(), set(), []
    for i in items:
        if i.verified_by:
            names = i.verified_by.split(", ")
            missing |= {n for n in names if n not in codes}
            i = i.model_copy(update={"verified_by": ", ".join(sorted({codes.get(n, n) for n in names}))})
        out.append(i)
    return out, missing


def build(langs: list[str] | tuple[str, ...] = LANGS_V2, *, pilot: bool = False) -> dict:
    """Write the frozen set (or the pilot copy) and its manifest. Strict unless `pilot`."""
    collected = {lang: collect(lang, pilot=pilot) for lang in langs}
    problems = [p for _, probs in collected.values() for p in probs]
    if problems and not pilot:
        kinds = Counter(p.rsplit(": ", 1)[-1] if ": " in p else p for p in problems)
        summary = ", ".join(f"{n} {k}" for k, n in kinds.most_common(6))
        raise SystemExit(f"refusing to build the frozen test set: {len(problems)} problems ({summary})\n  "
                         + "\n  ".join(problems[:25]) + ("\n  ..." if len(problems) > 25 else "")
                         + "\nRun `python -m corpus.testset status`, or `build --pilot` to see the shape.")

    out = PILOT if pilot else OUT
    counts: dict[str, dict[str, dict[str, int]]] = {}
    goals: dict[str, set[str]] = {"calibration": set(), "test": set()}
    families: set[str] = set()
    files: dict[str, str] = {}
    for lang, (items, _) in collected.items():
        path = out / f"{lang}.jsonl"
        _write(path, "".join(i.model_dump_json() + "\n" for i in items))
        files[path.name] = file_sha(path)
        c: dict[str, dict[str, int]] = defaultdict(lambda: {"calibration": 0, "test": 0})
        for i in items:
            c[i.kind][i.half] += 1
            if i.goal:
                goals[i.half].add(i.goal)
            if i.family:
                families.add(i.family)
        counts[lang] = dict(sorted(c.items()))
    manifest = {"version": 1, "pilot": pilot, "salt": SALT, "langs": list(langs), "counts": counts,
                "goals": {h: sorted(v) for h, v in goals.items()}, "families": sorted(families),
                "files": files, "problems_at_build": len(problems)}
    _write(out / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return manifest


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def print_status(langs) -> None:
    idx = provenance_index()
    print(f"{'lang':<9}{'benign':>7}{'verified':>10}{'to read':>9}{'stale':>7}{'changed':>9} | "
          f"{'native templates':>17}{'native benign':>15}{'signed':>8}{'problems':>10}")
    for lang in langs:
        st = Counter(s.state for s in benign_status(lang, idx).values())
        line = (f"{lang:<9}{sum(st.values()):>7}{st['verified']:>10}{st['unverified']:>9}"
                f"{st['stale']:>7}{st['changed']:>9} | ")
        if lang in NATIVE_LANGS:
            attacks, nb, probs = native_entries(lang)
            ns = Counter(s.state for s in native_status(lang, attacks + nb, idx).values())
            line += f"{len(attacks):>17}{len(nb):>15}{ns['verified']:>8}{len(probs):>10}"
        else:
            line += f"{'-':>17}{'-':>15}{'-':>8}{'-':>10}"
        print(line)
    print("\nGoal for G1 (docs/v2-plan.md): every benign item verified; hi, hi_latn, mixed: 50 native "
          "templates + 50 native benign texts each, all signed; then `build`.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m corpus.testset", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--langs", nargs="+", default=list(LANGS_V2))
    r = sub.add_parser("review")
    r.add_argument("action", choices=["export", "import"])
    r.add_argument("lang")
    r.add_argument("--all", action="store_true", help="export every item, not only those left to read")
    n = sub.add_parser("native")
    n.add_argument("action", choices=["check", "import"])
    n.add_argument("lang", choices=NATIVE_LANGS)
    n.add_argument("--author")
    b = sub.add_parser("build")
    b.add_argument("--langs", nargs="+", default=list(LANGS_V2))
    b.add_argument("--pilot", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "status":
        print_status(a.langs)
    elif a.cmd == "review" and a.action == "export":
        path = review_export(a.lang, everything=a.all)
        with path.open(encoding="utf-8-sig") as f:
            rows = sum(1 for _ in csv.DictReader(f))
        print(f"wrote {rows} items to {path}: fix `text` where needed, put your name in `verified_by`, "
              f"save as CSV, then `python -m corpus.testset review import {a.lang}`")
    elif a.cmd == "review":
        edited, signed = review_import(a.lang)
        print(f"{a.lang}: {signed} items signed ({edited} edited)")
    elif a.cmd == "native" and a.action == "check":
        attacks, nb, probs = native_entries(a.lang)
        print(f"{a.lang}: {len(attacks)} templates, {len(nb)} benign texts, {len(probs)} problems")
        for p in probs:
            print("  " + p)
        return 1 if probs else 0
    elif a.cmd == "native":
        if not a.author:
            ap.error("native import needs --author \"Your Name\"")
        print(f"{a.lang}: {native_import(a.lang, author=a.author)} native items signed by {a.author}")
    else:
        m = build(a.langs, pilot=a.pilot)
        where = PILOT if a.pilot else OUT
        print(f"wrote {', '.join(m['files'])} + manifest.json to {where}")
        for lang, kinds in m["counts"].items():
            print(f"  {lang:<8}" + "  ".join(f"{k} {v['calibration']}/{v['test']}" for k, v in kinds.items()))
        print(f"  goals: {len(m['goals']['calibration'])} calibration / {len(m['goals']['test'])} test; "
              f"families: {len(m['families'])}")
        if a.pilot and m["problems_at_build"]:
            print(f"  pilot: {m['problems_at_build']} items not yet verified (see `status`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
