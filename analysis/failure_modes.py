"""Why did a capability-probe goal fail? One label per failed goal.

C(L) says *how much* capability drops off-English; this says *why*, which is
what decides whether a drop is the model's or the corpus's:

    tool_call_as_text  the model chose an action but emitted the call as plain
                       JSON instead of a tool call, so nothing ran
    wrong_language     the reply is mostly in a script the condition does not
                       use (Qwen drifting into Thai or Chinese on Hindi input)
    no_action          no tool call at all: it asked for details, refused, or
                       answered in prose
    wrong_action       it called tools, but the checker says the goal was not met
                       (wrong recipient, wrong item, incomplete)
    errored            the run failed for infrastructure reasons

Only `wrong_action` and `no_action` can plausibly come from a bad translation;
the first two are model behaviour. Heuristic labels -- read the cases before
quoting a number.

    python -m analysis.failure_modes analysis/out/14b
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Letters we expect in each condition's replies. Latin is always allowed: models
# answer in English, and identifiers (IBANs, emails) are Latin anyway.
EXPECTED = {
    "en": {"latin"},
    "hi": {"latin", "devanagari"},
    "hi_latn": {"latin", "devanagari"},
    "bn": {"latin", "bengali"},
    "mixed": {"latin", "devanagari"},
}
SCRIPTS = {
    "latin": re.compile(r"[A-Za-z]"),
    "devanagari": re.compile(r"[ऀ-ॿ]"),
    "bengali": re.compile(r"[ঀ-৿]"),
    "thai": re.compile(r"[฀-๿]"),
    "cjk": re.compile(r"[぀-ヿ一-鿿가-힯]"),
    "arabic": re.compile(r"[؀-ۿ]"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]"),
}
# One call `{"name": ...}` or a list of them `[{"name": ...}, ...]`, optionally fenced.
TOOL_JSON = re.compile(r'^\s*(```(json)?\s*)?\[?\s*\{\s*"name"\s*:', re.DOTALL)


def foreign_share(text: str, lang: str) -> float:
    """Share of letters in scripts this condition never uses.

    A share, not a majority: a Thai reply quoting an English URL and channel
    names is still a Thai reply.
    """
    counts = {name: len(rx.findall(text)) for name, rx in SCRIPTS.items()}
    total = sum(counts.values())
    if total < 20:
        return 0.0
    allowed = EXPECTED.get(lang, {"latin"})
    return sum(n for name, n in counts.items() if name not in allowed) / total


def label(goal: dict, lang: str) -> str:
    if goal.get("errored"):
        return "errored"
    text = goal.get("final_text") or ""
    if not goal.get("tool_calls") and TOOL_JSON.match(text):
        return "tool_call_as_text"
    if foreign_share(text, lang) > 0.3:
        return "wrong_language"
    if not goal.get("tool_calls"):
        return "no_action"
    return "wrong_action"


MODES = ("tool_call_as_text", "wrong_language", "no_action", "wrong_action", "errored")


def summarise(path: Path) -> tuple[str, Counter, int, list[tuple[str, str]]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    lang = d["lang"]
    goals = d["goals"]
    modes: Counter = Counter()
    rows = []
    for g in goals:
        if g["task_succeeded"]:
            continue
        m = label(g, lang)
        modes[m] += 1
        rows.append((f"{g['domain']}/{g['injection_task']}", m))
    return lang, modes, len(goals), rows


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path("analysis/out")
    files = sorted(root.glob("phase2_probe_*.json"))
    files = [f for f in files if not f.name.endswith("_fake.json")]
    if not files:
        print(f"no probe files in {root}")
        return 1
    order = {"en": 0, "hi": 1, "hi_latn": 2, "bn": 3, "mixed": 4}
    results = sorted((summarise(f) for f in files), key=lambda r: order.get(r[0], 9))

    print(f"failed goals by cause ({root})\n")
    print(f"{'lang':<8} {'failed':>7}  " + "  ".join(f"{m:>17}" for m in MODES))
    for lang, modes, n, _ in results:
        failed = sum(modes.values())
        print(f"{lang:<8} {failed:>3}/{n:<3}  " + "  ".join(f"{modes[m]:>17}" for m in MODES))

    # The column that matters: goals English could do and this language could not.
    # Failures shared with English (the hard workspace goals) say nothing about language.
    en = next((r for r in results if r[0] == "en"), None)
    if en is not None:
        en_failed = {goal for goal, _ in en[3]}
        print(f"\nonly the goals lost relative to en (en failed {len(en_failed)})\n")
        print(f"{'lang':<8} {'lost':>5}  " + "  ".join(f"{m:>17}" for m in MODES))
        for lang, _, _, rows in results:
            if lang == "en":
                continue
            lost = Counter(m for goal, m in rows if goal not in en_failed)
            print(f"{lang:<8} {sum(lost.values()):>5}  " + "  ".join(f"{lost[m]:>17}" for m in MODES))
    else:
        print("\n(no phase2_probe_en.json here: cannot separate language-specific losses "
              "from goals that fail in English too)")

    if "-v" in argv:
        for lang, _, _, rows in results:
            print(f"\n{lang}:")
            for goal, m in sorted(rows):
                print(f"  {goal:<32} {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
