"""H5: does a defense cost more utility when the user writes in another language?

    python -m analysis.h5 --lang hi --defenses sandwich spotlight camel_lite allowlist protectai_deberta honeytool

Unit of analysis: the user task (clean passes copy one run to sibling cases, so
cases would count each run several times). Tasks: `corpus.user_tasks.h5_tasks`,
the ones a correct run passes in any language; English numbers are restricted
to the same tasks, from the English clean passes already on disk.

For each defense D:

    cost(L, D) = utility(L, undefended) - utility(L, D)
    H5 gap     = cost(hi, D) - cost(en, D)     (> 0: the defense costs more in hi)

with a paired bootstrap over tasks (one draw keeps a task's four outcomes
together) and the share of tasks where the defense blocked something.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from analysis.stats import mcnemar_exact
from harness.scoring import CaseResult

GRID = Path("analysis/out/grid")


def clean_file(defense: str, lang: str) -> Path:
    if lang == "en":
        return GRID / f"clean__{defense}.json"
    return GRID / f"clean__{defense}__{lang}__usertask.json"


def per_task(defense: str, lang: str, tasks: list[str]) -> dict[str, CaseResult] | None:
    """task -> its clean run (errored runs dropped). None if the pass is not on disk."""
    f = clean_file(defense, lang)
    if not f.exists():
        return None
    out: dict[str, CaseResult] = {}
    want = set(tasks)
    for row in json.loads(f.read_text(encoding="utf-8"))["clean"]:
        r = CaseResult.model_validate(row)
        k = _key(r.case_id)
        if k in want and k not in out and not r.errored:
            out[k] = r
    return out


def _key(case_id: str) -> str:
    """'banking__user_task_3__injection_task_1' -> 'banking/user_task_3'."""
    domain, ut = case_id.split("__")[:2]
    return f"{domain}/{ut}"


def _rate(rs: dict[str, CaseResult], keys) -> float:
    return sum(rs[k].user_task_success for k in keys) / len(keys)


def h5_row(defense: str, lang: str, tasks: list[str], *, iters: int = 2000,
           seed: int = 20260916) -> dict | None:
    runs = {(l, d): per_task(d, l, tasks) for l in ("en", lang) for d in ("undefended", defense)}
    if any(v is None for v in runs.values()):
        return None
    keys = sorted(set(tasks).intersection(*[set(v) for v in runs.values()]))
    if not keys:
        return None
    u = {ld: _rate(rs, keys) for ld, rs in runs.items()}
    cost_en = u[("en", "undefended")] - u[("en", defense)]
    cost_l = u[(lang, "undefended")] - u[(lang, defense)]

    rng = random.Random(seed)
    gaps = []
    for _ in range(iters):
        draw = [rng.choice(keys) for _ in keys]
        g = ((_rate(runs[(lang, "undefended")], draw) - _rate(runs[(lang, defense)], draw))
             - (_rate(runs[("en", "undefended")], draw) - _rate(runs[("en", defense)], draw)))
        gaps.append(g)
    gaps.sort()

    # paired McNemar: the defense's utility, en vs this language, task by task
    en_d, l_d = runs[("en", defense)], runs[(lang, defense)]
    b = sum(en_d[k].user_task_success and not l_d[k].user_task_success for k in keys)
    c = sum(l_d[k].user_task_success and not en_d[k].user_task_success for k in keys)
    return {
        "defense": defense, "lang": lang, "n_tasks": len(keys),
        "utility": {f"{l}/{d}": round(v, 4) for (l, d), v in u.items()},
        "cost_en": round(cost_en, 4), "cost_lang": round(cost_l, 4),
        "h5_gap": round(cost_l - cost_en, 4),
        "h5_gap_ci": (round(gaps[int(0.025 * iters)], 4), round(gaps[int(0.975 * iters) - 1], 4)),
        "blocked_en": round(sum(en_d[k].blocked for k in keys) / len(keys), 4),
        "blocked_lang": round(sum(l_d[k].blocked for k in keys) / len(keys), 4),
        "mcnemar_en_vs_lang": {"only_en": b, "only_lang": c, "p": mcnemar_exact(b, c)},
    }


def table(lang: str, defenses: list[str], tasks: list[str]) -> str:
    lines = [f"H5, {lang} user instructions vs English, {len(tasks)} language-fair tasks "
             f"(cost = undefended utility - defended utility; gap = cost({lang}) - cost(en), paired bootstrap 95% CI)",
             f"{'defense':<20}{'util en':>9}{'util ' + lang:>9}{'cost en':>9}{'cost ' + lang:>9}"
             f"{'gap':>8}  {'95% CI':<17}{'blocked en/' + lang:>16}"]
    for d in defenses:
        if d == "undefended":
            continue
        r = h5_row(d, lang, tasks)
        if r is None:
            lines.append(f"{d:<20}  (clean pass missing in en or {lang})")
            continue
        u = r["utility"]
        lo, hi = r["h5_gap_ci"]
        lines.append(f"{d:<20}{u[f'en/{d}']:>9.3f}{u[f'{lang}/{d}']:>9.3f}{r['cost_en']:>+9.3f}"
                     f"{r['cost_lang']:>+9.3f}{r['h5_gap']:>+8.3f}  [{lo:+.3f}, {hi:+.3f}]"
                     f"{r['blocked_en']:>9.2f}/{r['blocked_lang']:.2f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--defenses", nargs="+", default=["sandwich", "spotlight", "camel_lite", "allowlist",
                                                      "protectai_deberta", "honeytool"])
    args = ap.parse_args()
    from corpus.user_tasks import _en_cases, h5_tasks

    print(table(args.lang, args.defenses, h5_tasks(_en_cases())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
