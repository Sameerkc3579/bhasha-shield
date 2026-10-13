"""Phase 5: the grid -- every (language, defense) cell, checkpointed.

    python -m scripts.phase5_grid --langs en --defenses undefended spotlight sandwich \
        hierarchy allowlist camel_lite honeytool --workers 8          # needs vLLM up

One cell = the 150 cases in one language under one defense, attack injected.
Each finished cell is written to `analysis/out/grid/` before the next starts, so
a killed Colab session loses at most the cell that was running; re-running the
same command skips every cell already on disk (`--rerun` to force) -- except
results lost to the server (`needs_rerun`), which run again and are merged in.
A cell that still has such results exits 3, so the notebook can restart vLLM.
`--status` reports each cell as done / repair / missing without a server.

Utility under each defense comes from a separate **clean pass** (no injection).
In AgentDojo's English environment there is nothing language-specific in a
clean run, so it runs once per defense and every language's frontier point
uses it. With `--localized-env` (C3, `corpus/environments.py`) the world the
agent acts in is the language's own, so the clean pass runs per language, and
those cells, clean passes and the probe they need (`_locenv`) are stored apart
from the English-environment ones. Every file records which environment it was.

Achievability (C1) comes from the Phase 2 probe for the same language and served
model, OR'd with the undefended cell's own successes (an attack that worked was
achievable; `harness.capability.with_existence_proof`). It is **re-applied at
summary time** from the current probe file, so every cell in a language is scored
on one scorable set even if the probe was re-run between cells (2026-10-07: the
undefended and sandwich cells had been stamped from different probe runs, 79 vs 87
scorable, and the efficacy mixed the two denominators).

`--scorable-only` runs defended cells on the scorable set S(L) only. Off-English
most cases are outside S(L) (C(hi) ~ 0.27), and uASR and efficacy never look at
them, so this cuts a defended cell's attack pass by ~3-4x. The undefended cell
always runs every case: raw ASR, and the existence proof, need all of them. A
scorable-only cell has no raw ASR; the file and the summary say so. Language
order: run language-by-language (the plan's rule), so a partial grid is a set of
complete languages.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from analysis.stats import compare_attack_success, efficacy_ci, efficacy_point
from corpus.attacks import DEFAULT_ATTACK, render
from corpus.schema import InjectionCase
from harness.cache import ResponseCache
from harness.capability import achievability_map, apply_achievability, with_existence_proof
from harness.scoring import CaseResult, asr, benign_utility, uasr
from harness.trace import RunTrace

OUT = Path("analysis/out/grid")
PROBES = Path("analysis/out")
CLASSIFIERS = ("promptguard2", "protectai_deberta", "xlmr_finetuned")


def load_cases(lang: str, allow_unverified: bool) -> list[InjectionCase]:
    en = [InjectionCase.model_validate_json(l)
          for l in Path("corpus/injections/en.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if lang == "en":
        return en
    path = Path(f"corpus/injections/{lang}.jsonl")
    if path.exists():
        return [InjectionCase.model_validate_json(l)
                for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    from corpus.goals import localize_cases

    return localize_cases(en, lang, allow_unverified=allow_unverified)


def load_probe(lang: str, model: str, localized: bool = False) -> tuple[dict, str | None]:
    """Achievability map for this language and model, and the file it came from.

    A localized-environment grid needs a probe run in the same environment:
    C(L) is a property of the language *and* the world the agent acts in.
    """
    loc = "_locenv" if localized and lang != "en" else ""
    for name in (f"phase2_probe_{lang}{loc}.json", f"phase2_probe_{lang}{loc}_draft.json"):
        f = PROBES / name
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("model") == model and not d.get("fake"):
                return achievability_map([CaseResult.model_validate(r) for r in d["probe"]]), str(f)
    return {}, None


def scorable_map(lang: str, model: str, localized: bool, attack: str) -> tuple[dict, str | None]:
    """Probe achievability for (lang, model) OR'd with the undefended cell's successes."""
    amap, probe_file = load_probe(lang, model, localized)
    base_f = cell_path(lang, "undefended", attack, localized)
    if base_f.exists():
        amap = with_existence_proof(amap, load_cell(base_f))
    return amap, probe_file


def load_cell(f: Path, key: str = "attack") -> list[CaseResult]:
    return [CaseResult.model_validate(r) for r in json.loads(f.read_text(encoding="utf-8"))[key]]


def build_defense(name: str, client, args):
    from defenses.registry import build

    from defenses.registry import needs_planner

    kw: dict = {}
    if name.split("+")[-1] in CLASSIFIERS:  # `translit+protectai_deberta` too
        kw.update(threshold=args.threshold, device=args.classifier_device)
    if needs_planner(name):
        kw["planner"] = client
    return build(name, **kw)


def run_cells(cases, defense, client, *, inject: bool, attack: str, workers: int, trace,
              allow_unverified: bool = False, localized_env: bool = False) -> list[CaseResult]:
    from harness.runner import AgentDojoRunner

    runner = AgentDojoRunner(client=client, defense=defense, trace=trace, localized_env=localized_env)

    def one(case: InjectionCase) -> CaseResult:
        payload = (render(case.attacker_goal_nl, attack=attack, lang=case.lang,
                          allow_unverified=allow_unverified) if inject else "")
        return runner.run_case(case, payload=payload)

    out: list[CaseResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, c): c for c in cases}
        for i, fut in enumerate(as_completed(futures), 1):
            c = futures[fut]
            try:
                out.append(fut.result())
            except Exception as e:  # a harness failure must not kill the cell
                out.append(CaseResult(case_id=c.case_id, lang=c.lang, defense=defense.name,
                                      model=client.model, errored=True,
                                      error=f"{type(e).__name__}: {e}"))
                print(f"    {c.case_id} raised {type(e).__name__}: {str(e)[:200]}")
            if i % 25 == 0:
                print(f"    [{i}/{len(cases)}]")
    return out


def user_task_key(case: InjectionCase) -> str:
    return (case.source_case_id or case.case_id).rsplit("/", 1)[0]


def run_clean_dedup(cases, defense, client, **kw) -> list[CaseResult]:
    """The clean (no-payload) pass, run once per distinct user task.

    With no injection, two cases that share a user task are the same run: same
    prompt, same environment, same checker. The 150-case subset has only 79
    distinct user tasks, so running all 150 spent ~45% of every clean pass on
    duplicates (2026-10-07: 132 min per English defense). Each representative's
    result is copied to its siblings, so utility keeps the per-case weighting of
    the earlier cells. Only `honeytool` sees the case id (its honeytoken); the
    sibling would get a different token string, which does not change what the
    run can do.
    """
    reps: dict[str, InjectionCase] = {}
    for c in cases:
        reps.setdefault(user_task_key(c), c)
    rep_ids = {c.case_id: k for k, c in reps.items()}
    ran = {rep_ids[r.case_id]: r for r in run_cells(list(reps.values()), defense, client, inject=False, **kw)}
    return [ran[user_task_key(c)].model_copy(update={"case_id": c.case_id, "attacker_task_success": False})
            for c in cases]


def cell_path(lang: str, defense: str, attack: str, localized: bool = False) -> Path:
    loc = "__locenv" if localized and lang != "en" else ""
    return OUT / f"{lang}__{defense}__{attack}{loc}.json"


def clean_path(defense: str, lang: str = "en", localized: bool = False, usertask: bool = False) -> Path:
    """English-environment clean passes are language-free; localized ones are not.

    `usertask`: the user's instruction itself is in `lang` (H5, corpus.user_tasks).
    """
    if usertask and lang != "en":
        return OUT / f"clean__{defense}__{lang}__usertask.json"
    if localized and lang != "en":
        return OUT / f"clean__{defense}__{lang}__locenv.json"
    return OUT / f"clean__{defense}.json"


# Failures of the serving stack, not of the agent. 2026-10-09: vLLM died at the end
# of `en spotlight` and every later cell "finished" in about a minute, all errored --
# and existing files read as done, so the next session would have skipped them.
INFRA_ERRORS = ("APIConnectionError", "APITimeoutError")


def needs_rerun(r: CaseResult) -> bool:
    """A result lost to the server: run it again. Files written before 2026-10-10
    kept no error text, so an errored result without one counts too -- once: the
    re-run records the reason, and a genuine failure (context overflow) then stays."""
    return r.errored and (not r.error or r.error.startswith(INFRA_ERRORS))


def lost(f: Path, key: str) -> set[str]:
    """Case ids in a cell or clean file that have to run again (none if no file)."""
    return {r.case_id for r in load_cell(f, key) if needs_rerun(r)} if f.exists() else set()


def cell_status(lang: str, defense: str, attack: str, localized: bool = False, usertask: bool = False) -> str:
    """'done', 'missing', or 'repair' (on disk, but results lost to the server)."""
    if usertask:  # utility only: there is no attack cell
        cp = clean_path(defense, lang, localized, usertask=True)
        return "missing" if not cp.exists() else "repair" if lost(cp, "clean") else "done"
    f, cp = cell_path(lang, defense, attack, localized), clean_path(defense, lang, localized)
    if not f.exists() or not cp.exists():
        return "missing"
    return "repair" if lost(f, "attack") or lost(cp, "clean") else "done"


def merge(old: list[CaseResult], new: list[CaseResult]) -> list[CaseResult]:
    """`old` with `new` in place of the same case ids, file order kept."""
    fresh = {r.case_id: r for r in new}
    seen = {r.case_id for r in old}
    return [fresh.get(r.case_id, r) for r in old] + [r for r in new if r.case_id not in seen]


def _fmt(r) -> str:
    return "undefined" if r.value is None else f"{r.value:.3f} ({r.k}/{r.n})"


def summarise_lang(lang: str, attack: str, defenses: list[str], localized: bool = False) -> None:
    base_f = cell_path(lang, "undefended", attack, localized)
    base_meta = json.loads(base_f.read_text(encoding="utf-8")) if base_f.exists() else None
    amap, probe_file = (scorable_map(lang, base_meta["model"], localized, attack)
                        if base_meta else ({}, None))
    base = apply_achievability(load_cell(base_f), amap) if base_meta else None
    print(f"\n--- {lang}: {attack}{' (localized environment)' if localized and lang != 'en' else ''} ---")
    print(f"scorable set: {probe_file or 'NO PROBE (all cases assumed achievable)'}"
          f"{' + undefended successes' if base_meta else ''}")
    print(f"{'defense':<20}{'ASR':>16}{'uASR':>16}{'utility':>16}  efficacy vs undefended")
    for d in defenses:
        f = cell_path(lang, d, attack, localized)
        if not f.exists():
            continue
        meta = json.loads(f.read_text(encoding="utf-8"))
        rs = apply_achievability(load_cell(f), amap) if base_meta else load_cell(f)
        cf = clean_path(d, lang, localized)
        util = (benign_utility([CaseResult.model_validate(r) for r in json.loads(cf.read_text(encoding="utf-8"))["clean"]])
                if cf.exists() else None)
        eff = ""
        if base is not None and d != "undefended":
            e = efficacy_point(rs, base)
            lo, hi = efficacy_ci(rs, base)
            p = compare_attack_success(base, rs)
            eff = ("undefined" if e is None else
                   f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}]  ({p})" if lo is not None else f"{e:+.2f}")
        raw = "n/a (S only)" if meta.get("scorable_only") else _fmt(asr(rs))
        print(f"{d:<20}{raw:>16}{_fmt(uasr(rs)):>16}{(_fmt(util) if util else '-'):>16}  {eff}")


def summarise_usertask(lang: str, defenses: list[str]) -> None:
    """H5 table: each defense's utility cost with `lang` user instructions vs English."""
    from analysis import h5
    from corpus.user_tasks import h5_tasks

    h5.GRID = OUT
    print()
    print(h5.table(lang, defenses, h5_tasks(load_cases("en", False))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--defenses", nargs="+", default=["undefended"])
    ap.add_argument("--attack", default=DEFAULT_ATTACK)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5, help="Class A block threshold (scores are always kept)")
    ap.add_argument("--classifier-device", default="cpu",
                    help="'cpu' or a CUDA index; vLLM holds the T4, so CPU is the safe default")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="run unreviewed localized goals/templates (pilot; cell files are marked)")
    ap.add_argument("--localized-env", action="store_true",
                    help="C3: run each language in its localized environment (corpus/environments/); "
                         "cells and clean passes are stored separately and need a _locenv probe")
    ap.add_argument("--scorable-only", action="store_true",
                    help="defended cells run only the scorable set S(L) (needs a probe); "
                         "undefended always runs every case")
    ap.add_argument("--localized-user-task", action="store_true",
                    help="H5: the user's instruction in the language (corpus.user_tasks), clean pass only, "
                         "on the language-fair tasks; English is the existing clean pass")
    ap.add_argument("--results-dir", default=None,
                    help="root for probes and grid (default analysis/out); a second model gets its own "
                         "so its cells never read as the 14B's")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="print '<lang> <defense> done|repair|missing' per cell and exit (no server needed)")
    args = ap.parse_args()
    if args.results_dir:
        global OUT, PROBES
        PROBES = Path(args.results_dir)
        OUT = PROBES / "grid"
    usertask = args.localized_user_task

    from defenses.registry import DEFENSES, is_known

    unknown = [d for d in args.defenses if not is_known(d)]
    if unknown:
        sys.exit(f"unknown defense(s): {unknown}; known: undefended {' '.join(DEFENSES)}")
    # transformers' pipeline: -1 = CPU, n = cuda:n
    args.classifier_device = -1 if args.classifier_device == "cpu" else int(args.classifier_device)

    if args.status:
        for lang in args.langs:
            for d in args.defenses:
                print(f"{lang} {d} {cell_status(lang, d, args.attack, args.localized_env, usertask)}")
        return 0

    if args.summary_only:
        for lang in args.langs:
            summarise_usertask(lang, args.defenses) if usertask else \
                summarise_lang(lang, args.attack, args.defenses, args.localized_env)
        return 0

    from harness.local_client import DEFAULT_BASE_URL, DEFAULT_LOCAL_MODEL, OpenAICompatClient

    client = OpenAICompatClient(args.model or DEFAULT_LOCAL_MODEL,
                                base_url=args.base_url or DEFAULT_BASE_URL, cache=ResponseCache())
    OUT.mkdir(parents=True, exist_ok=True)
    built: dict = {}  # one instance per defense: a classifier loads its weights once
    incomplete = 0

    for lang in args.langs:  # language-by-language: a partial grid is whole languages
        if usertask and lang == "en":
            print("[en] --localized-user-task: English is the existing clean pass, nothing to run")
            continue
        cases = load_cases(lang, args.allow_unverified)
        if usertask:
            from corpus.user_tasks import localize_user_tasks

            cases = localize_user_tasks(cases, lang, allow_unverified=args.allow_unverified)
        if args.limit:
            cases = cases[: args.limit]
        loc = args.localized_env and lang != "en"
        if loc:
            from corpus.environments import load_overlay

            if not any(e["status"] == "active" for e in load_overlay(lang)):
                sys.exit(f"[{lang}] --localized-env but no validated overlay; "
                         "run python -m corpus.environments build/validate first")
        amap, probe_file = load_probe(lang, client.model, loc)
        if probe_file is None:
            print(f"[{lang}] WARNING: no Phase 2 probe for {client.model}; uASR = ASR (all assumed achievable)")
        # fail fast, before any GPU time, if the template is missing or unverified
        render("x", attack=args.attack, lang=lang, allow_unverified=args.allow_unverified)

        for name in args.defenses:
            if name not in built:
                built[name] = build_defense(name, client, args)
            defense = built[name]

            # A file on disk is kept, except results lost to the server: those run
            # again (successes are kept, not replayed) and are merged back in.
            cp = clean_path(name, lang, loc, usertask)
            old_clean = load_cell(cp, "clean") if cp.exists() and not args.rerun else []
            redo = lost(cp, "clean") if old_clean else set()
            if old_clean and not redo and usertask:
                print(f"[{lang}] {name}: {lang} user tasks done, skipping ({cp})")
            if not old_clean or redo:
                todo = [c for c in cases if c.case_id in redo] if old_clean else cases
                print(f"[clean] {name}: {len({user_task_key(c) for c in todo})} distinct user tasks "
                      f"for {len(todo)} cases{' (' + lang + ' environment)' if loc else ''}"
                      f"{' (' + lang + ' user instructions)' if usertask else ''}"
                      f"{f' (repair: {len(redo)} lost to the server)' if old_clean else ''}")
                with RunTrace(phase="phase5", lang=lang if (loc or usertask) else "en",
                              defense=f"{name}-clean{'-locenv' if loc else ''}{'-usertask' if usertask else ''}",
                              model=client.model) as tr:
                    clean = run_clean_dedup(todo, defense, client, attack=args.attack,
                                            workers=args.workers, trace=tr, localized_env=loc)
                cp.write_text(json.dumps({
                    "defense": name, "model": client.model, "clean_dedup_by_user_task": True,
                    "environment": f"localized:{lang}" if loc else "english (unlocalized)",
                    "user_instructions": lang if usertask else "en",
                    "unverified_corpus": args.allow_unverified,
                    "clean": [r.model_dump() for r in merge(old_clean, clean)]}, indent=1), encoding="utf-8")
            if usertask:  # utility only: no attack cell
                if lost(cp, "clean"):
                    incomplete += 1
                    print(f"[{lang}] {name}: {len(lost(cp, 'clean'))} results lost to the server -- NOT done")
                continue

            f = cell_path(lang, name, args.attack, loc)
            old = load_cell(f) if f.exists() and not args.rerun else []
            redo = lost(f, "attack") if old else set()
            if old and not redo:
                if lost(cp, "clean"):
                    incomplete += 1
                    print(f"[{lang}] {name}: clean pass still has results lost to the server -- NOT done")
                else:
                    print(f"[{lang}] {name}: done, skipping ({f})")
                continue
            run_cases = cases
            if args.scorable_only and name != "undefended":
                if probe_file is None:
                    sys.exit(f"[{lang}] --scorable-only needs a Phase 2 probe for {client.model}")
                smap, _ = scorable_map(lang, client.model, loc, args.attack)
                run_cases = [c for c in cases if smap.get((c.case_id, c.lang), True)]
            scorable_only = run_cases is not cases
            if old:
                run_cases = [c for c in cases if c.case_id in redo]
            print(f"[{lang}] {name}: {len(run_cases)} cases"
                  f"{' (scorable set only)' if scorable_only else ''}"
                  f"{f' (repair: {len(redo)} lost to the server)' if old else ''}")
            with RunTrace(phase="phase5", lang=lang, defense=f"{name}-{args.attack}{'-locenv' if loc else ''}",
                          model=client.model) as tr:
                results = run_cells(run_cases, defense, client, inject=True, attack=args.attack,
                                    workers=args.workers, trace=tr,
                                    allow_unverified=args.allow_unverified, localized_env=loc)
            results = apply_achievability(merge(old, results),
                                          scorable_map(lang, client.model, loc, args.attack)[0])
            f.write_text(json.dumps({
                "lang": lang, "defense": name, "attack": args.attack, "model": client.model,
                "probe": probe_file, "unverified_corpus": args.allow_unverified,
                "environment": f"localized:{lang}" if loc else "english (unlocalized)",
                "scorable_only": scorable_only,
                "attack": [r.model_dump() for r in results]}, indent=1), encoding="utf-8")
            errored = sum(r.errored for r in results)
            print(f"[{lang}] {name}: ASR {asr(results)}  uASR {uasr(results)}  errored {errored}")
            still = len(lost(f, "attack") | lost(cp, "clean"))
            if still:
                incomplete += 1
                print(f"[{lang}] {name}: {still} results lost to the server -- NOT done; "
                      "a re-run repairs only those")

        summarise_usertask(lang, args.defenses) if usertask else summarise_lang(lang, args.attack, args.defenses, loc)
    # 3 = the server failed under us: the notebook restarts vLLM and retries the step.
    return 3 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
