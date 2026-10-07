"""Every paper table and figure, regenerated from files on disk (Phase 7 rule 1).

    python -m analysis.figures                       # all of them
    python -m analysis.figures --probe-dir analysis/out/14b

Never hand-copy a number into the paper: re-run this and paste its output.
Inputs, all written by other scripts -- this one runs no model:

- `phase2_probe_{lang}[_draft].json`   capability ceiling C(L)       (scripts/phase2_probe.py)
- `analysis/out/calibration/*.json`    detector frontier             (analysis/calibrate.py)
- `analysis/out/grid/*.json`           the results matrix            (scripts/phase5_grid.py)

Outputs go to `analysis/out/figures/`: `tables.md` (every table, markdown) and
one PNG per figure (needs matplotlib; tables do not).

Colour: one fixed hue per language, the same in every figure, with a distinct
line style per language as well, so identity never rests on colour alone.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from analysis.stats import compare_attack_success, efficacy_ci, efficacy_point
from harness.scoring import CaseResult, asr, benign_utility, uasr

OUT = Path("analysis/out/figures")
CALIB = Path("analysis/out/calibration")
GRID = Path("analysis/out/grid")
LANGS = ("en", "hi", "hi_latn", "bn", "mixed")
LABEL = {"en": "English", "hi": "Hindi", "hi_latn": "Hindi (Latin)", "bn": "Bengali", "mixed": "Code-mixed"}
# Fixed categorical order -- a language keeps its colour whatever subset is plotted.
COLOR = {"en": "#2a78d6", "hi": "#eb6834", "hi_latn": "#1baf7a", "bn": "#eda100", "mixed": "#e87ba4"}
DASH = {"en": "-", "hi": "--", "hi_latn": "-.", "bn": ":", "mixed": (0, (5, 1, 1, 1))}
INK, MUTED, GRIDC, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def _pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}%"


def _ci(lo: float | None, hi: float | None) -> str:
    return "" if lo is None else f" [{100 * lo:.0f}, {100 * hi:.0f}]"


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def load_probes(probe_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for lang in LANGS:
        for name in (f"phase2_probe_{lang}.json", f"phase2_probe_{lang}_draft.json"):
            f = probe_dir / name
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                if not d.get("fake"):
                    out[lang] = d
                    break
    return out


def table_capability(probes: dict[str, dict]) -> str:
    if not probes:
        return "_No capability probe files found._\n"
    model = next(iter(probes.values()))["model"]
    rows = [f"**Capability ceiling C(L)** -- {model}. Attacker goal issued as a plain user instruction.\n",
            "| Language | C(L), cases | C(L), goals | corpus |", "|---|---|---|---|"]
    for lang, d in probes.items():
        c, g = d["c_cases"], d["c_goals"]
        rows.append(f"| {LABEL[lang]} | {_pct(c['value'])}{_ci(c['ci_low'], c['ci_high'])} ({c['k']}/{c['n']}) "
                    f"| {_pct(g['value'])} ({g['k']}/{g['n']}) | {'draft' if d.get('draft') else 'verified'} |")
    return "\n".join(rows) + "\n"


def load_calibrations() -> list[dict]:
    return [json.loads(Path(f).read_text(encoding="utf-8"))
            for f in sorted(glob.glob(str(CALIB / "*.json"))) if not f.endswith("thresholds.json")]


def table_calibration(cals: list[dict]) -> str:
    if not cals:
        return "_No calibration files found (python -m analysis.calibrate)._\n"
    out = []
    for c in cals:
        out += [f"**{c['defense']}** -- held-out split, target FPR {c['target_fpr']:.0%}. "
                "Shipped = English threshold; calibrated = per-language threshold at the same "
                "target FPR; matched = threshold giving English-level detection.\n",
                "| Language | AUC | shipped FPR | shipped TPR | calibrated FPR | calibrated TPR "
                "| matched FPR | adv-benign FPR (shipped) | p50 ms |",
                "|---|---|---|---|---|---|---|---|---|"]
        for lang, r in c["langs"].items():
            s, k, m = r["shipped"], r["calibrated"], r["matched"]
            star = "\\*" if r["unverified_corpus"] else ""
            out.append(f"| {LABEL[lang]}{star} | {r['auc']:.3f} | {_pct(s['fpr'])} | {_pct(s['tpr'])} "
                       f"| {_pct(k['fpr'])} | {_pct(k['tpr'])} | {_pct(m['fpr'])} "
                       f"| {_pct(r['adv_benign_fpr']['shipped'])} | {r['latency_ms_p50']:.0f} |")
        adaptive = {l: r["adaptive_tpr_at_calibrated"] for l, r in c["langs"].items()}
        names = sorted({n for v in adaptive.values() for n in v})
        if names:
            out += ["", f"Adaptive rewrites, TPR at the calibrated threshold ({c['defense']}):", "",
                    "| Language | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
            for lang, v in adaptive.items():
                out.append(f"| {LABEL[lang]} | " + " | ".join(_pct(v.get(n)) if n in v else "n/a"
                                                           for n in names) + " |")
        out += ["", "\\* unverified machine-drafted corpus -- pilot, not a result.", ""]
    return "\n".join(out) + "\n"


SCORABLE_ONLY: set[tuple[str, str]] = set()


def load_grid(attack: str) -> dict[tuple[str, str], list[CaseResult]]:
    """English-environment cells, restamped from the current probe + existence proof.

    Same rule as `scripts.phase5_grid.summarise_lang`, so the paper table and the
    grid printout can never disagree about the scorable set.
    """
    from harness.capability import apply_achievability
    from scripts.phase5_grid import scorable_map

    cells: dict[tuple[str, str], list[CaseResult]] = {}
    for f in glob.glob(str(GRID / f"*__*__{attack}.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        amap, _ = scorable_map(d["lang"], d["model"], False, attack)
        cells[(d["lang"], d["defense"])] = apply_achievability(
            [CaseResult.model_validate(r) for r in d["attack"]], amap)
        if d.get("scorable_only"):
            SCORABLE_ONLY.add((d["lang"], d["defense"]))
    return cells


def table_grid(attack: str) -> str:
    cells = load_grid(attack)
    if not cells:
        return f"_No grid cells for `{attack}` yet (scripts/phase5_grid.py)._\n"
    defenses = sorted({d for _, d in cells}, key=lambda d: (d != "undefended", d))
    out = [f"**The grid** -- `{attack}`. uASR = ASR on the cases the agent could perform in L; "
           "gap = uASR - ASR; E = efficacy vs this language's undefended uASR, paired bootstrap 95% CI, "
           "McNemar p.\n"]
    for lang in [l for l in LANGS if any(k[0] == l for k in cells)]:
        base = cells.get((lang, "undefended"))
        out += [f"*{LABEL[lang]}*\n", "| Defense | ASR | uASR | gap | E | p | utility |",
                "|---|---|---|---|---|---|---|"]
        for d in defenses:
            rs = cells.get((lang, d))
            if rs is None:
                continue
            a, u = asr(rs), uasr(rs)
            gap = None if a.value is None or u.value is None else u.value - a.value
            e = p = "--"
            if base is not None and d != "undefended":
                ev = efficacy_point(rs, base)
                lo, hi = efficacy_ci(rs, base)
                e = "undefined" if ev is None else f"{ev:+.2f}" + ("" if lo is None else f" [{lo:+.2f}, {hi:+.2f}]")
                p = f"{compare_attack_success(base, rs).p:.3g}"
            clean = GRID / f"clean__{d}.json"
            util = (benign_utility([CaseResult.model_validate(r) for r in
                                    json.loads(clean.read_text(encoding="utf-8"))["clean"]]).value
                    if clean.exists() else None)
            if (lang, d) in SCORABLE_ONLY:  # no raw ASR: only S(L) was run
                a_s, gap_s = "n/a (S only)", "--"
            else:
                a_s, gap_s = _pct(a.value), _pct(gap)
            out.append(f"| {d} | {a_s} | {_pct(u.value)} | {gap_s} | {e} | {p} | {_pct(util)} |")
        out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRIDC, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_frontier(c: dict, path: Path) -> None:
    """ROC per language on the held-out split; o = shipped (English threshold), ● = calibrated."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor=SURFACE)
    _style(ax)
    for lang, r in c["langs"].items():
        pts = r["roc_eval"]
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=COLOR[lang], linestyle=DASH[lang],
                linewidth=2, label=f"{LABEL[lang]}{'*' if r['unverified_corpus'] else ''} (AUC {r['auc']:.2f})")
        s, k = r["shipped"], r["calibrated"]
        ax.plot(s["fpr"], s["tpr"], "o", ms=8, mfc=SURFACE, mec=COLOR[lang], mew=2, zorder=3)
        ax.plot(k["fpr"], k["tpr"], "o", ms=8, color=COLOR[lang], mec=SURFACE, mew=1.5, zorder=4)
    ax.axvline(c["target_fpr"], color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
    ax.text(c["target_fpr"] + 0.01, 0.97, f"target FPR {c['target_fpr']:.0%}", color=MUTED, fontsize=8)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("false-positive rate on benign tool output", color=INK, fontsize=10)
    ax.set_ylabel("detection rate on injected tool output", color=INK, fontsize=10)
    ax.set_title(f"{c['defense']}: hollow = English threshold, filled = calibrated",
                 color=INK, fontsize=10, loc="left")
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_capability(probes: dict[str, dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    langs = list(probes)
    fig, ax = plt.subplots(figsize=(5.6, 0.55 * len(langs) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    for i, lang in enumerate(langs):
        c = probes[lang]["c_cases"]
        ax.barh(i, c["value"], height=0.55, color=COLOR[lang])
        ax.errorbar(c["value"], i, xerr=[[c["value"] - c["ci_low"]], [c["ci_high"] - c["value"]]],
                    color=INK, capsize=3, linewidth=1)
        ax.text(c["ci_high"] + 0.02, i, f"{100 * c['value']:.0f}%  ({c['k']}/{c['n']})",
                va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(langs)), [LABEL[l] + ("*" if probes[l].get("draft") else "") for l in langs],
                  color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("C(L): attacker goals the agent can perform when asked directly", color=INK, fontsize=9)
    ax.set_title(probes[langs[0]]["model"], color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default="analysis/out/14b")
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    probes = load_probes(Path(args.probe_dir))
    cals = load_calibrations()
    tables = "\n".join([
        "# Tables (generated by `python -m analysis.figures` -- do not edit by hand)\n",
        table_capability(probes), table_calibration(cals), table_grid(args.attack),
    ])
    (OUT / "tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            print("matplotlib not installed: tables only (pip install -e '.[analysis]')")
            return 0
        if probes:
            fig_capability(probes, OUT / "capability_ceiling.png")
        for c in cals:
            fig_frontier(c, OUT / f"frontier_{c['defense'].replace('+', '_')}.png")
        print(f"figures -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
