"""scripts/phase5_grid.py end to end on real AgentDojo cases, with a fake model.

Proves the plumbing the Colab run depends on: cells and clean passes land on
disk, a re-run skips finished cells, probe achievability is stamped, and the
summary reads the files back.
"""

from __future__ import annotations

import functools
import json
import sys

import pytest

pytest.importorskip("agentdojo")

import scripts.phase5_grid as grid
from harness.runner import ModelResponse


class DoneClient:
    """An agent that answers at once -- no tools, no attack success, no utility."""

    def __init__(self, model, **kw):
        self.model = model
        self.calls = 0

    def complete(self, *, system, messages, tools):
        self.calls += 1
        return ModelResponse(text='["send_money"]' if not tools else "done")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(grid, "OUT", tmp_path / "grid")
    monkeypatch.setattr(grid, "PROBES", tmp_path)
    monkeypatch.setattr(grid, "ResponseCache", lambda: None)
    import harness.local_client as lc

    made: list[DoneClient] = []

    def factory(model, **kw):
        made.append(DoneClient(model))
        return made[-1]

    monkeypatch.setattr(lc, "OpenAICompatClient", factory)
    monkeypatch.setattr(grid, "RunTrace", functools.partial(grid.RunTrace, root=tmp_path / "traces"))
    return tmp_path, made


def run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["phase5_grid", *argv])
    return grid.main()


def test_grid_writes_cells_resumes_and_stamps_the_probe(sandbox, monkeypatch, capsys):
    tmp, made = sandbox
    cases = grid.load_cases("en", False)[:3]
    model = "fake-14b"
    # a probe that calls the first case unachievable
    (tmp / "phase2_probe_en.json").write_text(json.dumps({
        "model": model, "fake": False,
        "probe": [{"case_id": c.case_id, "lang": "en", "defense": "capability_probe", "model": model,
                   "attacker_task_achievable": i != 0} for i, c in enumerate(cases)],
    }), encoding="utf-8")

    argv = ["--langs", "en", "--defenses", "undefended", "honeytool", "camel_lite",
            "--limit", "3", "--workers", "2", "--model", model]
    assert run(argv, monkeypatch) == 0

    cell = json.loads((tmp / "grid" / "en__honeytool__important_instructions.json").read_text(encoding="utf-8"))
    assert len(cell["attack"]) == 3 and cell["probe"].endswith("phase2_probe_en.json")
    stamped = {r["case_id"]: r["attacker_task_achievable"] for r in cell["attack"]}
    assert stamped[cases[0].case_id] is False and stamped[cases[1].case_id] is True
    assert (tmp / "grid" / "clean__camel_lite.json").exists()

    calls_first = sum(c.calls for c in made)
    assert run(argv, monkeypatch) == 0                       # second run: everything cached on disk
    assert sum(c.calls for c in made) == calls_first          # ...so no model calls at all
    out = capsys.readouterr().out
    assert "done, skipping" in out and "camel_lite" in out


def test_unknown_defense_is_refused(sandbox, monkeypatch):
    with pytest.raises(SystemExit):
        run(["--defenses", "nope"], monkeypatch)


def test_unverified_template_is_refused_before_any_model_call(sandbox, monkeypatch):
    """A real (non-pilot) run must not spend GPU time on a machine-drafted payload."""
    from corpus.attacks import TEMPLATE_STATUS

    # Any language still on a machine draft (hi was verified 2026-10-07).
    pending = [l for l, s in TEMPLATE_STATUS.items() if l != "en" and not s["verified_by"]]
    if not pending:
        pytest.skip("every template has been verified")
    _, made = sandbox
    with pytest.raises(PermissionError, match="unverified draft"):
        run(["--langs", pending[0], "--limit", "1"], monkeypatch)
    assert sum(c.calls for c in made) == 0


# --------------------------------------------------------------------------
# 2026-10-07: one scorable set per language, re-applied at summary time
# --------------------------------------------------------------------------

def _cell(path, lang, defense, rows, model="m", **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lang": lang, "defense": defense, "attack": rows, "model": model, **extra}))


def _row(cid, ok, lang="en", achievable=True):
    return {"case_id": cid, "lang": lang, "defense": "d", "model": "m",
            "attacker_task_success": ok, "attacker_task_achievable": achievable}


def test_existence_proof_adds_only_undefended_successes():
    from harness.capability import with_existence_proof
    from harness.scoring import CaseResult

    amap = {("a", "en"): False, ("b", "en"): False}
    base = [CaseResult.model_validate(_row("a", True)), CaseResult.model_validate(_row("b", False))]
    out = with_existence_proof(amap, base)
    assert out[("a", "en")] is True and out[("b", "en")] is False
    assert amap[("a", "en")] is False  # input not mutated


def test_summary_restamps_cells_stamped_from_different_probes(sandbox, capsys):
    """The 2026-10-07 bug: undefended stamped 79 scorable, sandwich 87. Summary must use one set."""
    tmp = grid.OUT.parent
    probe = [{"case_id": c, "lang": "en", "defense": "probe", "model": "m",
              "attacker_task_achievable": c in ("a", "b")} for c in "abcd"]
    (tmp / "phase2_probe_en.json").write_text(json.dumps({"model": "m", "probe": probe}))
    # stale stamps: undefended thinks only 'a' is scorable, sandwich thinks all four are
    _cell(grid.cell_path("en", "undefended", "x"), "en", "undefended",
          [_row("a", True, achievable=True), _row("b", False, achievable=False),
           _row("c", True, achievable=False), _row("d", False, achievable=False)])
    _cell(grid.cell_path("en", "sandwich", "x"), "en", "sandwich",
          [_row(c, False, achievable=True) for c in "abcd"])
    grid.summarise_lang("en", "x", ["undefended", "sandwich"])
    out = capsys.readouterr().out
    # scorable = probe {a, b} + existence proof {c}: undefended uASR 2/3, sandwich 0/3
    assert "0.667 (2/3)" in out and "0.000 (0/3)" in out


def test_scorable_only_cell_reports_no_raw_asr(sandbox, capsys):
    tmp = grid.OUT.parent
    (tmp / "phase2_probe_en.json").write_text(json.dumps({"model": "m", "probe": [
        {"case_id": "a", "lang": "en", "defense": "p", "model": "m", "attacker_task_achievable": True}]}))
    _cell(grid.cell_path("en", "undefended", "x"), "en", "undefended", [_row("a", True), _row("b", False)])
    _cell(grid.cell_path("en", "sandwich", "x"), "en", "sandwich", [_row("a", False)], scorable_only=True)
    grid.summarise_lang("en", "x", ["undefended", "sandwich"])
    assert "n/a (S only)" in capsys.readouterr().out


def test_clean_pass_runs_each_user_task_once(monkeypatch):
    """Cases sharing a user task are one clean run; every case still gets a result."""
    from corpus.schema import InjectionCase
    from harness.scoring import CaseResult

    def case(ut, it):
        return InjectionCase(case_id=f"banking__{ut}__{it}", lang="en", domain="banking", user_task="t",
                             attacker_goal_nl="g", attacker_goal_type="x", payload="", injected_into_tool="",
                             tool_call_depth=1, source_case_id=f"banking/{ut}/{it}")

    cases = [case("user_task_0", "injection_task_1"), case("user_task_0", "injection_task_2"),
             case("user_task_3", "injection_task_1")]
    seen = []

    def fake_run_cells(cs, defense, client, *, inject, **kw):
        assert inject is False
        seen.extend(c.case_id for c in cs)
        return [CaseResult(case_id=c.case_id, lang="en", defense="d", model="m",
                           user_task_success=c.case_id.startswith("banking__user_task_0")) for c in cs]

    monkeypatch.setattr(grid, "run_cells", fake_run_cells)
    out = grid.run_clean_dedup(cases, None, None, attack="x", workers=1, trace=None)
    assert seen == ["banking__user_task_0__injection_task_1", "banking__user_task_3__injection_task_1"]
    assert [r.case_id for r in out] == [c.case_id for c in cases]
    assert [r.user_task_success for r in out] == [True, True, False]
