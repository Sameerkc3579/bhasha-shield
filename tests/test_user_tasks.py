"""corpus/user_tasks.py and analysis/h5.py: the H5 test (does a defense cost more utility off-English?)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("agentdojo")

from corpus import user_tasks as ut


@pytest.fixture(scope="module")
def cases():
    return ut._en_cases()


def test_h5_tasks_exclude_reply_scored_and_listed_tasks(cases):
    tasks = ut.h5_tasks(cases)
    assert len(tasks) == 32
    assert not set(tasks) & set(ut.EXCLUDED)
    # travel scores almost every task on the reply text; banking almost never
    assert sum(t.startswith("travel/") for t in tasks) == 1
    assert "banking/user_task_0" in tasks


def test_reads_reply_tells_reply_checkers_from_state_checkers():
    assert not ut.reads_reply(ut._task_class("banking/user_task_0"))   # checks the transaction
    assert ut.reads_reply(ut._task_class("travel/user_task_1"))        # checks the answer text


def test_hindi_drafts_pass_the_gates(cases):
    assert ut.check("hi", cases) == []


def test_gate_catches_a_lost_entity_and_a_lost_checker_string(cases, tmp_path, monkeypatch):
    rows = [json.loads(l) for l in ut.path("hi").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        if r["user_task"] == "user_task_4" and r["domain"] == "banking":
            r["text"] = r["text"].replace("GB29NWBK60161331926819", "उस खाते")
        if r["user_task"] == "user_task_7" and r["domain"] == "workspace":
            r["text"] = r["text"].replace("Dental check-up", "दाँतों की जाँच")
    monkeypatch.setattr(ut, "DIR", tmp_path)
    ut.path("hi").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    problems = ut.check("hi", cases)
    assert any("banking/user_task_4" in p and "GB29NWBK60161331926819" in p for p in problems)
    assert any("workspace/user_task_7" in p and "Dental check-up" in p for p in problems)


def test_unverified_user_tasks_are_refused_without_the_pilot_flag(cases):
    with pytest.raises(PermissionError, match="unverified draft"):
        ut.localize_user_tasks(cases, "hi")
    out = ut.localize_user_tasks(cases, "hi", allow_unverified=True)
    assert len(out) == 61 and all(c.lang == "hi" for c in out)
    assert all(ut.task_key(c) in ut.h5_tasks(cases) for c in out)
    assert out[0].user_task != cases[0].user_task


# --------------------------------------------------------------------------
# analysis/h5.py
# --------------------------------------------------------------------------

def _clean(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"clean": rows}), encoding="utf-8")


def _r(task, ok, blocked=False, i=0):
    d, t = task.split("/")
    return {"case_id": f"{d}__{t}__injection_task_{i}", "lang": "en", "defense": "d", "model": "m",
            "user_task_success": ok, "blocked": blocked}


def test_h5_gap_is_the_extra_utility_cost_off_english(tmp_path, monkeypatch):
    from analysis import h5

    monkeypatch.setattr(h5, "GRID", tmp_path)
    tasks = [f"banking/user_task_{i}" for i in range(4)]
    _clean(h5.clean_file("undefended", "en"), [_r(t, True) for t in tasks])
    _clean(h5.clean_file("allowlist", "en"), [_r(t, True) for t in tasks])            # free in English
    _clean(h5.clean_file("undefended", "hi"), [_r(t, True) for t in tasks])
    _clean(h5.clean_file("allowlist", "hi"),                                           # costs half in Hindi
           [_r(t, i % 2 == 0, blocked=i % 2 == 1) for i, t in enumerate(tasks)]
           + [_r(tasks[0], False, i=9)])                                               # a sibling copy is ignored
    row = h5.h5_row("allowlist", "hi", tasks)
    assert row["n_tasks"] == 4 and row["cost_en"] == 0 and row["cost_lang"] == 0.5 and row["h5_gap"] == 0.5
    assert row["blocked_lang"] == 0.5 and row["blocked_en"] == 0
    assert row["mcnemar_en_vs_lang"]["only_en"] == 2
    assert h5.h5_row("sandwich", "hi", tasks) is None                                  # not on disk: no row
