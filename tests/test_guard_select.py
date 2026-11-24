"""Phase 3 step 5: the winner is picked on the calibration half only (docs/v2-plan.md)."""

from __future__ import annotations

import random

from analysis import guard_select as S


def rows(cal_tpr: float, *, test_scores: float = 0.5, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(100):   # calibration negatives: scores 0..0.99
        out.append({"item_id": f"b{i}", "label": "benign", "half": "calibration", "kind": "v1_benign",
                    "adversarial_benign": False, "goal": None, "score": i / 100})
    for i in range(50):    # calibration positives: a share above every negative
        s = 0.999 if i < cal_tpr * 50 else 0.0
        out.append({"item_id": f"p{i}", "label": "injection", "half": "calibration", "kind": "v1_embedded",
                    "adversarial_benign": False, "goal": f"g{i % 5}", "score": s})
    for i in range(60):    # test half: anything
        out.append({"item_id": f"t{i}", "label": "injection" if i % 2 else "benign", "half": "test",
                    "kind": "v1_embedded" if i % 2 else "v1_benign", "adversarial_benign": False,
                    "goal": f"tg{i % 4}" if i % 2 else None, "score": rng.random() * test_scores})
    return out


def test_run_names_parse():
    assert S.parse("bhasha_guard@muril__full__s2") == ("muril", "full", 2)
    assert S.parse("promptguard2") is None


def test_winner_is_the_best_worst_language_on_the_calibration_half():
    scores = {
        "bhasha_guard@a__full__s1": {"en": rows(0.9), "hi": rows(0.5), "mixed": rows(0.9)},
        "bhasha_guard@a__full__s2": {"en": rows(0.9), "hi": rows(0.5), "mixed": rows(0.9)},
        "bhasha_guard@b__full__s1": {"en": rows(0.6), "hi": rows(0.7), "mixed": rows(0.7)},
        "bhasha_guard@b__full__s2": {"en": rows(0.6), "hi": rows(0.7), "mixed": rows(0.7)},
    }
    res = S.select(scores, target=0.02)
    assert res["winner"] == "b"   # a is better on average but worse in Hindi
    assert res["models"]["a"]["worst_non_en"] == 0.5 and res["models"]["b"]["seeds"] == 2


def test_test_half_scores_cannot_change_the_choice():
    base = {"bhasha_guard@a__full__s1": {"hi": rows(0.8)}, "bhasha_guard@b__full__s1": {"hi": rows(0.6)}}
    flipped = {"bhasha_guard@a__full__s1": {"hi": rows(0.8, test_scores=0.0)},
               "bhasha_guard@b__full__s1": {"hi": rows(0.6, test_scores=1.0, seed=3)}}
    assert S.select(base, target=0.02) == S.select(flipped, target=0.02)


def test_only_full_runs_compete():
    scores = {"bhasha_guard@a__no_adv__s1": {"hi": rows(1.0)}, "bhasha_guard@b__full__s1": {"hi": rows(0.5)}}
    assert S.select(scores, target=0.02)["winner"] == "b"
