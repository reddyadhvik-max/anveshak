from oe_score import score, apply_conflict_rule, voc_ap

K = ["a", "b"]
GT = [("t1", "a", [0, 0, 10, 10]), ("t1", "b", [20, 20, 30, 30]), ("t1", "unknown", [50, 50, 60, 60])]


def P(rows):
    return [{"tile": t, "label": l, "score": s, "box": b, "row": i} for i, (t, l, s, b) in enumerate(rows)]


def test_perfect():
    r = score(P([("t1", "a", .9, [0, 0, 10, 10]), ("t1", "b", .9, [20, 20, 30, 30]),
                 ("t1", "unknown", .9, [50, 50, 60, 60])]), GT, K)
    assert r["mAP_K"] == 1 and r["U-F1"] == 1 and r["OE"] == 100


def test_no_unknowns_scores_zero():
    r = score(P([("t1", "a", .9, [0, 0, 10, 10]), ("t1", "b", .9, [20, 20, 30, 30])]), GT, K)
    assert r["mAP_K"] == 1 and r["OE"] == 0


def test_unknown_flood_on_known_kills_map():
    # higher-scored unknown on the 'a' object removes the known prediction (conflict rule)
    r = score(P([("t1", "a", .5, [0, 0, 10, 10]), ("t1", "unknown", .9, [0, 0, 10, 10]),
                 ("t1", "b", .9, [20, 20, 30, 30]), ("t1", "unknown", .9, [50, 50, 60, 60])]), GT, K)
    assert r["AP"]["a"] == 0 and r["U-Prec"] == 0.5 and r["U-Rec"] == 1


def test_conflict_tie_earlier_row_wins():
    kept = apply_conflict_rule(P([("t1", "unknown", .5, [0, 0, 10, 10]), ("t1", "a", .5, [0, 0, 10, 10])]))
    assert [p["label"] for p in kept] == ["unknown"]


def test_known_vs_known_overlap_untouched():
    assert len(apply_conflict_rule(P([("t1", "a", .5, [0, 0, 10, 10]), ("t1", "b", .6, [0, 0, 10, 10])]))) == 2


def test_duplicate_is_fp_and_known_on_unknown_is_fp():
    r = score(P([("t1", "a", .9, [0, 0, 10, 10]), ("t1", "a", .8, [0, 0, 10, 10]),
                 ("t1", "b", .95, [50, 50, 60, 60]), ("t1", "b", .9, [20, 20, 30, 30])]), GT, K)
    assert r["AP"]["a"] == 1.0  # duplicate after the TP does not lower all-point AP
    assert abs(r["AP"]["b"] - 0.5) < 1e-9  # FP on the unknown object ranked first -> precision 0.5 at recall 1


def test_class_without_gt_is_skipped():
    r = score(P([("t1", "a", .9, [0, 0, 10, 10])]), [("t1", "a", [0, 0, 10, 10])], ["a", "b"])
    assert r["skipped_classes"] == ["b"] and r["mAP_K"] == 1


def test_voc_ap_half():
    assert abs(voc_ap([0, 1], 1) - 0.5) < 1e-9
