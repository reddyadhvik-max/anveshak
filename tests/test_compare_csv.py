from tools.compare_csv import diff


def P(t, l, s, b):
    return {"tile": t, "label": l, "score": s, "box": b, "row": 0}


def test_diff():
    a = [P("t", "car", 0.9, [0, 0, 10, 10]), P("t", "unknown", 0.2, [5, 5, 9, 9])]
    assert diff(a, list(reversed(a)), 0.05, 1e-4) == (2, 2, 0)
    assert diff(a, [P("t", "car", 0.9, [0, 0, 10.2, 10]), a[1]], 0.05, 1e-4)[2] == 1
    assert diff(a, a[:1], 0.05, 1e-4)[2] == 1
