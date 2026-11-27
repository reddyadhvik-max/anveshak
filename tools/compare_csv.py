"""Reproduction check: two submission CSVs must hold the same rows (same tile, class, box within --tol px, score
within --tol-score). Exits non-zero otherwise.

  python -m tools.compare_csv runs/submissions/r1_final3_g025.csv runs/submissions/r1_final3_repro.csv
"""
import argparse
import sys

from tools.oe_score import read_preds


def diff(a, b, tol, tol_score):
    """Returns (n_a, n_b, n_mismatched) after sorting both by (tile, label, score, box)."""
    key = lambda p: (p["tile"], p["label"], -p["score"], p["box"])
    a, b = sorted(a, key=key), sorted(b, key=key)
    if len(a) != len(b):
        return len(a), len(b), abs(len(a) - len(b))
    bad = sum(1 for p, q in zip(a, b) if p["tile"] != q["tile"] or p["label"] != q["label"]
              or abs(p["score"] - q["score"]) > tol_score or max(abs(x - y) for x, y in zip(p["box"], q["box"])) > tol)
    return len(a), len(b), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--tol-score", type=float, default=1e-4)
    x = ap.parse_args()
    na, nb, bad = diff(read_preds(x.a), read_preds(x.b), x.tol, x.tol_score)
    print(f"{na} vs {nb} rows, {bad} mismatched")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
