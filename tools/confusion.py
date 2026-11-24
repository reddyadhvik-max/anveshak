"""Why is a class's AP low? For every GT box, the highest-scoring prediction (any class, score >= --min-score) with
IoU >= 0.5 decides: right class, a wrong class (which one), or nothing there (missed / mislocalised).

  python tools/confusion.py runs/eval/A_val.csv --labels ~/data/round1/dataset_A/labels --tiles splits/dataset_A_val.txt
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

from tools.oe_score import IOU_T, iou_matrix, read_gts, read_preds


def confusion(preds, gts, min_score):
    """Returns {gt_label: Counter(top-prediction label or None)}."""
    by_tile = defaultdict(list)
    for p in preds:
        if p["score"] >= min_score:
            by_tile[p["tile"]].append(p)
    out = defaultdict(Counter)
    gt_tile = defaultdict(list)
    for t, label, box in gts:
        gt_tile[t].append((label, box))
    for t, g in gt_tile.items():
        ps = by_tile.get(t, [])
        if not ps:
            for label, _ in g:
                out[label][None] += 1
            continue
        m = iou_matrix([b for _, b in g], [p["box"] for p in ps])
        sc = np.array([p["score"] for p in ps])
        for i, (label, _) in enumerate(g):
            hit = np.nonzero(m[i] >= IOU_T)[0]
            out[label][ps[hit[sc[hit].argmax()]]["label"] if len(hit) else None] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--min-score", type=float, default=0.05)
    a = ap.parse_args()
    tiles = [l.strip() for l in open(a.tiles) if l.strip()]
    res = confusion(read_preds(a.preds), read_gts(a.labels, tiles), a.min_score)
    print(f"top prediction (score >= {a.min_score}, IoU >= {IOU_T}) for each GT box")
    for label, c in sorted(res.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        wrong = [(k, v) for k, v in c.most_common() if k not in (label, None)][:3]
        print(f"{label:<14} n {n:>6}  right {c[label] / n:5.1%}  none {c[None] / n:5.1%}  "
              + "  ".join(f"{k} {v / n:.1%}" for k, v in wrong))


if __name__ == "__main__":
    main()
