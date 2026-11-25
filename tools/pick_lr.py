"""Pick B_final2's lr0 from B_final's forgetting: A-val mAP_K of A2 (its start) minus B_final's.

  python -m tools.pick_lr --root ~/data/round1 --before A2 --after final --out runs/eval/bf2_lr0.txt
Scores come from runs/eval/A_val_<name>.csv. B-val is inside B_final's training data, so the few-shot side of the
trade-off can't be measured; lr is only lowered when the base-class loss is clearly large.
"""
import argparse
from pathlib import Path

from tools.oe_score import read_gts, read_known, read_preds, score


def choose_lr(drop, base=0.001, limit=0.01):
    """Halve lr0 when the A-val drop exceeds limit (0.01 on 30 classes ~ 0.0075 on the 40-class mean)."""
    return base / 2 if drop > limit else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    known = read_known([a.root / "class_list.csv"])
    gts = read_gts(a.root / "dataset_A" / "labels", [l.strip() for l in open("splits/dataset_A_val.txt") if l.strip()])
    m = {n: score(read_preds(f"runs/eval/A_val_{n}.csv"), gts, known)["mAP_K"] for n in (a.before, a.after)}
    lr = choose_lr(m[a.before] - m[a.after])
    print(f"{a.before} {m[a.before]:.4f} -> {a.after} {m[a.after]:.4f} (drop {m[a.before] - m[a.after]:+.4f}): lr0 {lr}")
    Path(a.out).write_text(f"{lr}\n")


if __name__ == "__main__":
    main()
