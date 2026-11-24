"""Pick the A2 training data: the data yaml of the ablation arm with the best A-val mAP_K.

  python -m tools.pick_data --root ~/data/round1 --arm abl_tile_s25 data_A_rfs.yaml --arm abl_tile_s25_rfs05 data_A_rfs05.yaml \
      --out runs/eval/a2_data.txt
Scores come from runs/eval/A_val_<arm>.csv.
"""
import argparse
from pathlib import Path

from tools.oe_score import read_gts, read_known, read_preds, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "DATA_YAML"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    known = read_known([a.root / "class_list.csv"])
    gts = read_gts(a.root / "dataset_A" / "labels", [l.strip() for l in open("splits/dataset_A_val.txt") if l.strip()])
    res = {n: score(read_preds(f"runs/eval/A_val_{n}.csv"), gts, known)["mAP_K"] for n, _ in a.arm}
    win = max(res, key=res.get)
    print({n: round(v, 4) for n, v in res.items()}, "->", win)
    Path(a.out).write_text(f"{a.root / 'yolo' / dict(a.arm)[win]}\n")


if __name__ == "__main__":
    main()
