"""Pick the B variant for the final stage: best B-val mAP_K (10 few-shot classes) among the variants whose A-val
mAP_K (30 base classes) stays within --max-drop of the best A-val. Writes the winner's final-stage data yaml path.

  python tools/pick_variant.py --root ~/data/round1 --variant B data_B_mix_all.yaml --variant B2 data_B_mix_all_cp_all.yaml \
      --out runs/eval/final_data.txt
Scores come from runs/eval/A_val_<name>.csv and runs/eval/B_val_<name>.csv (written by the eval steps).
"""
import argparse
import json
from pathlib import Path

from tools.oe_score import read_gts, read_known, read_preds, score


def choose(results, max_drop):
    """results: {name: (a_val, b_val)}. Returns the name with the best b_val among those with a_val >= best - max_drop."""
    best_a = max(a for a, _ in results.values())
    ok = {n: r for n, r in results.items() if r[0] >= best_a - max_drop}
    return max(ok, key=lambda n: (ok[n][1], ok[n][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="data root ({D})")
    ap.add_argument("--variant", nargs=2, action="append", required=True, metavar=("NAME", "FINAL_YAML"))
    ap.add_argument("--max-drop", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    known = read_known([a.root / "class_list.csv"])
    gts = {s: read_gts(a.root / f"dataset_{s}" / "labels", [l.strip() for l in open(f"splits/dataset_{s}_val.txt") if l.strip()])
           for s in "AB"}
    res = {n: tuple(score(read_preds(f"runs/eval/{s}_val_{n}.csv"), gts[s], known)["mAP_K"] for s in "AB") for n, _ in a.variant}
    win = choose(res, a.max_drop)
    print(json.dumps({n: {"A_val_mAP_K": round(r[0], 4), "B_val_mAP_K": round(r[1], 4)} for n, r in res.items()}, indent=1))
    path = a.root / "yolo" / dict(a.variant)[win]
    Path(a.out).write_text(f"{path}\n")
    print(f"chosen {win} -> {path}")


if __name__ == "__main__":
    main()
