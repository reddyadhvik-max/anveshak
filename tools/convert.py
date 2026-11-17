"""round1 JSON labels -> YOLO txt (written next to each json, which is where Ultralytics looks:
images/x.tif -> labels/x.txt), a deterministic stratified train/val split (committed in splits/), and
dataset yamls for each stage.

  python tools/convert.py --root ~/data/round1
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

NAMES = {}  # filled from class_list.csv; slots 40-49 reserved for Round 2, 50 unknown, 51 objectness
SPLITS = Path(__file__).resolve().parent.parent / "splits"


def load_names(root):
    import csv
    names = {i: f"r2_reserved_{i}" for i in range(40, 50)}
    for r in csv.DictReader(open(root / "class_list.csv", encoding="utf-8")):
        names[int(r["class_id"])] = r["label"]
    names[51] = "object"
    return dict(sorted(names.items()))


def to_yolo(label_path):
    d = json.loads(label_path.read_text())
    assert d["bbox_format"] == "xyxy", label_path
    w, h = d["width"], d["height"]
    lines, classes = [], set()
    for a in d["annotations"]:
        x1, y1, x2, y2 = a["bbox"]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{a['class_id']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        classes.add(a["class_id"])
    label_path.with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return classes


def stratified_val(tile_classes, frac, seed=0):
    """Every class gets >= ceil(frac * its tile count) tiles in val; empty tiles are sampled at frac."""
    rng = random.Random(seed)
    tiles = sorted(tile_classes)
    rng.shuffle(tiles)
    of = defaultdict(list)
    for t in tiles:
        for c in tile_classes[t]:
            of[c].append(t)
    val, have = set(), defaultdict(int)
    for c in sorted(of, key=lambda c: len(of[c])):  # rarest first
        need = math.ceil(frac * len(of[c]))
        for t in of[c]:
            if have[c] >= need:
                break
            if t not in val:
                val.add(t)
                for c2 in tile_classes[t]:
                    have[c2] += 1
    empty = [t for t in tiles if not tile_classes[t]]
    val |= set(empty[: round(frac * len(empty))])
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="unzipped round1/ folder")
    ap.add_argument("--val-frac", type=float, default=0.10)
    a = ap.parse_args()
    root = a.root.expanduser().resolve()
    names = load_names(root)
    SPLITS.mkdir(exist_ok=True)
    out = root / "yolo"
    out.mkdir(exist_ok=True)

    for ds, frac in (("dataset_A", a.val_frac), ("dataset_B", 0.25)):
        tc = {p.stem: to_yolo(p) for p in sorted((root / ds / "labels").glob("*.json"))}
        split_file = SPLITS / f"{ds}_val.txt"
        if split_file.exists():  # the split is frozen once committed
            val = set(split_file.read_text().split())
        else:
            val = stratified_val(tc, frac)
            split_file.write_text("\n".join(sorted(val)) + "\n")
        for part, ids in (("train", sorted(set(tc) - val)), ("val", sorted(val)), ("all", sorted(tc))):
            (out / f"{ds}_{part}.txt").write_text("".join(f"{root / ds / 'images' / t}.tif\n" for t in ids))
        print(ds, "tiles", len(tc), "val", len(val))

    names_yaml = "".join(f"  {i}: '{n}'\n" for i, n in names.items())
    for stage, train, val in (("A", "dataset_A_train", "dataset_A_val"),
                              ("B", "dataset_B_train", "dataset_B_val")):
        (out / f"data_{stage}.yaml").write_text(
            f"path: {out}\ntrain: {out / (train + '.txt')}\nval: {out / (val + '.txt')}\nnc: 52\nnames:\n{names_yaml}")
    print("wrote", out)


if __name__ == "__main__":
    main()
