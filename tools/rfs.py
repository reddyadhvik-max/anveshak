"""Repeat-factor sampling (LVIS, Gupta et al. 2019) as an oversampled train list, so Ultralytics needs no change.

  python tools/rfs.py ~/data/round1/yolo/data_A.yaml --thr 0.1
  -> yolo/dataset_A_train_rfs.txt and yolo/data_A_rfs.yaml (train it with --data .../data_A_rfs.yaml)

r_class = max(1, sqrt(thr / f_class)), where f_class is the fraction of train tiles containing that class.
r_tile = the max over its classes (1 for empty tiles). Fractional repeats are rounded stochastically with a fixed seed.
Val is untouched.
"""
import argparse
import math
import random
from collections import Counter
from pathlib import Path


def repeat_factors(tile_classes, thr):
    """tile_classes: list of sets of class ids, one per tile. Returns a float repeat factor per tile."""
    n = len(tile_classes)
    freq = Counter(c for cs in tile_classes for c in cs)
    r = {c: max(1.0, math.sqrt(thr / (k / n))) for c, k in freq.items()}
    return [max((r[c] for c in cs), default=1.0) for cs in tile_classes]


def main():
    from ultralytics.data.utils import img2label_paths
    from ultralytics.utils import YAML
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, help="data_A.yaml written by tools/convert.py")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="_rfs", help="output suffix (e.g. _rfs3 for thr 0.3)")
    a = ap.parse_args()
    data = YAML.load(a.data)
    train = Path(data["train"])
    imgs = [l.strip() for l in open(train) if l.strip()]
    classes = []
    for lp in img2label_paths(imgs):
        p = Path(lp)
        classes.append({int(l.split()[0]) for l in open(p) if l.strip()} if p.exists() else set())
    rnd = random.Random(a.seed)
    out = []
    for img, r in zip(imgs, repeat_factors(classes, a.thr)):
        out += [img] * (int(r) + (rnd.random() < r - int(r)))
    dst = train.with_name(train.stem + a.tag + ".txt")
    dst.write_text("\n".join(out) + "\n")
    data["train"] = str(dst)
    YAML.save(a.data.with_name(a.data.stem + a.tag + ".yaml"), data)
    print(f"{len(imgs)} tiles -> {len(out)} samples per epoch (x{len(out) / len(imgs):.2f}); wrote {dst}")


if __name__ == "__main__":
    main()
