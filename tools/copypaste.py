"""Few-shot copy-paste (PLAN 5.5): paste Set B few-shot objects onto Set A tiles, in free space.

  python -m tools.copypaste --data ~/data/round1/yolo/data_B.yaml --bg ~/data/round1/yolo/data_A.yaml \
      --mix ~/data/round1/yolo/data_B_mix.yaml --n 1500
  -> <root>/dataset_B_cp/{images, labels}, yolo/data_B_mix_cp.yaml (= the B mix list + the pasted tiles)

Each crop is a Set B train GT box (classes 30-39) with 10% context, randomly flipped, rotated by 90-degree steps
and scaled 0.8-1.25. Classes are drawn uniformly (not by frequency). A paste must not touch any labelled box
(Set A tiles are fully labelled for the base classes) or another paste. B-val is never used.
"""
import argparse
import random
from pathlib import Path

import numpy as np


def transform(crop, box, rnd):
    """crop: HxWx3, box: (x1, y1, x2, y2) of the object inside it. Returns the augmented (crop, box)."""
    mask = np.zeros(crop.shape[:2], np.uint8)
    mask[box[1]:box[3], box[0]:box[2]] = 1
    if rnd.random() < 0.5:
        crop, mask = crop[:, ::-1], mask[:, ::-1]
    if rnd.random() < 0.5:
        crop, mask = crop[::-1], mask[::-1]
    k = rnd.randrange(4)
    crop, mask = np.rot90(crop, k), np.rot90(mask, k)
    s = rnd.uniform(0.8, 1.25)
    import cv2
    size = (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s)))
    crop = cv2.resize(np.ascontiguousarray(crop), size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(np.ascontiguousarray(mask), size, interpolation=cv2.INTER_NEAREST)
    ys, xs = mask.nonzero()
    if not len(xs):  # a 1-px object can vanish in the resize; fall back to the whole crop
        return crop, (0, 0, crop.shape[1], crop.shape[0])
    return crop, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def free(box, taken):
    """True if box (x1, y1, x2, y2) overlaps none of the taken boxes."""
    return all(box[2] <= t[0] or box[0] >= t[2] or box[3] <= t[1] or box[1] >= t[3] for t in taken)


def yolo_boxes(txt, w, h):
    rows = [l.split() for l in open(txt) if l.strip()] if Path(txt).exists() else []
    return rows, [((float(r[1]) - float(r[3]) / 2) * w, (float(r[2]) - float(r[4]) / 2) * h,
                   (float(r[1]) + float(r[3]) / 2) * w, (float(r[2]) + float(r[4]) / 2) * h) for r in rows]


def main():
    import cv2
    from ultralytics.data.utils import img2label_paths
    from ultralytics.utils import YAML
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="data_B.yaml (crops come from its train list)")
    ap.add_argument("--bg", required=True, type=Path, help="data_A.yaml (background tiles from its train list)")
    ap.add_argument("--mix", required=True, type=Path, help="data_B_mix.yaml from tools.pseudo; its train list is extended")
    ap.add_argument("--n", type=int, default=1500, help="pasted tiles to make")
    ap.add_argument("--per-tile", type=int, default=3, help="max pastes per tile")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    data, bg, mix = YAML.load(a.data), YAML.load(a.bg), YAML.load(a.mix)

    crops = {}  # class -> [(crop, box in crop)]
    for img in (l.strip() for l in open(data["train"]) if l.strip()):
        im = cv2.imread(img)
        h, w = im.shape[:2]
        rows, boxes = yolo_boxes(img2label_paths([img])[0], w, h)
        for r, (x1, y1, x2, y2) in zip(rows, boxes):
            c = int(r[0])
            if not 30 <= c <= 39:
                continue
            mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            X1, Y1, X2, Y2 = int(max(x1 - mx, 0)), int(max(y1 - my, 0)), int(min(x2 + mx, w)), int(min(y2 + my, h))
            b = (int(x1) - X1, int(y1) - Y1, max(int(x2) - X1, int(x1) - X1 + 1), max(int(y2) - Y1, int(y1) - Y1 + 1))
            crops.setdefault(c, []).append((im[Y1:Y2, X1:X2].copy(), b))
    print("crops per class:", {c: len(v) for c, v in sorted(crops.items())})

    bgs = [l.strip() for l in open(bg["train"]) if l.strip()]
    first = Path(bgs[0])
    dst = first.parent.parent.with_name("dataset_B_cp")
    for d in ("images", "labels"):
        (dst / d).mkdir(parents=True, exist_ok=True)
    out, pasted = [], 0
    for i in range(a.n):
        src = rnd.choice(bgs)
        im = cv2.imread(src)
        h, w = im.shape[:2]
        rows, taken = yolo_boxes(img2label_paths([src])[0], w, h)
        lines = [" ".join(r) for r in rows]
        for _ in range(rnd.randint(1, a.per_tile)):
            c = rnd.choice(sorted(crops))
            crop, (bx1, by1, bx2, by2) = transform(*rnd.choice(crops[c]), rnd)
            ch, cw = crop.shape[:2]
            if ch >= h or cw >= w:
                continue
            for _ in range(30):  # find free space
                x, y = rnd.randrange(w - cw), rnd.randrange(h - ch)
                if free((x, y, x + cw, y + ch), taken):
                    im[y:y + ch, x:x + cw] = crop
                    taken.append((x, y, x + cw, y + ch))
                    X1, Y1, X2, Y2 = x + bx1, y + by1, x + bx2, y + by2
                    lines.append(f"{c} {(X1 + X2) / 2 / w:.6f} {(Y1 + Y2) / 2 / h:.6f} {(X2 - X1) / w:.6f} {(Y2 - Y1) / h:.6f}")
                    pasted += 1
                    break
        name = f"cp{i:05d}_{Path(src).stem}"
        cv2.imwrite(str(dst / "images" / f"{name}.png"), im)
        (dst / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        out.append(str(dst / "images" / f"{name}.png"))
    train = Path(mix["train"])
    lst = train.with_name("dataset_B_mix_cp_train.txt")
    lst.write_text(train.read_text().rstrip("\n") + "\n" + "\n".join(out) + "\n")
    YAML.save(a.mix.with_name("data_B_mix_cp.yaml"), {**mix, "train": str(lst)})
    print(f"pasted {pasted} few-shot objects into {len(out)} Set A tiles; wrote data_B_mix_cp.yaml")


if __name__ == "__main__":
    main()
