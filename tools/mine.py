"""Unknown mining (PLAN 5.4): label confident "object but no known class" boxes in Set A train tiles as class 50.

  python -m tools.mine --weights runs/A/weights/best.pt --data ~/data/round1/yolo/data_A.yaml
  -> <root>/dataset_A_mine/{images (hard links), labels (original + mined lines)}, yolo/data_A_mine.yaml
  then: python tools/rfs.py .../data_A_mine.yaml  and train A_mine with --from runs/A/weights/last.pt

A box is mined if: objectness (slot 51) >= --obj, max known score <= --known-max, it survives class-agnostic NMS,
IoU < --iou-gt with every GT box, its size is inside the GT size range (p1..p99 of sqrt area), and it covers
<= --max-area of the tile. Val is untouched. Overlays for visual QA go to runs/mine_qa (not committed: tile imagery).
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou, nms

from anveshak.infer import forward
from anveshak.patch import OBJ_SLOT, UNK_SLOT

KNOWN = list(range(30))  # Set A classes


def select(out, gt, pp):
    """out: raw (4+52, N) head output; gt: (G, 4) xyxy px. Returns (M, 5) [x1, y1, x2, y2, obj] mined boxes."""
    xywh, s = out[:4].T, out[4:].T
    boxes = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    obj, kmax = s[:, OBJ_SLOT], s[:, KNOWN].amax(1)
    keep = (obj >= pp["obj"]) & (kmax <= pp["known_max"])
    boxes, obj = boxes[keep], obj[keep]
    k = nms(boxes, obj, 0.5)
    boxes, obj = boxes[k], obj[k]
    side = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)).sqrt()
    keep = (side >= pp["min_side"]) & (side <= pp["max_side"]) & (side ** 2 <= pp["max_area"] * pp["tile_area"])
    if len(gt):
        keep &= box_iou(boxes, gt).amax(1) < pp["iou_gt"] if len(boxes) else keep
    return torch.cat([boxes[keep], obj[keep, None]], 1)[: pp["max_per_tile"]]


def read_gt(txt, w, h):
    rows = [l.split() for l in open(txt) if l.strip()] if txt.exists() else []
    a = np.array([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4)
    return torch.tensor(np.c_[(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h,
                              (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h], dtype=torch.float32), rows


def link(src, dst):
    if not dst.exists():
        try:
            os.link(src, dst)  # NTFS hard links need no admin
        except OSError:
            shutil.copy2(src, dst)


def main():
    from ultralytics import YOLO
    from ultralytics.data.utils import img2label_paths
    from ultralytics.utils import YAML
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--obj", type=float, default=0.5)
    ap.add_argument("--known-max", type=float, default=0.2)
    ap.add_argument("--iou-gt", type=float, default=0.3)
    ap.add_argument("--max-area", type=float, default=0.25)
    ap.add_argument("--max-per-tile", type=int, default=100)
    ap.add_argument("--qa", type=int, default=50, help="overlay images for visual QA")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-mean", type=float, default=0.0, help="fail (needs a human) if fewer mined boxes per tile")
    ap.add_argument("--max-mean", type=float, default=1e9, help="fail (needs a human) if more mined boxes per tile")
    ap.add_argument("--limit", type=int, help="only the first N tiles (smoke test)")
    a = ap.parse_args()
    data = YAML.load(a.data)
    imgs = [Path(l.strip()) for l in open(data["train"]) if l.strip()][: a.limit]
    txts = [Path(p) for p in img2label_paths([str(i) for i in imgs])]
    src_root = imgs[0].parent.parent
    dst_root = src_root.with_name(src_root.name + "_mine")
    (dst_root / "images").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels").mkdir(parents=True, exist_ok=True)

    w = h = 1024  # all Round 1 tiles are 1024 x 1024 (tiles.csv)
    sides = torch.cat([((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])).sqrt() for g in (read_gt(t, w, h)[0] for t in txts)])
    pp = dict(obj=a.obj, known_max=a.known_max, iou_gt=a.iou_gt, max_area=a.max_area, tile_area=w * h,
              max_per_tile=a.max_per_tile, min_side=sides.quantile(0.01).item(), max_side=sides.quantile(0.99).item())
    print(f"GT size range (sqrt area px): {pp['min_side']:.1f}..{pp['max_side']:.1f}")

    model = YOLO(a.weights).model.float().eval().to(a.device)
    mined, counts = {}, []
    with torch.inference_mode():
        for n, (img, txt) in enumerate(zip(imgs, txts)):
            gt, rows = read_gt(txt, w, h)
            m = select(forward(model, img, a.device), gt, pp)
            counts.append(len(m))
            lines = [" ".join(r) for r in rows]
            lines += [f"{UNK_SLOT} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                      for x1, y1, x2, y2, _ in m.tolist()]
            link(img, dst_root / "images" / img.name)
            (dst_root / "labels" / txt.name).write_text("\n".join(lines) + ("\n" if lines else ""))
            if len(m):
                mined[img] = (gt, m)
            if (n + 1) % 500 == 0:
                print(f"{n + 1}/{len(imgs)} tiles, {sum(counts)} mined", flush=True)

    out = Path(data["train"]).with_name(Path(data["train"]).stem.replace("_train", "_mine_train") + ".txt")
    out.write_text("".join(f"{dst_root / 'images' / i.name}\n" for i in imgs))
    data["train"] = str(out)
    YAML.save(a.data.with_name(a.data.stem + "_mine.yaml"), data)
    c = np.array(counts)
    print(f"mined {c.sum()} unknown boxes in {(c > 0).sum()}/{len(c)} tiles "
          f"(per tile mean {c.mean():.2f}, p90 {np.percentile(c, 90):.0f}, max {c.max()}); wrote {a.data.stem}_mine.yaml")

    if a.qa and mined:
        import cv2
        qa = Path("runs/mine_qa")
        qa.mkdir(parents=True, exist_ok=True)
        for img in random.Random(0).sample(sorted(mined), min(a.qa, len(mined))):
            gt, m = mined[img]
            im = cv2.imread(str(img))
            for b in gt.int().tolist():
                cv2.rectangle(im, b[:2], b[2:], (0, 200, 0), 1)
            for *b, o in m.tolist():
                b = [int(v) for v in b]
                cv2.rectangle(im, b[:2], b[2:], (0, 0, 255), 2)
                cv2.putText(im, f"{o:.2f}", (b[0], max(b[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.imwrite(str(qa / f"{img.stem}.jpg"), im)
        print(f"QA overlays (green GT, red mined): {qa.resolve()}")
    if not a.min_mean <= c.mean() <= a.max_mean:
        raise SystemExit(f"mined {c.mean():.2f} boxes per tile, outside [{a.min_mean}, {a.max_mean}]: check runs/mine_qa, "
                         "then re-run with another --obj or widen the bounds")


if __name__ == "__main__":
    main()
