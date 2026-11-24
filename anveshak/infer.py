"""Checkpoint -> submission CSV, with our own open-world post-processing (not Ultralytics' NMS).

  python -m anveshak.infer --weights runs/B/weights/best.pt --images ~/data/round1/test/images \
      --tiles ~/data/round1/test/tiles.csv --classes ~/data/round1/class_list.csv --out preds.csv

The raw head gives 52 sigmoid scores per anchor (see patch.py):
  known   = slots whose class_id appears in --classes (0-39 in R1, plus 40-49 in R2); per-class NMS, multi-label
  unknown = max(s[50], s[51] * (1 - max known)): the unknown slot, or "objectness without a known class"
Then the competition's conflict rule is applied by us, so the scorer has nothing left to drop.
Thresholds live in configs/postproc.yaml (tuned on val OE, shipped with the weights). FP32, no TTA: deterministic.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou, nms

from anveshak.patch import OBJ_SLOT, UNK_SLOT
from tools.make_csv import validate, write_csv
from tools.oe_score import UNKNOWN, apply_conflict_rule

REPO = Path(__file__).resolve().parent.parent


def load_image(path):
    from ultralytics.utils.patches import imread  # same reader as the training dataloader
    img = imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, -1)
    return torch.from_numpy(np.ascontiguousarray(img[..., :3][..., ::-1].transpose(2, 0, 1))).float() / 255  # BGR->RGB like Format


def forward(model, path, device):
    """Raw head output (4+52, N) for one tile at native resolution (padded to a stride multiple, never resized)."""
    img = load_image(path)
    stride = int(model.stride.max())
    h, w = img.shape[1:]
    img = torch.nn.functional.pad(img, (0, -w % stride, 0, -h % stride), value=114 / 255)
    return model(img[None].to(device))[0][0].float().cpu()


def postprocess(out, slots, pp):
    """out: (4+52, N) raw head output for one image (xywh px, sigmoid scores). slots: active known slot ids.
    Returns [(slot or UNK_SLOT, score, x1, y1, x2, y2)]."""
    xywh, s = out[:4].T, out[4:].T
    boxes = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    sk = s[:, slots]
    dets = []
    i, j = (sk > pp["known_conf"]).nonzero(as_tuple=True)  # multi-label: one candidate per (anchor, class)
    sc = sk[i, j]
    if len(sc) > pp["max_candidates"]:
        sc, top = sc.topk(pp["max_candidates"])
        i, j = i[top], j[top]
    keep = batched_nms(boxes[i], sc, j, pp["known_iou"])[: pp["max_known"]]
    for a, c, v in zip(i[keep].tolist(), j[keep].tolist(), sc[keep].tolist()):
        dets.append((slots[c], v, *boxes[a].tolist()))
    kmax = sk.amax(1) if len(slots) else torch.zeros(len(s))
    su = torch.maximum(s[:, UNK_SLOT], s[:, OBJ_SLOT] * (1 - kmax))
    side2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    a = ((su > pp["unknown_conf"]) & (side2 >= pp.get("unknown_min_side", 0) ** 2)).nonzero(as_tuple=True)[0]
    a = a[nms(boxes[a], su[a], pp["unknown_iou"])]
    km = pp.get("unknown_known_max", 1.0)
    strong = [d[2:] for d in dets if d[1] >= km]
    if km < 1.0 and strong and len(a):  # known wins: the conflict rule would otherwise delete that known box
        a = a[box_iou(boxes[a], torch.tensor(strong)).amax(1) < 0.5]
    for k in a[: pp["max_unknown"]].tolist():
        dets.append((UNK_SLOT, su[k].item(), *boxes[k].tolist()))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--tiles", required=True, help="tiles.csv (tile_id,width,height)")
    ap.add_argument("--classes", required=True, nargs="+", help="class_list.csv files; their class_ids are the active known slots")
    ap.add_argument("--subset", help="optional txt of tile ids (e.g. a val split) instead of every tile in --tiles")
    ap.add_argument("--postproc", default=str(REPO / "configs" / "postproc.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[], help="override postproc values, e.g. unknown_conf=0.4")
    ap.add_argument("--raw", action="store_true", help="skip the conflict rule (for oe_score --sweep, which applies it per threshold)")
    ap.add_argument("--validator", help="validate_predictions.py; run it on the CSV when given")
    a = ap.parse_args()
    a.device = f"cuda:{a.device}" if a.device.isdigit() else a.device  # Ultralytics-style --device 0

    from ultralytics.utils import YAML
    pp = YAML.load(a.postproc)
    pp.update({k: float(v) if "." in v else int(v) for k, v in (kv.split("=", 1) for kv in a.set)})
    names = {}
    for p in a.classes:
        names.update({int(r["class_id"]): r["label"] for r in csv.DictReader(open(p, encoding="utf-8")) if r["label"] != UNKNOWN})
    slots = sorted(names)
    sizes = {r["tile_id"]: (float(r["width"]), float(r["height"])) for r in csv.DictReader(open(a.tiles, encoding="utf-8"))}
    tiles = [l.strip() for l in open(a.subset) if l.strip()] if a.subset else sorted(sizes)

    from ultralytics import YOLO
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = YOLO(a.weights).model.float().eval().to(a.device)
    rows = []
    with torch.inference_mode():
        for n, t in enumerate(tiles):
            out = forward(model, next(a.images.glob(f"{t}.*")), a.device)
            for slot, s, x1, y1, x2, y2 in postprocess(out, slots, pp):
                rows.append({"tile": t, "label": UNKNOWN if slot == UNK_SLOT else names[slot], "score": s,
                             "box": [x1, y1, x2, y2], "row": len(rows)})
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(tiles)} tiles", flush=True)
    if not a.raw:
        rows = apply_conflict_rule(rows)
    n = write_csv(((r["tile"], r["label"], r["score"], *r["box"]) for r in rows), a.out, sizes)
    print(f"wrote {n} rows for {len(tiles)} tiles to {a.out}")
    if a.validator:
        validate(a.out, a.validator, a.tiles, a.classes)


if __name__ == "__main__":
    main()
