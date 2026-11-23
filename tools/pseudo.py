"""Set B training data (PLAN 5.5): pseudo-label base classes in Set B tiles, then mix with Set A replay.

  python -m tools.pseudo --weights runs/A/weights/best.pt --data ~/data/round1/yolo/data_B.yaml \
      --replay ~/data/round1/yolo/data_A_rfs.yaml
  -> <root>/dataset_B_pseudo/{images, labels}, yolo/dataset_B_mix_train.txt, yolo/data_B_mix.yaml

Set B labels only the 10 few-shot classes, so its base objects would be trained as background (forgetting).
Per Set B train tile: base-class (0-29) detections scoring >= --hi become labels; weaker ones (>= --lo) that don't
overlap a label are blanked with grey 114, so they are neither positive nor background. Unknowns are not
pseudo-labelled: mining probes on 23 Nov showed confident "no known class" boxes are mostly missed base labels.
Train list = Set B train x --repeat + --replay-n tiles drawn from the replay list. Val = A-val + B-val.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import torch
from torchvision.ops import box_iou

from anveshak.infer import forward, postprocess
from tools.mine import read_gt

BASE = list(range(30))


def select(dets, gt, hi, lo):
    """dets: [(slot, score, x1, y1, x2, y2)] after NMS; gt: (G, 4). Returns (labels [(slot, box)], blanks [box])."""
    lab = [(c, b) for c, s, *b in dets if s >= hi]
    weak = torch.tensor([b for c, s, *b in dets if lo <= s < hi]).reshape(-1, 4)
    keep_out = torch.cat([gt, torch.tensor([b for _, b in lab]).reshape(-1, 4)])
    if len(weak) and len(keep_out):
        weak = weak[box_iou(weak, keep_out).amax(1) == 0]  # never grey out any part of a labelled object
    return lab, weak.tolist()


def main():
    import cv2
    from ultralytics import YOLO
    from ultralytics.data.utils import img2label_paths
    from ultralytics.utils import YAML
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, type=Path, help="data_B.yaml")
    ap.add_argument("--replay", required=True, type=Path, help="data yaml whose train list is replayed (RFS)")
    ap.add_argument("--hi", type=float, default=0.4)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--replay-n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, help="only the first N tiles (smoke test)")
    ap.add_argument("--all", action="store_true", help="use Set B train + val (final stage, after variants are chosen)")
    ap.add_argument("--tag", default="", help="suffix for the outputs, e.g. _all -> data_B_mix_all.yaml")
    a = ap.parse_args()
    data, replay = YAML.load(a.data), YAML.load(a.replay)
    lists = [data["train"], data["val"]] if a.all else [data["train"]]
    imgs = [Path(l.strip()) for f in lists for l in open(f) if l.strip()][: a.limit]
    txts = [Path(p) for p in img2label_paths([str(i) for i in imgs])]
    dst = imgs[0].parent.parent.with_name(imgs[0].parent.parent.name + "_pseudo" + a.tag)
    for d in ("images", "labels"):
        (dst / d).mkdir(parents=True, exist_ok=True)

    pp = {**YAML.load(Path(__file__).resolve().parent.parent / "configs" / "postproc.yaml"),
          "known_conf": a.lo, "unknown_conf": 2.0}  # unknown off
    model = YOLO(a.weights).model.float().eval().to(a.device)
    n_lab = n_blank = 0
    with torch.inference_mode():
        for n, (img, txt) in enumerate(zip(imgs, txts)):
            im = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            h, w = im.shape[:2]
            gt, rows = read_gt(txt, w, h)
            lab, blank = select(postprocess(forward(model, img, a.device), BASE, pp), gt, a.hi, a.lo)
            out = dst / "images" / img.name
            out.unlink(missing_ok=True)  # reruns must not write through a hard link into the original
            if blank:
                for x1, y1, x2, y2 in blank:
                    im[max(int(y1), 0):int(y2) + 1, max(int(x1), 0):int(x2) + 1] = 114
                cv2.imwrite(str(out), im)
            else:
                try:
                    os.link(img, out)
                except OSError:
                    shutil.copy2(img, out)
            lines = [" ".join(r) for r in rows] + [
                f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                for c, (x1, y1, x2, y2) in lab]
            (dst / "labels" / txt.name).write_text("\n".join(lines) + ("\n" if lines else ""))
            n_lab, n_blank = n_lab + len(lab), n_blank + len(blank)
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(imgs)} tiles", flush=True)
    print(f"pseudo-labelled {n_lab} base boxes, blanked {n_blank} weak ones in {len(imgs)} Set B train tiles "
          f"({n_lab / len(imgs):.1f} / {n_blank / len(imgs):.1f} per tile)")

    rep = [l.strip() for l in open(replay["train"]) if l.strip()]
    train = [str(dst / "images" / i.name) for i in imgs] * a.repeat + random.Random(a.seed).sample(rep, min(a.replay_n, len(rep)))
    yolo = Path(data["train"]).parent
    lst = yolo / f"dataset_B_mix{a.tag}_train.txt"
    lst.write_text("\n".join(train) + "\n")
    vals = [l for v in (replay["val"], data["val"]) for l in open(v) if l.strip()]
    (yolo / "dataset_AB_val.txt").write_text("".join(vals))
    YAML.save(a.data.with_name(f"data_B_mix{a.tag}.yaml"),
              {**data, "train": str(lst), "val": str(yolo / "dataset_AB_val.txt")})
    print(f"train {len(train)} samples ({len(imgs)} B x{a.repeat} + {len(train) - len(imgs) * a.repeat} A replay), "
          f"val {len(vals)} tiles; wrote data_B_mix{a.tag}.yaml")


if __name__ == "__main__":
    main()
