"""Local re-implementation of the Anveshak OE-Score, written from the published rules (brief p.6-7).

  conflict rule -> mAP_K (VOC all-point AP @ IoU 0.5 over known classes with GT)
               -> U-Rec / U-Prec / U-F1 -> OE = 100 * HM(mAP_K, U-F1)

Assumptions (the brief does not pin these down):
  - VOC-style matching: a prediction is compared with the highest-IoU GT of its class; if that GT is
    already taken, the prediction is a FP (no fall-back to the next GT).
  - IoU uses continuous coordinates (no +1 pixel).
  - Conflict losers are found against the original set (all pairs at once), then removed.

Usage (dev-sim: treat some known classes as unknown):
  python tools/oe_score.py preds.csv --labels <dir_with_jsons> --tiles val_tiles.txt \
      --classes class_list.csv [--as-unknown 3 17 25]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

UNKNOWN = "unknown"
IOU_T = 0.5


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-12)


def apply_conflict_rule(preds):
    """preds: list of dicts (tile,label,score,box,row). Drops the lower-scored side of every
    known-vs-unknown pair with IoU >= 0.5; ties go to the earlier row."""
    by_tile = defaultdict(list)
    for p in preds:
        by_tile[p["tile"]].append(p)
    drop = set()
    for ps in by_tile.values():
        k = [p for p in ps if p["label"] != UNKNOWN]
        u = [p for p in ps if p["label"] == UNKNOWN]
        if not k or not u:
            continue
        m = iou_matrix([p["box"] for p in k], [p["box"] for p in u])
        for i, j in zip(*np.nonzero(m >= IOU_T)):
            a, b = k[i], u[j]
            win_a = a["score"] > b["score"] or (a["score"] == b["score"] and a["row"] < b["row"])
            drop.add((b if win_a else a)["row"])
    return [p for p in preds if p["row"] not in drop]


def greedy_match(preds, gts):
    """preds of one label over all tiles; gts: tile -> list of boxes. Returns tp flags (score order)."""
    preds = sorted(preds, key=lambda p: (-p["score"], p["row"]))
    used = {t: np.zeros(len(b), bool) for t, b in gts.items()}
    tp = []
    for p in preds:
        g = gts.get(p["tile"], [])
        if len(g) == 0:
            tp.append(0)
            continue
        ious = iou_matrix([p["box"]], g)[0]
        j = int(ious.argmax())
        if ious[j] >= IOU_T and not used[p["tile"]][j]:
            used[p["tile"]][j] = True
            tp.append(1)
        else:
            tp.append(0)
    return tp


def voc_ap(tp, n_gt):
    if n_gt == 0:
        return 0.0
    tp = np.asarray(tp, float)
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-12)
    mrec = np.r_[0.0, rec, 1.0]
    mpre = np.r_[0.0, prec, 0.0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def score(preds, gts, known_labels):
    """preds: list of dicts tile,label,score,box,row. gts: list of (tile,label,box), label may be 'unknown'.
    Returns dict with mAP_K, per-class AP, U-Rec, U-Prec, U-F1, OE."""
    preds = apply_conflict_rule(preds)
    gt_by = defaultdict(lambda: defaultdict(list))
    for t, l, b in gts:
        gt_by[l][t].append(b)
    pred_by = defaultdict(list)
    for p in preds:
        pred_by[p["label"]].append(p)

    ap = {}
    for c in known_labels:
        n = sum(len(v) for v in gt_by[c].values())
        if n:  # classes with no GT are left out, as in the official scorer
            ap[c] = voc_ap(greedy_match(pred_by[c], gt_by[c]), n)
    map_k = float(np.mean(list(ap.values()))) if ap else 0.0

    n_ugt = sum(len(v) for v in gt_by[UNKNOWN].values())
    u_tp = sum(greedy_match(pred_by[UNKNOWN], gt_by[UNKNOWN]))
    u_rec = u_tp / n_ugt if n_ugt else 0.0
    u_prec = u_tp / len(pred_by[UNKNOWN]) if pred_by[UNKNOWN] else 0.0
    u_f1 = 2 * u_rec * u_prec / (u_rec + u_prec) if u_rec + u_prec else 0.0
    oe = 100 * 2 * map_k * u_f1 / (map_k + u_f1) if map_k + u_f1 else 0.0
    return {"mAP_K": map_k, "U-Rec": u_rec, "U-Prec": u_prec, "U-F1": u_f1, "OE": oe,
            "AP": ap, "n_unknown_gt": n_ugt, "n_unknown_pred": len(pred_by[UNKNOWN]),
            "skipped_classes": sorted(set(known_labels) - set(ap))}


def pick(res, min_prec=None):
    """res: [(threshold, score dict)]. Lowest threshold reaching min_prec (else the most precise one), or best U-F1."""
    if min_prec is not None:
        ok = [t for t, r in res if r["U-Prec"] >= min_prec]
        return min(ok) if ok else max(res, key=lambda tr: (tr[1]["U-Prec"], tr[0]))[0]
    return max(res, key=lambda tr: (tr[1]["U-F1"], tr[0]))[0]


def read_preds(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{"tile": r["image_id"], "label": r["class_name"], "score": float(r["score"]),
                 "box": [float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")], "row": i}
                for i, r in enumerate(csv.DictReader(f))]


def read_gts(label_dir, tiles, as_unknown=()):
    as_unknown = {int(c) for c in as_unknown}
    gts = []
    for t in tiles:
        for a in json.load(open(Path(label_dir) / f"{t}.json"))["annotations"]:
            gts.append((t, UNKNOWN if a["class_id"] in as_unknown or a["label"] == UNKNOWN else a["label"], a["bbox"]))
    return gts


def read_known(class_files, as_unknown=()):
    out = []
    for p in class_files:
        with open(p, newline="", encoding="utf-8") as f:
            out += [r["label"] for r in csv.DictReader(f)
                    if r["label"] != UNKNOWN and int(r["class_id"]) not in {int(c) for c in as_unknown}]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tiles", required=True, help="txt with one tile_id per line")
    ap.add_argument("--classes", required=True, nargs="+")
    ap.add_argument("--as-unknown", nargs="*", default=[])
    ap.add_argument("--pick-out", help="write the chosen threshold here")
    ap.add_argument("--pick-prec", type=float, help="choose the lowest threshold with U-Prec >= this (mining); default: best U-F1")
    ap.add_argument("--sweep", nargs="*", type=float, help="unknown score thresholds to try (preds from infer --raw)")
    a = ap.parse_args()
    tiles = [l.strip() for l in open(a.tiles) if l.strip()]
    preds, gts, known = read_preds(a.preds), read_gts(a.labels, tiles, a.as_unknown), read_known(a.classes, a.as_unknown)
    if a.sweep:
        print("unk_thr  mAP_K   U-Rec   U-Prec  U-F1    OE      n_unk")
        res = []
        for t in a.sweep:
            r = score([p for p in preds if p["label"] != UNKNOWN or p["score"] >= t], gts, known)
            res.append((t, r))
            print(f"{t:<8.2f} {r['mAP_K']:.4f}  {r['U-Rec']:.4f}  {r['U-Prec']:.4f}  {r['U-F1']:.4f}  {r['OE']:6.2f}  {r['n_unknown_pred']}")
        if a.pick_out:
            t = pick(res, a.pick_prec)
            Path(a.pick_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.pick_out).write_text(f"{t}\n")
            print(f"picked {t} -> {a.pick_out}")
        return
    r = score(preds, gts, known)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "AP"}, indent=1))


if __name__ == "__main__":
    main()
