import torch

from anveshak.infer import postprocess
from anveshak.patch import OBJ_SLOT, UNK_SLOT

PP = dict(known_conf=0.001, known_iou=0.6, max_candidates=30000, max_known=900,
          unknown_conf=0.3, unknown_iou=0.5, max_unknown=100)


def head(anchors):
    """anchors: list of (cx, cy, w, h, {slot: score}) -> raw (4+52, N) head output."""
    out = torch.zeros(56, len(anchors))
    for n, (cx, cy, w, h, sc) in enumerate(anchors):
        out[:4, n] = torch.tensor([cx, cy, w, h])
        for k, v in sc.items():
            out[4 + k, n] = v
    return out


def test_known_nms_unknown_from_objectness_and_inactive_slots():
    dets = postprocess(head([
        (50, 50, 20, 20, {3: 0.9}),                 # known class 3
        (51, 50, 20, 20, {3: 0.8}),                 # duplicate of it: NMS drops
        (200, 200, 30, 30, {OBJ_SLOT: 0.8}),        # object but no known class -> unknown 0.8
        (400, 400, 30, 30, {OBJ_SLOT: 0.9, 5: 0.9}),  # object that is known -> unknown score 0.09, not emitted
        (600, 600, 30, 30, {UNK_SLOT: 0.5}),        # unknown slot directly
        (800, 800, 30, 30, {45: 0.9}),              # R2 slot, inactive in R1
    ]), slots=list(range(40)), pp=PP)
    known = [d for d in dets if d[0] != UNK_SLOT]
    unk = sorted((round(d[1], 3), d[2]) for d in dets if d[0] == UNK_SLOT)
    assert [(d[0], round(d[1], 3)) for d in known] == [(3, 0.9), (5, 0.9)]
    assert known[0][2:] == (40.0, 40.0, 60.0, 60.0)  # xywh -> xyxy
    assert unk == [(0.5, 585.0), (0.8, 185.0)]


def test_repeat_factors():
    from tools.rfs import repeat_factors
    tiles = [{0}] * 98 + [{0, 1}, set()]  # class 1 is in 1% of tiles
    r = repeat_factors(tiles, thr=0.04)
    assert r[0] == 1.0 and r[-1] == 1.0 and abs(r[98] - 2.0) < 1e-9  # sqrt(0.04 / 0.01)
