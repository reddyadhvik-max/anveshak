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


def test_mine_select():
    from tools.mine import select
    pp = dict(obj=0.5, known_max=0.2, iou_gt=0.3, max_area=0.25, tile_area=1024 ** 2, max_per_tile=100,
              min_side=4, max_side=200)
    gt = torch.tensor([[40., 40., 60., 60.]])
    dets = select(head([
        (300, 300, 20, 20, {OBJ_SLOT: 0.9}),            # mined
        (302, 300, 20, 20, {OBJ_SLOT: 0.8}),            # duplicate: NMS
        (50, 50, 20, 20, {OBJ_SLOT: 0.9}),              # on a GT box
        (500, 500, 20, 20, {OBJ_SLOT: 0.9, 3: 0.6}),    # looks like a known class
        (700, 700, 20, 20, {OBJ_SLOT: 0.3}),            # not confident
        (512, 512, 600, 600, {OBJ_SLOT: 0.9}),          # too big
    ]), gt, pp)
    assert dets[:, :4].tolist() == [[290.0, 290.0, 310.0, 310.0]]


def test_pick_threshold():
    from tools.oe_score import pick
    res = [(0.1, {"U-Prec": 0.3, "U-F1": 0.40}), (0.3, {"U-Prec": 0.55, "U-F1": 0.45}), (0.5, {"U-Prec": 0.7, "U-F1": 0.35})]
    assert pick(res, 0.5) == 0.3      # lowest threshold reaching the precision target (mining)
    assert pick(res, 0.9) is None     # none reach it: refuse, no silent fallback
    assert pick(res) == 0.3           # best U-F1 (submission)


def test_pipeline_placeholders(tmp_path, monkeypatch):
    from anveshak import pipeline
    monkeypatch.setenv("ANVESHAK_DATA", "/data")
    (pipeline.REPO / "runs" / "eval").mkdir(parents=True, exist_ok=True)
    f = pipeline.REPO / "runs" / "eval" / "_t.txt"
    f.write_text("0.4\n")
    try:
        assert pipeline.expand(["python", "{D}/x", "--obj", "{read:runs/eval/_t.txt}", "k={read:runs/eval/_t.txt}"])[1:] == \
            ["/data/x", "--obj", "0.4", "k=0.4"]
    finally:
        f.unlink()
    assert len(pipeline.load_steps()) >= 13


def test_pseudo_select():
    import torch
    from tools.pseudo import select
    gt = torch.tensor([[0., 0, 10, 10]])
    dets = [(3, 0.9, 100, 100, 120, 120),   # confident -> label
            (4, 0.2, 200, 200, 220, 220),   # weak, free -> blank
            (5, 0.2, 5, 5, 15, 15),         # weak but touches GT -> left alone
            (6, 0.2, 110, 110, 130, 130),   # weak but touches a pseudo-label -> left alone
            (7, 0.05, 300, 300, 310, 310)]  # below lo (postprocess already drops these) -> ignored
    lab, blank = select(dets, gt, hi=0.4, lo=0.1)
    assert lab == [(3, [100, 100, 120, 120])]
    assert blank == [[200., 200., 220., 220.]]


def test_copypaste_transform_tracks_box():
    import random
    import numpy as np
    from tools.copypaste import free, transform
    crop = np.zeros((40, 60, 3), np.uint8)
    crop[10:20, 5:25] = 255  # the object, box (5, 10, 25, 20)
    for seed in range(20):
        c, (x1, y1, x2, y2) = transform(crop, (5, 10, 25, 20), random.Random(seed))
        inside = c[y1:y2, x1:x2, 0]
        assert inside.mean() > 200 and (c[..., 0] > 128).sum() <= inside.size * 1.2  # box = the white object (bilinear edges blur ~1 px)
    assert free((0, 0, 10, 10), [(10, 0, 20, 10)]) and not free((0, 0, 11, 10), [(10, 0, 20, 10)])


def test_copypaste_feather():
    import numpy as np
    from tools.copypaste import feather
    al = feather(np.zeros((30, 40, 3)), (10, 5, 30, 25))
    assert (al[5:25, 10:30] == 1).all()                       # the object itself is pasted as is
    assert al[0, 0] < 0.15 and al[:, 0].max() < 0.1 and al[0].max() < 0.15  # edges fade into the background


def test_pick_variant_choose():
    from tools.pick_variant import choose
    r = {"B": (0.2517, 0.0786), "B2": (0.2400, 0.1200), "B3": (0.2490, 0.0950)}
    assert choose(r, 0.01) == "B3"     # B2 is best on few-shot but lost too much on base classes
    assert choose(r, 0.02) == "B2"
