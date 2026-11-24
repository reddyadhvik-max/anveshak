from pathlib import Path
import torch
from anveshak import patch

CFG = str(Path(__file__).resolve().parents[1] / "configs" / "yolo11-p2.yaml")


def test_objectness_target_is_max_of_class_targets():
    patch.apply()
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel(CFG, nc=52, verbose=False)
    m.args = get_cfg()
    crit = m.init_criterion()
    seen = {}
    inner = crit.assigner.forward
    def spy(*a, **k):
        out = inner(*a, **k)
        seen["ts"] = out[2]
        return out
    crit.assigner.forward = spy
    m.train()
    img = torch.rand(2, 3, 128, 128)
    batch = {"batch_idx": torch.tensor([0., 0., 1.]), "cls": torch.tensor([[3.], [50.], [29.]]),
             "bboxes": torch.tensor([[.3, .3, .2, .2], [.7, .7, .2, .2], [.5, .5, .3, .3]]), "img": img}
    crit(m(img), batch)
    ts = seen["ts"]
    assert ts[..., 51].max() > 0
    assert torch.allclose(ts[..., 51], ts[..., :51].amax(-1))


def test_amp_check_never_downloads():
    patch.apply()
    import ultralytics.engine.trainer as tr
    assert tr.check_amp(None) is True


def test_random_init_hash_differs_per_seed():
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(0); a = patch.weights_sha256(DetectionModel(CFG, nc=52, verbose=False))
    torch.manual_seed(1); b = patch.weights_sha256(DetectionModel(CFG, nc=52, verbose=False))
    assert a != b


def test_per_image_assignment_matches_batched(monkeypatch):
    patch.apply()
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(0)
    m = DetectionModel(CFG, nc=52, verbose=False)
    m.args = get_cfg()
    m.train()
    img = torch.rand(2, 3, 128, 128)
    batch = {"batch_idx": torch.tensor([0., 0., 1.]), "cls": torch.tensor([[3.], [50.], [29.]]),
             "bboxes": torch.tensor([[.3, .3, .2, .2], [.7, .7, .2, .2], [.5, .5, .3, .3]]), "img": img}
    preds = m(img)
    full = m.init_criterion()(preds, batch)[0]
    monkeypatch.setattr(patch, "ASSIGN_LIMIT", 0)  # force the per-image path
    crit = m.init_criterion()
    per_image = crit(preds, batch)[0]
    assert crit.assigner._oom_warned  # the per-image path really ran
    assert torch.allclose(full, per_image)


def test_chunked_assignment_matches_original(monkeypatch):
    from ultralytics.utils.tal import TaskAlignedAssigner
    patch.apply()
    torch.manual_seed(0)
    na_side, stride = 32, 8
    ys, xs = torch.meshgrid(torch.arange(na_side), torch.arange(na_side), indexing="ij")
    anc = torch.stack([xs, ys], -1).view(-1, 2).float() * stride + stride / 2
    n = 300  # dense, overlapping boxes so multi-positive anchors occur
    xy = torch.rand(1, n, 2) * 240
    wh = torch.rand(1, n, 2) * 40 + 4
    gt = torch.cat([xy, xy + wh], -1)
    labels = torch.randint(0, 52, (1, n, 1)).float()
    mask = torch.ones(1, n, 1)
    mask[0, -20:] = 0  # padded rows
    half = torch.rand(len(anc), 2) * 20 + 2
    pd_boxes = torch.cat([anc - half, anc + half], -1)[None]  # boxes around each anchor: real overlaps
    pd_scores = torch.rand(1, len(anc), 52)
    a = TaskAlignedAssigner(topk=10, num_classes=52)
    monkeypatch.setattr(patch, "GT_CHUNK", 10**9)  # original path
    ref = a(pd_scores, pd_boxes, anc, labels, gt, mask)
    monkeypatch.setattr(patch, "GT_CHUNK", 64)
    out = a(pd_scores, pd_boxes, anc, labels, gt, mask)
    fg = ref[3]
    assert fg.sum() > 100 and torch.equal(fg, out[3])
    assert torch.equal(ref[4][fg], out[4][fg]) and torch.equal(ref[0][fg], out[0][fg])
    assert torch.allclose(ref[1][fg], out[1][fg]) and torch.allclose(ref[2], out[2], atol=1e-6)
