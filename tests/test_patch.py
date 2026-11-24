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
