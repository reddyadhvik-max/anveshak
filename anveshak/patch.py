"""Ultralytics patches. Import and call apply() before building or training any model.

1. No pretrained weights ever: the stock AMP check downloads and runs yolo11n.pt. We skip it.
2. Class-agnostic objectness slot: for every anchor the assigner makes positive, the target for
   OBJ_SLOT is the max target over all real slots. So slot 51 learns "is this any object" from every
   labelled box of every class. That is the base signal for flagging unknowns.
"""
import hashlib

import torch

UNK_SLOT = 50
OBJ_SLOT = 51
NC = 52

_applied = False


def apply():
    global _applied
    if _applied:
        return
    import ultralytics.engine.trainer as trainer
    from ultralytics.utils import loss as L

    trainer.check_amp = lambda model: True  # stock check downloads yolo11n.pt; rules forbid any pretrained weights

    orig_init = L.v8DetectionLoss.__init__

    def init(self, model, *a, **k):
        orig_init(self, model, *a, **k)
        if self.nc != NC:
            return
        assign = self.assigner.forward

        def assign_with_obj(*args, **kw):
            out = list(assign(*args, **kw))
            ts = out[2]  # target_scores (b, anchors, nc)
            ts[..., OBJ_SLOT] = ts[..., :OBJ_SLOT].amax(-1)
            return tuple(out)

        self.assigner.forward = assign_with_obj

    L.v8DetectionLoss.__init__ = init
    _applied = True


def weights_sha256(model) -> str:
    """Hash of a state_dict. Logged at init as evidence that training starts from random weights."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
