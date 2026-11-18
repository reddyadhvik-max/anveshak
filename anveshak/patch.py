"""Ultralytics patches. Import and call apply() before building or training any model.

1. No pretrained weights ever: the stock AMP check downloads and runs yolo11n.pt. We skip it.
2. Class-agnostic objectness slot: for every anchor the assigner makes positive, the target for
   OBJ_SLOT is the max target over all real slots. So slot 51 learns "is this any object" from every
   labelled box of every class. That is the base signal for flagging unknowns.
3. Per-image assignment on dense batches: the assigner allocates (batch, max_objects, anchors) tensors, and with
   the P2 head at 1024 (87k anchors) and up to ~1200 boxes per tile that is several GB. Ultralytics retries
   per image only after a CUDA OOM, but Windows' sysmem fallback never raises one: it silently spills into
   system RAM and training slows ~2x. So we take the per-image path up front when the batch tensor is too big.
"""
import hashlib

UNK_SLOT = 50
OBJ_SLOT = 51
NC = 52
ASSIGN_LIMIT = 50_000_000  # batch*max_objects*anchors above this -> assign one image at a time (~200 MB per fp32 tensor)

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

    from ultralytics.utils.tal import TaskAlignedAssigner
    orig_forward = TaskAlignedAssigner._forward

    def _forward(self, pd_scores, *a, **k):
        if self.bs > 1 and self.bs * self.n_max_boxes * pd_scores.shape[1] > ASSIGN_LIMIT:
            raise RuntimeError("out of memory (anveshak: dense batch, assigning per image)")  # -> Ultralytics' per-image path
        return orig_forward(self, pd_scores, *a, **k)

    TaskAlignedAssigner._forward = _forward
    _applied = True


def weights_sha256(model) -> str:
    """Hash of a state_dict. Logged at init as evidence that training starts from random weights."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
