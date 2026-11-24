"""Ultralytics patches. Import and call apply() before building or training any model.

1. No pretrained weights ever: the stock AMP check downloads and runs yolo11n.pt. We skip it.
2. Class-agnostic objectness slot: for every anchor the assigner makes positive, the target for
   OBJ_SLOT is the max target over all real slots. So slot 51 learns "is this any object" from every
   labelled box of every class. That is the base signal for flagging unknowns.
3. Per-image assignment on dense batches: the assigner allocates (batch, max_objects, anchors) tensors, and with
   the P2 head at 1024 (87k anchors) and up to ~1200 boxes per tile that is several GB. Ultralytics retries
   per image only after a CUDA OOM, but Windows' sysmem fallback never raises one: it silently spills into
   system RAM and training slows ~2x. So we take the per-image path up front when the batch tensor is too big.
4. Even one image with ~1100 boxes needs ~2 GB of (boxes x 87k anchors) temporaries, which OOMed m at batch 2.
   chunked_assign() computes the identical assignment over chunks of boxes (tested against the original).
"""
import hashlib

UNK_SLOT = 50
OBJ_SLOT = 51
NC = 52
GT_CHUNK = 128  # one image with more boxes than this is assigned in chunks of boxes: memory stays ~O(chunk x anchors)
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
        if self.bs == 1 and self.n_max_boxes > GT_CHUNK and self.topk2 == self.topk and not k:
            return chunked_assign(self, pd_scores, *a)
        return orig_forward(self, pd_scores, *a, **k)

    TaskAlignedAssigner._forward = _forward
    _applied = True


def chunked_assign(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    """TaskAlignedAssigner._forward for one image, GT_CHUNK boxes at a time. Same outputs as the original:
    per anchor we keep its positive count, its single positive box, and its max-overlap box (+ their metrics)."""
    import torch
    n, na = self.n_max_boxes, pd_scores.shape[1]
    dev = pd_scores.device
    count = torch.zeros(na, dtype=torch.int32, device=dev)
    pos_idx = torch.zeros(na, dtype=torch.long, device=dev)
    pos_al, pos_ov = torch.zeros(na, device=dev), torch.zeros(na, device=dev)
    best_ov = torch.full((na,), -1.0, device=dev)
    best_idx = torch.zeros(na, dtype=torch.long, device=dev)
    best_al = torch.zeros(na, device=dev)
    ar = torch.arange(na, device=dev)
    try:
        for c0 in range(0, n, GT_CHUNK):
            sl = slice(c0, min(c0 + GT_CHUNK, n))
            self.n_max_boxes = sl.stop - c0  # get_box_metrics indexes with it
            mp, al, ov = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels[:, sl], gt_bboxes[:, sl], anc_points, mask_gt[:, sl])
            mp, al, ov = mp[0], al[0].float(), ov[0].float()  # (chunk, na)
            count += mp.sum(0, dtype=torch.int32)
            has = mp.any(0)
            j = mp.argmax(0)
            pos_idx = torch.where(has, c0 + j, pos_idx)
            pos_al = torch.where(has, al[j, ar], pos_al)
            pos_ov = torch.where(has, ov[j, ar], pos_ov)
            m, jm = ov.max(0)
            upd = m > best_ov  # strict: the earliest box wins ties, like argmax over all boxes
            best_ov, best_idx = torch.where(upd, m, best_ov), torch.where(upd, c0 + jm, best_idx)
            best_al = torch.where(upd, al[jm, ar], best_al)
    finally:
        self.n_max_boxes = n
    fg = count > 0
    multi = count > 1  # several positive boxes: the max-overlap box wins (even if it wasn't positive itself)
    gt_idx = torch.where(multi, best_idx, torch.where(fg, pos_idx, torch.zeros_like(pos_idx)))
    a_al = torch.where(multi, best_al, pos_al) * fg
    a_ov = torch.where(multi, best_ov.clamp(min=0), pos_ov) * fg
    max_al = torch.zeros(n, device=dev).scatter_reduce_(0, gt_idx, a_al, "amax")  # per box, over its anchors
    max_ov = torch.zeros(n, device=dev).scatter_reduce_(0, gt_idx, a_ov, "amax")
    norm = a_al * max_ov[gt_idx] / (max_al[gt_idx] + self.eps)
    labels, boxes, scores = self.get_targets(gt_labels, gt_bboxes, gt_idx[None], fg[None])
    return labels, boxes, scores * norm[None, :, None], fg[None], gt_idx[None]


def weights_sha256(model) -> str:
    """Hash of a state_dict. Logged at init as evidence that training starts from random weights."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
