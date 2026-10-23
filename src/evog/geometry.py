import torch
import torch.nn.functional as F


def canonical_boxes(boxes):
    low = torch.minimum(boxes[..., :2], boxes[..., 2:])
    high = torch.maximum(boxes[..., :2], boxes[..., 2:])
    return torch.cat((low, high), dim=-1).clamp(0, 1)


def box_iou(a, b, generalized=False):
    low = torch.maximum(a[..., :2], b[..., :2])
    high = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (high - low).clamp_min(0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    union = area_a + area_b - inter
    iou = inter / union.clamp_min(1e-7)
    if not generalized:
        return iou
    enclosure = (
        (torch.maximum(a[..., 2:], b[..., 2:]) - torch.minimum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    )
    return iou - (enclosure - union) / enclosure.clamp_min(1e-7)


def mask_boxes(masks):
    boxes, visible = [], []
    for mask in masks:
        y, x = torch.where(mask > 0)
        if len(x):
            h, w = mask.shape
            boxes.append(torch.stack((x.min() / w, y.min() / h, (x.max() + 1) / w, (y.max() + 1) / h)))
            visible.append(True)
        else:
            boxes.append(mask.new_zeros(4, dtype=torch.float32))
            visible.append(False)
    return torch.stack(boxes), torch.tensor(visible, device=masks.device)


def point_membership(points, masks):
    h, w = masks.shape[-2:]
    x = (points[..., 0].detach() * w).long().clamp(0, w - 1)
    y = (points[..., 1].detach() * h).long().clamp(0, h - 1)
    t = torch.arange(len(masks), device=masks.device)[:, None]
    values = masks[t, y, x]
    return (values > 0).float(), values >= 0


def gaussian_moments(points, eps=1e-4):
    xy = points[..., :2].float()
    mean = xy.mean(-2)
    centered = xy - mean.unsqueeze(-2)
    cov = centered.transpose(-1, -2) @ centered / xy.shape[-2]
    cov = cov + eps * torch.eye(2, device=xy.device, dtype=xy.dtype)
    return mean, cov


def symmetric_kl(a, b, eps=1e-4):
    ma, ca = gaussian_moments(a, eps)
    mb, cb = gaussian_moments(b, eps)
    delta = (ma - mb).unsqueeze(-1)
    trace = torch.diagonal(torch.linalg.solve(cb, ca) + torch.linalg.solve(ca, cb), dim1=-2, dim2=-1).sum(-1)
    quad = (
        (delta.transpose(-1, -2) @ (torch.linalg.solve(ca, delta) + torch.linalg.solve(cb, delta)))
        .squeeze(-1)
        .squeeze(-1)
    )
    return (0.25 * (trace + quad - 4)).clamp_min(0)


def geometric_score(boxes, points):
    xy = points[..., :2]
    return ((xy >= boxes[..., None, :2]) & (xy <= boxes[..., None, 2:])).all(-1).float().mean(-1)


def select_keyframes(boxes, points, object_scores, count=3):
    if count < 1 or len(boxes) == 0:
        raise ValueError("Keyframe selection requires frames and count >= 1")
    scores = (geometric_score(boxes, points) + object_scores.flatten().float()) / 2
    indices = torch.argsort(scores, descending=True, stable=True)[: min(count, len(scores))]
    return indices, scores


def straight_through_values(logits, hard, temperature=1.0):
    grid = torch.linspace(0, 1, logits.shape[-1], device=logits.device)
    soft = (F.softmax(logits.float() / temperature, dim=-1) * grid).sum(-1)
    return hard.float() + soft - soft.detach()
