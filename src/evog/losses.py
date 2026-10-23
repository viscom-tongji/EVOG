import torch
import torch.nn.functional as F

from evog.geometry import box_iou, mask_boxes, point_membership, symmetric_kl


def point_calibration(history, masks):
    terms = []
    for points in history:
        membership, valid = point_membership(points, masks)
        confidence = points[..., 2].float()
        loss = confidence * (1 - membership) + (1 - confidence) * membership
        terms.append((loss * valid).sum() / valid.sum().clamp_min(1))
    if not terms:
        raise ValueError("Point calibration requires at least one prediction")
    return torch.stack(terms).mean()


def bounding_box_loss(boxes, masks, l1_weight=5.0, giou_weight=2.0):
    targets, visible = mask_boxes(masks)
    if not visible.any():
        return boxes.sum() * 0
    l1 = F.l1_loss(boxes[visible].float(), targets[visible].float(), reduction="none").sum(-1).mean()
    giou = 1 - box_iou(boxes[visible].float(), targets[visible].float(), generalized=True)
    return l1_weight * l1 + giou_weight * giou.mean()


def mask_loss(logits, targets):
    logits = logits.float()
    targets = targets.float()
    valid = targets >= 0
    truth = targets.clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    bce = (bce * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
    prob = logits.sigmoid() * valid
    truth = truth * valid
    dice = 1 - (2 * (prob * truth).flatten(1).sum(1) + 1) / (
        prob.flatten(1).sum(1) + truth.flatten(1).sum(1) + 1
    )
    return (bce + dice).mean()


def consistency_loss(original, auxiliary, eps=1e-4):
    box = (1 - box_iou(original.boxes.float(), auxiliary.boxes.float())).mean()
    point = symmetric_kl(original.points, auxiliary.points, eps).mean()
    return box, point
