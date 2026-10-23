import numpy as np
from scipy.ndimage import binary_dilation


def region_iou(prediction, target, valid=None):
    valid = np.ones_like(target, dtype=bool) if valid is None else valid
    prediction, target = prediction.astype(bool) & valid, target.astype(bool) & valid
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float(intersection / union) if union else 1.0, int(intersection), int(union)


def boundary(mask):
    mask = mask.astype(bool)
    east, south, southeast = np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)
    east[:, :-1] = mask[:, 1:]
    south[:-1, :] = mask[1:, :]
    southeast[:-1, :-1] = mask[1:, 1:]
    result = (mask ^ east) | (mask ^ south) | (mask ^ southeast)
    result[-1, :] = mask[-1, :] ^ east[-1, :]
    result[:, -1] = mask[:, -1] ^ south[:, -1]
    result[-1, -1] = False
    return result


def boundary_f(prediction, target, valid=None, tolerance=0.008):
    valid = np.ones_like(target, dtype=bool) if valid is None else valid
    pred_boundary = boundary(prediction & valid)
    true_boundary = boundary(target & valid)
    radius = max(1, int(np.ceil(tolerance * np.linalg.norm(target.shape))))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = x * x + y * y <= radius * radius
    pred_dilated = binary_dilation(pred_boundary, structure=disk)
    true_dilated = binary_dilation(true_boundary, structure=disk)
    n_pred, n_true = pred_boundary.sum(), true_boundary.sum()
    if not n_pred and not n_true:
        return 1.0
    if not n_pred or not n_true:
        return 0.0
    precision = (pred_boundary & true_dilated).sum() / n_pred
    recall = (true_boundary & pred_dilated).sum() / n_true
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
