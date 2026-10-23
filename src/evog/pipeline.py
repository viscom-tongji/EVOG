import heapq

import numpy as np
import torch
from torch import nn

from evog.geometry import geometric_score
from evog.grounder import Grounder
from evog.losses import bounding_box_loss, consistency_loss, mask_loss, point_calibration
from evog.segmenter import Segmenter


class TrainingModel(nn.Module):
    def __init__(self, cfg, device, adapter=None):
        super().__init__()
        self.cfg = cfg
        self.grounder = Grounder(cfg["model"], device, trainable=True, adapter=adapter)
        self.segmenter = Segmenter(cfg["sam2"], device)

    def forward(self, sample):
        frames, query = sample["frames"], sample["query"]
        masks = torch.from_numpy(sample["masks"]).to(self.grounder.device)
        original = self.grounder(frames, query, self.cfg["method"]["train_refinements"], differentiable=True)
        auxiliary = None
        if self.cfg["method"]["query_alignment"]:
            auxiliary_query = self.grounder.auxiliary_query(frames, query, sample["masks"])
            auxiliary = self.grounder(
                frames, auxiliary_query, self.cfg["method"]["train_refinements"], differentiable=True
            )
        predictions = [original] + ([auxiliary] if auxiliary is not None else [])
        box_losses, point_losses, mask_losses = [], [], []
        features = [self.segmenter.encode(frame) for frame in frames]
        for prediction in predictions:
            box_losses.append(
                bounding_box_loss(
                    prediction.boxes, masks, self.cfg["loss"]["box_l1"], self.cfg["loss"]["box_giou"]
                )
            )
            point_losses.append(point_calibration(prediction.history, masks))
            logits = torch.stack(
                [
                    self.segmenter.decode(feature, prediction.boxes[i], prediction.points[i])[0]
                    for i, feature in enumerate(features)
                ]
            )
            mask_losses.append(mask_loss(logits, masks))
        losses = {
            "bbox": torch.stack(box_losses).mean(),
            "point": torch.stack(point_losses).mean(),
            "mask": torch.stack(mask_losses).mean(),
        }
        if auxiliary is not None:
            losses["consistency_box"], losses["consistency_point"] = consistency_loss(
                original, auxiliary, self.cfg["loss"]["covariance_epsilon"]
            )
        else:
            losses["consistency_box"] = original.boxes.sum() * 0
            losses["consistency_point"] = original.points.sum() * 0
        consistency = (
            self.cfg["loss"]["align_box"] * losses["consistency_box"]
            + self.cfg["loss"]["align_point"] * losses["consistency_point"]
        )
        losses["total"] = (
            losses["bbox"]
            + losses["point"]
            + losses["mask"]
            + self.cfg["loss"]["consistency_weight"] * consistency
        )
        return losses


class InferencePipeline:
    def __init__(self, cfg, device, adapter=None, base_only=False):
        if not adapter and not base_only:
            raise ValueError(
                "Supply a trained --adapter, or explicitly use --base-only for unadapted inference"
            )
        self.cfg = cfg
        self.device = torch.device(device)
        self.grounder = Grounder(cfg["model"], device, trainable=False, adapter=adapter)
        self.segmenter = Segmenter(cfg["sam2"], device)
        self.propagator = None

    @torch.no_grad()
    def __call__(self, frames, query):
        if not frames or not query.strip():
            raise ValueError("Inference needs frames and a nonempty query")
        chunk = self.cfg["inference"]["chunk_frames"]
        if chunk < 1:
            raise ValueError("inference.chunk_frames must be positive")
        overview = None
        if len(frames) > chunk:
            ids = (
                np.linspace(0, len(frames) - 1, min(len(frames), self.cfg["inference"]["overview_frames"]))
                .round()
                .astype(int)
            )
            overview = [frames[i] for i in ids]
        boxes, points, dialogues = [], [], []
        for start in range(0, len(frames), chunk):
            prediction = self.grounder(
                frames[start : start + chunk],
                query,
                self.cfg["method"]["test_refinements"],
                overview=overview,
            )
            boxes.append(prediction.boxes)
            points.append(prediction.points)
            dialogues.append({"first_frame": start, "messages": prediction.dialogue})
        boxes, points = torch.cat(boxes), torch.cat(points)
        geometry = geometric_score(boxes, points)
        candidates = []
        presence, final_scores = [], []
        count = min(self.cfg["method"]["keyframes"], len(frames))
        for i, frame in enumerate(frames):
            logits, score = self.segmenter(frame, boxes[i], points[i])
            s1, s2 = float(score), float(geometry[i])
            mode = self.cfg["method"]["selection"]
            final = (s1 + s2) / 2 if mode == "combined" else s1 if mode == "presence" else s2
            presence.append(s1)
            final_scores.append(final)
            item = (final, -i, (logits > 0).cpu().numpy())
            if len(candidates) < count:
                heapq.heappush(candidates, item)
            elif item[:2] > candidates[0][:2]:
                heapq.heapreplace(candidates, item)
        anchors = {-entry[1]: entry[2] for entry in candidates}
        if len(frames) == 1:
            masks = np.stack([anchors[0]])
        else:
            from evog.propagation import Propagator

            if self.propagator is None:
                self.propagator = Propagator(self.cfg["cutie"], self.device)
            masks = self.propagator(frames, anchors)
        trace = {
            "query": query,
            "boxes": boxes.cpu().tolist(),
            "points": points.cpu().tolist(),
            "presence_scores": presence,
            "geometric_scores": geometry.cpu().tolist(),
            "keyframe_scores": final_scores,
            "keyframes": sorted(anchors),
            "dialogues": dialogues,
        }
        return masks, trace
