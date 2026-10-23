import numpy as np
import torch
import torch.nn.functional as F

from evog.config import require_file


class Segmenter:
    def __init__(self, cfg, device):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = require_file(cfg["checkpoint"], "SAM2")
        self.device = torch.device(device)
        self.model = build_sam2(
            cfg["config"], str(checkpoint), device=str(device), mode="eval", apply_postprocessing=False
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.predictor = SAM2ImagePredictor(self.model)

    @torch.no_grad()
    def encode(self, frame):
        self.predictor.set_image(np.asarray(frame))
        return {
            "image_embed": self.predictor._features["image_embed"].detach(),
            "high_res_feats": [f.detach() for f in self.predictor._features["high_res_feats"]],
            "size": (frame.height, frame.width),
        }

    def decode(self, features, box, points):
        coords = torch.cat((box.reshape(2, 2), points[:, :2]), dim=0).unsqueeze(0).float()
        coords = coords * self.model.image_size
        labels = torch.ones((1, len(points) + 2), dtype=torch.int32, device=self.device)
        labels[0, :2] = torch.tensor([2, 3], dtype=torch.int32, device=self.device)
        sparse, dense = self.model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        low, _, _, presence_logits = self.model.sam_mask_decoder(
            image_embeddings=features["image_embed"],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=features["high_res_feats"],
        )
        logits = F.interpolate(low.float(), size=features["size"], mode="bilinear", align_corners=False)[0, 0]
        return logits, presence_logits.float().sigmoid().reshape(())

    def __call__(self, frame, box, points):
        return self.decode(self.encode(frame), box, points)
