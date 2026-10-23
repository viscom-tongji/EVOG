from pathlib import Path
import inspect
import copy
import math

import numpy as np
import torch

from evog.config import require_file


def load_cutie_config():
    from cutie.model.cutie import CUTIE
    from cutie.inference.utils.args_utils import get_dataset_cfg
    from omegaconf import OmegaConf

    directory = Path(inspect.getfile(CUTIE)).resolve().parents[1] / "config"
    config = OmegaConf.load(directory / "eval_config.yaml")
    config.pop("defaults", None)
    config.pop("hydra", None)
    config.model = OmegaConf.load(directory / "model" / "base.yaml")
    config.dataset = "generic"
    get_dataset_cfg(config)
    return config


class Propagator:
    def __init__(self, cfg, device):
        from cutie.model.cutie import CUTIE
        from omegaconf import open_dict

        self.device = torch.device(device)
        require_file(cfg["checkpoint"], "Cutie")
        for filename in ("resnet18-5c106cde.pth", "resnet50-19c8e357.pth"):
            require_file(Path(cfg["resnet_cache"]) / filename, "Cutie ResNet initialization")
        config = load_cutie_config()
        with open_dict(config):
            config.model.resnet_model_path = cfg["resnet_cache"]
            config.max_internal_size = cfg["max_internal_size"]
            config.mem_every = cfg["mem_every"]
        self.config = config
        self.model = CUTIE(config).to(device).eval()
        state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
        self.model.load_weights(state)
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, frames, anchors):
        from cutie.inference.inference_core import InferenceCore

        if not anchors:
            raise ValueError("Cutie requires at least one keyframe")
        count = len(frames)
        h, w = frames[0].height, frames[0].width
        config = copy.deepcopy(self.config)
        scale = min(1.0, config.max_internal_size / min(h, w)) if config.max_internal_size > 0 else 1.0
        tokens = math.ceil(int(h * scale) / 16) * math.ceil(int(w * scale) / 16)
        if config.top_k is not None:
            config.top_k = min(config.top_k, tokens)
        if any(index < 0 or index >= count for index in anchors):
            raise ValueError("Anchor index outside the video")
        anchors = {int(k): np.asarray(v, dtype=np.float32) for k, v in anchors.items()}
        if any(mask.shape != (h, w) for mask in anchors.values()):
            raise ValueError("Anchor mask resolution differs from video")
        forward, backward = {}, {}
        sorted_anchors = sorted(anchors)
        for direction, outputs in ((1, forward), (-1, backward)):
            start = sorted_anchors[0] if direction == 1 else sorted_anchors[-1]
            stop = count if direction == 1 else -1
            core = InferenceCore(self.model, cfg=config)
            active = False
            for index in range(start, stop, direction):
                image = (
                    torch.from_numpy(np.asarray(frames[index]).copy())
                    .permute(2, 0, 1)
                    .float()
                    .to(self.device)
                    / 255
                )
                if index in anchors:
                    mask = torch.from_numpy((anchors[index] > 0.5).astype(np.int64)).to(self.device)
                    if not mask.any():
                        core = InferenceCore(self.model, cfg=config)
                        active = False
                        outputs[index] = anchors[index]
                        continue
                    probabilities = core.step(image, mask, objects=[1], idx_mask=True)
                    active = True
                    outputs[index] = anchors[index]
                elif active:
                    probabilities = core.step(image, end=index + direction == stop)
                    outputs[index] = probabilities[1].float().cpu().numpy()
                else:
                    outputs[index] = np.zeros((h, w), dtype=np.float32)
            core.clear_memory()
        result = []
        for index in range(count):
            if index in anchors:
                mask = anchors[index]
            elif index not in forward:
                mask = backward[index]
            elif index not in backward:
                mask = forward[index]
            else:
                left = max(k for k in sorted_anchors if k < index)
                right = min(k for k in sorted_anchors if k > index)
                alpha = (right - index) / (right - left)
                mask = alpha * forward[index] + (1 - alpha) * backward[index]
            result.append(mask > 0.5)
        return np.stack(result)
