import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from evog.codec import SpatialCodec
from evog.config import require_file
from evog.prompts import AUXILIARY, INITIAL, REFINE


@dataclass
class SpatialPrediction:
    boxes: torch.Tensor
    points: torch.Tensor
    history: list
    dialogue: list


class Grounder(nn.Module):
    def __init__(self, cfg, device, trainable=False, adapter=None):
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg = cfg
        self.device = torch.device(device)
        dtype = getattr(torch, cfg["dtype"])
        if self.device.type == "cpu":
            dtype = torch.float32
        self.processor = AutoProcessor.from_pretrained(
            cfg["base_model"],
            min_pixels=cfg["min_pixels"],
            max_pixels=cfg["max_pixels"],
            local_files_only=True,
        )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["base_model"], torch_dtype=dtype, attn_implementation="sdpa", local_files_only=True
        )
        base.requires_grad_(False)
        if adapter:
            require_file(Path(adapter) / "adapter_model.safetensors", "EVOG adapter")
            require_file(Path(adapter) / "adapter_config.json", "EVOG adapter configuration")
            self.model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
        elif trainable:
            lora = LoraConfig(
                r=cfg["lora_rank"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=r".*language_model\..*\.("
                + "|".join(re.escape(name) for name in cfg["lora_targets"])
                + r")",
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base, lora)
        else:
            self.model = base
        self.model.to(self.device)
        self.model.train(trainable)
        if trainable and cfg["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()

    def pack(self, messages, videos):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        arrays = [np.stack([np.asarray(f) for f in video]) for video in videos]
        inputs = self.processor(
            text=[text], videos=arrays, fps=[self.cfg["fps"]] * len(arrays), return_tensors="pt", padding=True
        )
        if inputs.input_ids.shape[1] > self.cfg["max_context_tokens"]:
            raise ValueError(
                "Visual dialogue exceeds max_context_tokens; reduce model.max_pixels or inference.chunk_frames"
            )
        return inputs.to(self.device)

    def constrained(self, messages, videos, count, include_boxes, differentiable):
        inputs = self.pack(messages, videos)
        codec = SpatialCodec(self.processor.tokenizer, count, self.cfg["points"], include_boxes)
        prefix = inputs.input_ids.shape[1]
        training = self.model.training
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=len(codec.layout),
                min_new_tokens=len(codec.layout),
                do_sample=False,
                num_beams=1,
                use_cache=True,
                prefix_allowed_tokens_fn=codec.allowed(prefix),
                eos_token_id=None,
                forced_eos_token_id=None,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        self.model.train(training)
        generated = generated[:, prefix:]
        numeric_logits = None
        if differentiable:
            replay = dict(inputs)
            replay["input_ids"] = torch.cat((inputs.input_ids, generated), dim=1)
            replay["attention_mask"] = torch.ones_like(replay["input_ids"])
            base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
            if hasattr(base.model, "rope_deltas"):
                base.model.rope_deltas = None
            states = base.model(**replay, use_cache=False, return_dict=True).last_hidden_state
            positions = torch.tensor(codec.numeric_positions, device=self.device) + prefix - 1
            hidden = states[0, positions]
            token_ids = torch.tensor(codec.numeric_ids, device=self.device)
            head = base.get_output_embeddings()
            bias = head.bias[token_ids] if getattr(head, "bias", None) is not None else None
            numeric_logits = F.linear(hidden, head.weight[token_ids], bias)
        boxes, points = codec.values(generated, numeric_logits, self.cfg["coordinate_temperature"])
        text = self.processor.tokenizer.decode(generated[0], skip_special_tokens=False)
        return boxes, points, text

    def forward(self, frames, query, refinements, differentiable=False, overview=None):
        videos = [frames] if overview is None else [overview, frames]
        content = [{"type": "video"} for _ in videos]
        prefix = (
            ""
            if overview is None
            else "The first video provides full-video context. Only output predictions for the frames of the second video. "
        )
        content.append({"type": "text", "text": prefix + INITIAL.format(k=self.cfg["points"], query=query)})
        messages = [{"role": "user", "content": content}]
        boxes, points, text = self.constrained(messages, videos, len(frames), True, differentiable)
        messages.append({"role": "assistant", "content": text})
        history = []
        for _ in range(refinements):
            messages.append({"role": "user", "content": REFINE.format(k=self.cfg["points"], query=query)})
            _, points, text = self.constrained(messages, videos, len(frames), False, differentiable)
            history.append(points)
            messages.append({"role": "assistant", "content": text})
        if not history:
            history = [points]
        return SpatialPrediction(boxes, points, history, messages)

    @torch.no_grad()
    def auxiliary_query(self, frames, query, masks):
        from PIL import Image

        overlays = []
        for image, mask in zip(frames, masks):
            rgb = np.asarray(image).astype(np.float32).copy()
            selected = mask > 0
            rgb[selected] = 0.45 * rgb[selected] + 0.55 * np.array([255, 0, 0])
            overlays.append(Image.fromarray(rgb.astype(np.uint8)))
        perspective = random.choice(
            [
                "visual appearance: color, shape, size, texture",
                "action or interaction with another object",
                "spatial relation to other objects",
            ]
        )
        prompt = AUXILIARY.format(query=query, perspective=perspective)
        messages = [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "video"}, {"type": "text", "text": prompt}],
            }
        ]
        inputs = self.pack(messages, [frames, overlays])
        training = self.model.training
        self.model.eval()
        output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
        self.model.train(training)
        text = self.processor.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        if not text:
            raise RuntimeError("Auxiliary-query generation returned empty text")
        return text
