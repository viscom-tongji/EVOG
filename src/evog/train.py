import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from evog.config import load_config
from evog.data import VideoDataset, single_collate


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evog_7b.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--resume")
    parser.add_argument("--init-adapter")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    from evog.pipeline import TrainingModel

    training = cfg["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation"],
        mixed_precision=training["mixed_precision"],
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=training["seed"]),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(training["seed"], device_specific=True)
    dataset = VideoDataset(
        cfg["data"]["train_manifests"],
        cfg["data"]["root"],
        cfg["data"]["train_frames"],
        cfg["data"]["frame_interval"],
        training["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=training["workers"],
        collate_fn=single_collate,
        pin_memory=False,
    )
    model = TrainingModel(cfg, accelerator.device, adapter=args.init_adapter)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    updates_per_epoch = math.ceil(len(loader) / training["gradient_accumulation"])
    total_updates = training["max_steps"] or training["epochs"] * updates_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, training["warmup_steps"], total_updates)
    accelerator.register_for_checkpointing(scheduler)
    destination = Path(training["output_dir"])
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
    accelerator.wait_for_everyone()

    def save_hook(models, weights, directory):
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(models[0])
            unwrapped.grounder.model.save_pretrained(Path(directory) / "adapter", safe_serialization=True)
            unwrapped.grounder.processor.save_pretrained(Path(directory) / "adapter")
        weights.clear()

    def load_hook(models, directory):
        while models:
            unwrapped = accelerator.unwrap_model(models.pop())
            state = load_file(str(Path(directory) / "adapter" / "adapter_model.safetensors"))
            set_peft_model_state_dict(unwrapped.grounder.model, state)

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    global_step, first_epoch, first_batch = 0, 0, 0
    if args.resume:
        cursor = json.loads((Path(args.resume) / "cursor.json").read_text(encoding="utf-8"))
        old = cursor["config"]
        if old != cfg or cursor["world_size"] != accelerator.num_processes:
            raise ValueError("Exact resume requires the saved configuration and world size")
        accelerator.load_state(args.resume)
        global_step, first_epoch, first_batch = cursor["step"], cursor["epoch"], cursor["next_batch"]

    def checkpoint(epoch, next_batch):
        folder = destination / f"step-{global_step:07d}"
        accelerator.save_state(str(folder))
        if accelerator.is_main_process:
            cursor = {
                "step": global_step,
                "epoch": epoch,
                "next_batch": next_batch,
                "config": cfg,
                "world_size": accelerator.num_processes,
            }
            (folder / "cursor.json").write_text(json.dumps(cursor, indent=2), encoding="utf-8")
        accelerator.wait_for_everyone()

    optimizer.zero_grad(set_to_none=True)
    last_saved = -1
    last_epoch, next_batch = first_epoch, first_batch
    epochs = max(training["epochs"], math.ceil(total_updates / max(updates_per_epoch, 1)))
    for epoch in range(first_epoch, epochs):
        if global_step >= total_updates:
            break
        dataset.set_epoch(epoch)
        loader.set_epoch(epoch)
        offset = first_batch if epoch == first_epoch else 0
        active_loader = accelerator.skip_first_batches(loader, offset) if offset else loader
        model.train()
        for batch_index, sample in enumerate(active_loader, offset):
            with accelerator.accumulate(model):
                losses = model(sample)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(
                        f"Nonfinite loss at step {global_step}: {sample['row']['video_id']}"
                    )
                accelerator.backward(losses["total"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, training["gradient_clip"])
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            last_epoch, next_batch = epoch, batch_index + 1
            if accelerator.sync_gradients:
                global_step += 1
                values = {
                    key: float(accelerator.reduce(value.detach(), reduction="mean"))
                    for key, value in losses.items()
                }
                values.update(step=global_step, epoch=epoch, learning_rate=scheduler.get_last_lr()[0])
                if accelerator.is_main_process:
                    with (destination / "train.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(values) + "\n")
                    print(json.dumps(values), flush=True)
                if global_step % training["save_every"] == 0:
                    checkpoint(last_epoch, next_batch)
                    last_saved = global_step
                if global_step >= total_updates:
                    break
        first_batch = 0
    if global_step != last_saved:
        checkpoint(last_epoch, next_batch)
    accelerator.end_training()


if __name__ == "__main__":
    main()
