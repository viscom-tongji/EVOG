import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evog.config import load_config
from evog.data import load_media, load_sample, read_manifest, safe_relative


def save_prediction(output, names, masks, trace, frames=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "complete.json").unlink(missing_ok=True)
    if len(names) != len(masks):
        raise ValueError("Prediction frame names and masks must have equal lengths")
    for i, (name, mask) in enumerate(zip(names, masks)):
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"{safe_relative(name)}.png")
        if frames is not None:
            overlay = np.asarray(frames[i]).copy().astype(np.float32)
            overlay[mask] = overlay[mask] * 0.45 + np.array([255, 65, 65]) * 0.55
            folder = output / "overlays"
            folder.mkdir(exist_ok=True)
            Image.fromarray(overlay.astype(np.uint8)).save(folder / f"{name}.jpg")
    (output / "trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "complete.json").write_text(json.dumps({"frames": names}), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evog_7b.yaml")
    parser.add_argument("--set", action="append", default=[])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--media")
    source.add_argument("--manifest")
    parser.add_argument("--query")
    parser.add_argument("--adapter")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overlays", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    if args.media and not args.query:
        parser.error("--media requires --query")
    cfg = load_config(args.config, args.set)
    from evog.pipeline import InferencePipeline

    pipeline = InferencePipeline(cfg, args.device, args.adapter, args.base_only)
    if args.media:
        frames, names = load_media(args.media)
        masks, trace = pipeline(frames, args.query)
        trace["base_only"] = args.base_only
        trace["config"] = cfg
        save_prediction(args.output, names, masks, trace, frames if args.overlays else None)
    else:
        rows = read_manifest(args.manifest)
        for row in rows[args.shard_index :: args.num_shards]:
            output = Path(args.output) / row["dataset"] / row["video_id"] / row["query_id"]
            sample = load_sample(row, cfg["data"]["root"])
            masks, trace = pipeline(sample["frames"], sample["query"])
            trace["base_only"] = args.base_only
            trace["config"] = cfg
            names = [Path(f).stem for f in row["frames"]]
            save_prediction(output, names, masks, trace, sample["frames"] if args.overlays else None)
            print(f"Saved {row['dataset']}/{row['video_id']}/{row['query_id']}", flush=True)


if __name__ == "__main__":
    main()
