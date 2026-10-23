import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from evog.config import load_config
from evog.data import load_sample, read_manifest


def check(cfg, weights=False, manifests=False, adapter=None):
    issues = []
    for name in ("model", "sam2", "cutie", "method", "loss", "data", "training", "inference"):
        if name not in cfg:
            issues.append(f"Missing configuration section: {name}")
    if issues:
        return {"ok": False, "issues": issues}
    for module in (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "accelerate",
        "sam2",
        "cutie",
        "pycocotools",
        "hydra",
    ):
        if importlib.util.find_spec(module) is None:
            issues.append(f"Missing module: {module}")
    for section, field in (
        ("model", "points"),
        ("method", "keyframes"),
        ("data", "train_frames"),
        ("data", "frame_interval"),
        ("training", "gradient_accumulation"),
        ("training", "save_every"),
        ("inference", "chunk_frames"),
    ):
        if cfg[section][field] < 1:
            issues.append(f"{section}.{field} must be positive")
    if cfg["method"]["selection"] not in ("combined", "presence", "geometry"):
        issues.append("method.selection must be combined, presence or geometry")
    if weights:
        paths = [cfg["sam2"]["checkpoint"], cfg["cutie"]["checkpoint"]]
        paths += [
            str(Path(cfg["cutie"]["resnet_cache"]) / n)
            for n in ("resnet18-5c106cde.pth", "resnet50-19c8e357.pth")
        ]
        base = Path(cfg["model"]["base_model"])
        paths += [
            str(base / "config.json"),
            str(base / "tokenizer_config.json"),
            str(base / "preprocessor_config.json"),
        ]
        if not any(p.stat().st_size > 0 for p in base.glob("*.safetensors")):
            issues.append(f"Missing Qwen safetensors: {base}")
        if adapter:
            paths += [str(Path(adapter) / n) for n in ("adapter_config.json", "adapter_model.safetensors")]
        for path in paths:
            if not Path(path).is_file() or Path(path).stat().st_size == 0:
                issues.append(f"Missing or empty asset: {path}")
    if manifests:
        for path in cfg["data"]["train_manifests"]:
            try:
                rows = read_manifest(path)
                for row in rows:
                    load_sample(row, cfg["data"]["root"], require_masks=True)
            except (ValueError, OSError, KeyError) as error:
                issues.append(f"Dataset check failed for {path}: {error}")
    versions = {}
    for name in ("torch", "torchvision", "transformers", "peft", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"ok": not issues, "issues": issues, "versions": versions}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evog_7b.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--datasets", action="store_true")
    parser.add_argument("--adapter")
    args = parser.parse_args(argv)
    result = check(load_config(args.config, args.set), args.weights, args.datasets, args.adapter)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
