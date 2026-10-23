import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


def download_file(url, path, md5=None):
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        if md5 is None or hashlib.md5(path.read_bytes()).hexdigest() == md5:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"Empty download: {url}")
    if md5 and hashlib.md5(temporary.read_bytes()).hexdigest() != md5:
        raise RuntimeError(f"Checksum mismatch: {path}")
    temporary.replace(path)


def main(command, argv=None):
    parser = argparse.ArgumentParser(prog=f"evog {command}")
    if command == "download":
        parser.add_argument("--root", default=os.environ.get("CHECKPOINT_ROOT", "/mnt/checkpoints/evog"))
        parser.add_argument("--scale", choices=["3B", "7B", "both"], default="7B")
        args = parser.parse_args(argv)
        root = Path(args.root)
        from huggingface_hub import snapshot_download

        for scale in ["3B", "7B"] if args.scale == "both" else [args.scale]:
            name = f"Qwen2.5-VL-{scale}-Instruct"
            snapshot_download(repo_id=f"Qwen/{name}", local_dir=root / name)
        download_file(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            root / "sam2.1_hiera_large.pt",
        )
        download_file(
            "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
            root / "cutie-base-mega.pth",
            "a6071de6136982e396851903ab4c083a",
        )
        for name in ("resnet18-5c106cde.pth", "resnet50-19c8e357.pth"):
            download_file(f"https://download.pytorch.org/models/{name}", root / "resnet" / name)
    elif command == "pack":
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--predictions", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args(argv)
        from evog.evaluate import pack_submission

        pack_submission(args.manifest, args.predictions, args.output)
    elif command == "merge":
        parser.add_argument("--config", required=True)
        parser.add_argument("--adapter", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--device", default="cpu")
        args = parser.parse_args(argv)
        from evog.config import load_config
        from evog.grounder import Grounder

        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Merged model destination must be empty: {output}")
        grounder = Grounder(load_config(args.config)["model"], args.device, adapter=args.adapter)
        model = grounder.model.merge_and_unload()
        model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
        grounder.processor.save_pretrained(output)
