import json
import random
import re
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw


def safe_relative(value):
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value or not path.parts:
        raise ValueError(f"Expected safe relative path: {value}")
    return str(path)


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(path))]


def read_manifest(path):
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("dataset", "video_id", "query_id", "query", "frames"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number} missing {key}")
            for key in ("dataset", "video_id", "query_id"):
                row[key] = safe_relative(row[key])
            if not row["query"].strip() or not row["frames"]:
                raise ValueError(f"Empty query or frames at {path}:{line_number}")
            row["frames"] = [safe_relative(p) for p in row["frames"]]
            names = [Path(p).stem for p in row["frames"]]
            if len(names) != len(set(names)):
                raise ValueError(f"Repeated output frame names: {row['video_id']}")
            if "masks" in row and len(row["masks"]) != len(row["frames"]):
                raise ValueError(f"Mask/frame count differs at {path}:{line_number}")
            identity = (row["dataset"], row["video_id"], row["query_id"])
            if identity in seen:
                raise ValueError(f"Duplicate query: {identity}")
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def decode_mask(spec, size, root):
    h, w = size
    if spec is None:
        return np.zeros((h, w), dtype=np.float32)
    if "union" in spec:
        masks = [decode_mask(s, size, root) for s in spec["union"]]
        return np.maximum.reduce(masks) if masks else np.zeros((h, w), dtype=np.float32)
    if "path" in spec:
        image = np.asarray(Image.open(Path(root) / safe_relative(spec["path"])))
        if image.ndim != 2 or image.shape != (h, w):
            raise ValueError(f"Expected indexed mask with shape {(h, w)}: {spec['path']}")
        ids = spec.get("ids")
        mask = np.isin(image, ids).astype(np.float32) if ids is not None else (image > 0).astype(np.float32)
        if "ignore_id" in spec:
            mask[image == spec["ignore_id"]] = -1
        return mask
    if "shapes" in spec:
        canvas = Image.new("I", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for shape in sorted(spec["shapes"], key=lambda s: -polygon_area(s["points"])):
            label = shape["label"]
            if label in ("target", "ignore"):
                draw.polygon([tuple(p) for p in shape["points"]], fill=1 if label == "target" else -1)
        return np.asarray(canvas, dtype=np.float32)
    from pycocotools import mask as coco_mask

    if "polygon" in spec:
        encoded = coco_mask.frPyObjects(spec["polygon"], h, w)
        encoded = coco_mask.merge(encoded)
    else:
        encoded = dict(spec.get("rle", spec))
        if isinstance(encoded.get("counts"), list):
            encoded = coco_mask.frPyObjects(encoded, h, w)
        elif isinstance(encoded.get("counts"), str):
            encoded["counts"] = encoded["counts"].encode("ascii")
    decoded = coco_mask.decode(encoded)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    if decoded.shape != (h, w):
        raise ValueError(f"Mask shape {decoded.shape} differs from image shape {(h, w)}")
    return decoded.astype(np.float32)


def polygon_area(points):
    xy = np.asarray(points)
    return abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 2


def sample_indices(length, count, interval, rng):
    if length < 1 or count < 1 or interval < 1:
        raise ValueError("Video length, sample count and interval must be positive")
    if length == 1:
        return [0]
    span = (count - 1) * interval
    if length > span:
        start = rng.randrange(length - span)
        return [start + i * interval for i in range(count)]
    return np.linspace(0, length - 1, min(length, count)).round().astype(int).tolist()


def load_sample(row, root, indices=None, require_masks=False):
    indices = list(range(len(row["frames"]))) if indices is None else indices
    frames = []
    for i in indices:
        with Image.open(Path(root) / row["frames"][i]) as image:
            frames.append(image.convert("RGB"))
    if len({im.size for im in frames}) != 1:
        raise ValueError(f"Frames have inconsistent sizes: {row['video_id']}")
    if require_masks and "masks" not in row:
        raise ValueError(f"Training/evaluation requires masks: {row['video_id']}/{row['query_id']}")
    masks = None
    if "masks" in row:
        masks = np.stack(
            [
                decode_mask(row["masks"][i], (frames[j].height, frames[j].width), root)
                for j, i in enumerate(indices)
            ]
        )
    return {"row": row, "indices": indices, "frames": frames, "masks": masks, "query": row["query"]}


class VideoDataset:
    def __init__(self, manifests, root, frames=4, interval=10, seed=42):
        self.rows = [row for manifest in manifests for row in read_manifest(manifest)]
        self.root = root
        self.frames = frames
        self.interval = interval
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random.Random(self.seed + self.epoch * len(self.rows) + index)
        indices = sample_indices(len(row["frames"]), self.frames, self.interval, rng)
        return load_sample(row, self.root, indices, require_masks=True)


def single_collate(batch):
    if len(batch) != 1:
        raise ValueError("Use batch size 1 and gradient accumulation for variable-length videos")
    return batch[0]


def load_media(path):
    path = Path(path)
    if path.is_dir():
        paths = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}],
            key=natural_key,
        )
        frames = [Image.open(p).convert("RGB") for p in paths]
        names = [p.stem for p in paths]
    elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frames, names = [Image.open(path).convert("RGB")], [path.stem]
    else:
        import cv2

        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        capture.release()
        names = [f"{i:05d}" for i in range(len(frames))]
    if not frames:
        raise ValueError(f"No readable frames in {path}")
    if len(set(names)) != len(names) or len({f.size for f in frames}) != 1:
        raise ValueError("Frames must have unique stems and a common resolution")
    return frames, names
