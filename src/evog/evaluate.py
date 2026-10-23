import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from evog.data import load_sample, read_manifest
from evog.metrics import boundary_f, region_iou


def summarize(records):
    if not records:
        raise ValueError("No evaluated records")
    j = float(np.mean([r["J"] for r in records]))
    f = float(np.mean([r["F"] for r in records]))
    union = sum(r["union"] for r in records)
    intersection = sum(r["intersection"] for r in records)
    return {
        "queries": len(records),
        "J": 100 * j,
        "F": 100 * f,
        "J&F": 50 * (j + f),
        "gIoU": 100 * j,
        "cIoU": 100 * intersection / union if union else 100.0,
    }


def evaluate(manifest, root, predictions):
    records = []
    for row in read_manifest(manifest):
        sample = load_sample(row, root, require_masks=True)
        folder = Path(predictions) / row["dataset"] / row["video_id"] / row["query_id"]
        js, fs, intersection, union = [], [], 0, 0
        for frame, target in zip(row["frames"], sample["masks"]):
            path = folder / (Path(frame).stem + ".png")
            with Image.open(path) as image:
                pred = np.asarray(image)
            if pred.ndim != 2 or pred.shape != target.shape:
                raise ValueError(f"Prediction shape mismatch: {path}")
            if not (target >= 0).any():
                continue
            pred = pred > 0
            j, inter, uni = region_iou(pred, target > 0, target >= 0)
            js.append(j)
            fs.append(boundary_f(pred, target > 0, target >= 0))
            intersection += inter
            union += uni
        if not js:
            raise ValueError(f"All pixels ignored: {row['video_id']}/{row['query_id']}")
        records.append(
            {
                "dataset": row["dataset"],
                "video_id": row["video_id"],
                "query_id": row["query_id"],
                "query_type": row.get("query_type", "unspecified"),
                "J": float(np.mean(js)),
                "F": float(np.mean(fs)),
                "intersection": intersection,
                "union": union,
            }
        )
    groups = defaultdict(list)
    for record in records:
        groups[record["dataset"]].append(record)
    results = {name: summarize(group) for name, group in groups.items()}
    for name, group in groups.items():
        split = {kind: [r for r in group if r["query_type"] == kind] for kind in ("reasoning", "referring")}
        if any(split.values()):
            results[name]["subsets"] = {kind: summarize(rows) for kind, rows in split.items() if rows}
        if all(split.values()):
            results[name]["balanced_overall"] = {
                metric: float(np.mean([results[name]["subsets"][kind][metric] for kind in split]))
                for metric in ("J", "F", "J&F")
            }
    return {"metrics": results, "per_query": records}


def pack_submission(manifest, predictions, output):
    rows = read_manifest(manifest)
    if len({r["dataset"] for r in rows}) != 1:
        raise ValueError("Submission archives must contain exactly one dataset")
    seen = set()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for row in rows:
            for frame in row["frames"]:
                relative = Path(row["video_id"]) / row["query_id"] / (Path(frame).stem + ".png")
                member = relative.as_posix()
                if member in seen:
                    raise ValueError(f"Duplicate submission entry: {member}")
                seen.add(member)
                source = Path(predictions) / row["dataset"] / relative
                if not source.is_file():
                    raise FileNotFoundError(source)
                archive.write(source, member)
    temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.manifest, args.root, args.predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
