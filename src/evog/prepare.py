import argparse
import json
import pickle
from pathlib import Path

from evog.config import load_config
from evog.data import safe_relative


def read_object(path, trust_pickle=False):
    path = Path(path)
    if path.suffix in (".pkl", ".p"):
        if not trust_pickle:
            raise ValueError(f"Trusted dataset pickle requires --trust-pickle: {path}")
        with path.open("rb") as stream:
            return pickle.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value):
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def segmentation_spec(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {"rle": jsonable(value)}
    if isinstance(value, list) and value and isinstance(value[0], (list, int, float)):
        return {"polygon": value if isinstance(value[0], list) else [value]}
    raise ValueError(f"Unsupported segmentation format: {type(value)}")


def convert_video(name, source, root, trust_pickle):
    metadata = read_object(root / source["metadata"])["videos"]
    mask_path = source.get("mask_dictionary")
    masks = read_object(root / mask_path, trust_pickle) if mask_path else None
    encoding = source.get("mask_encoding", "annotation_ids")
    ordinal = 0
    for video_id, video in metadata.items():
        frames = video["frames"]
        frame_paths = [
            str(Path(source["images"]) / video_id / (f if Path(f).suffix else f + ".jpg")) for f in frames
        ]
        for query_id, expression in video["expressions"].items():
            row = {
                "dataset": name,
                "video_id": video_id,
                "query_id": str(query_id),
                "query": expression["exp"].strip(),
                "frames": frame_paths,
            }
            if "type_id" in expression:
                row["query_type"] = "reasoning" if int(expression["type_id"]) == 1 else "referring"
            elif "type_id" in video:
                row["query_type"] = "reasoning" if int(video["type_id"]) == 1 else "referring"
            if masks is not None:
                if encoding == "annotation_ids":
                    tracks = [masks[str(a)] for a in expression["anno_id"]]
                    if any(len(t) != len(frames) for t in tracks):
                        raise ValueError(f"Track length mismatch: {video_id}/{query_id}")
                    row["masks"] = [
                        {"union": [segmentation_spec(t[i]) for t in tracks]} for i in range(len(frames))
                    ]
                elif encoding == "expression_order":
                    track = masks[str(ordinal)]
                    if len(track) != len(frames):
                        raise ValueError(f"Track length mismatch: {video_id}/{query_id}")
                    row["masks"] = [segmentation_spec(m) for m in track]
                else:
                    raise ValueError(f"Unknown mask encoding: {encoding}")
            elif source.get("annotations"):
                ids = expression.get("obj_id", expression.get("obj_ids"))
                ids = ids if isinstance(ids, list) else [ids]
                if any(i is None for i in ids):
                    raise ValueError(f"Missing object IDs: {video_id}/{query_id}")
                row["masks"] = [
                    {
                        "path": str(Path(source["annotations"]) / video_id / (Path(f).stem + ".png")),
                        "ids": [int(i) for i in ids],
                    }
                    for f in frames
                ]
            ordinal += 1
            yield row


def convert_refer(name, source, root, trust_pickle):
    refs = read_object(root / source["references"], trust_pickle)
    coco = read_object(root / source["instances"])
    images = {x["id"]: x for x in coco["images"]}
    annotations = {x["id"]: x for x in coco["annotations"]}
    for ref in refs:
        if ref["split"] != source["split"]:
            continue
        image = images[ref["image_id"]]
        ann = annotations[ref["ann_id"]]
        for sentence in ref["sentences"]:
            yield {
                "dataset": name,
                "video_id": str(image["id"]),
                "query_id": str(sentence["sent_id"]),
                "query": sentence["sent"],
                "frames": [str(Path(source["images"]) / image["file_name"])],
                "masks": [segmentation_spec(ann["segmentation"])],
            }


def convert_reasonseg(name, source, root):
    for image in sorted((root / source["images"]).glob("*.jpg")):
        annotation = read_object(image.with_suffix(".json"))
        shapes = [s for s in annotation["shapes"] if s["label"] in ("target", "ignore")]
        texts = annotation["text"]
        texts = [texts] if isinstance(texts, str) else texts
        for query_id, query in enumerate(texts):
            yield {
                "dataset": name,
                "video_id": image.stem,
                "query_id": str(query_id),
                "query": query,
                "frames": [str(image.relative_to(root))],
                "masks": [{"shapes": shapes}],
            }


def prepare(config, dataset, split, output, trust_pickle=False):
    root = Path(config["root"])
    source = config["datasets"][dataset][split]
    if source["format"] == "video":
        rows = convert_video(dataset, source, root, trust_pickle)
    elif source["format"] == "refer":
        rows = convert_refer(dataset, source, root, trust_pickle)
    elif source["format"] == "reasonseg":
        rows = convert_reasonseg(dataset, source, root)
    else:
        raise ValueError(source["format"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            row["frames"] = [safe_relative(f) for f in row["frames"]]
            for frame in row["frames"]:
                if not (root / frame).is_file():
                    raise FileNotFoundError(root / frame)
            stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"No records for {dataset}/{split}")
    temporary.replace(output)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-pickle", action="store_true")
    args = parser.parse_args(argv)
    count = prepare(load_config(args.config), args.dataset, args.split, args.output, args.trust_pickle)
    print(json.dumps({"records": count, "output": args.output}))


if __name__ == "__main__":
    main()
