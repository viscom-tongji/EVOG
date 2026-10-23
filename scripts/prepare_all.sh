set -euo pipefail
manifest_root="${MANIFEST_ROOT:-/mnt/dataset/evog_manifests}"
for dataset in revos mevis ref_youtube_vos ref_davis17 refcoco refcoco_plus refcocog reasonseg; do
  python -m evog.prepare --dataset "$dataset" --split train --output "$manifest_root/${dataset}_train.jsonl" --trust-pickle
done
