set -euo pipefail
export TOKENIZERS_PARALLELISM=false
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" -m evog.train --config configs/evog_3b.yaml "$@"
