# Beyond Token Representations: Explicit Visual Object Grounding for Video Reasoning Segmentation

**Accepted at NeurIPS 2026.**

Jun Huang, Junyang Zhang, Zijie Yue, Bingkun Wang, Yong Luo, and Miaojing Shi.

This repository implements explicit visual object grounding for video reasoning segmentation. `EVOG` is the repository shorthand for the framework. Qwen2.5-VL generates an enclosing box and seven points, refines the points through dialogue, and aligns original and auxiliary queries during training. SAM2 decodes spatial prompts, and Cutie propagates masks from three selected keyframes. Queries referring to multiple instances use their union as the target.

## Installation

Platform: Linux, Python 3.11, NVIDIA CUDA. The manuscript reports four NVIDIA A40 GPUs. The installation script selects PyTorch 2.7.1 with CUDA 12.6; a compatible NVIDIA driver is required.

```bash
conda create -n evog python=3.11 -y
conda activate evog
cd EVOG
bash scripts/install.sh
```

## Server layout

```text
/mnt/dataset/
  rvos_root/
    ReVOS/
    MeViS/
    Ref_Youtube_VOS/
    ref_davis17/
  lisa_root/
    refer_seg/
    reason_seg/ReasonSeg/
  evog_manifests/
/mnt/checkpoints/evog/
  Qwen2.5-VL-3B-Instruct/
  Qwen2.5-VL-7B-Instruct/
  sam2.1_hiera_large.pt
  cutie-base-mega.pth
/mnt/experiments/evog/
```

## Training

```bash
bash scripts/train_7b.sh
bash scripts/train_3b.sh
```

Run one of these commands for the required model size. Default global batch size is 32 videos: four processes, one video per process, and eight accumulation steps.

```bash
NPROC_PER_NODE=1 bash scripts/train_3b.sh --set training.gradient_accumulation=32
```

## Inference and evaluation

Use the latest trained adapter:

```bash
checkpoint=$(find "$OUTPUT_ROOT/7b" -maxdepth 1 -type d -name 'step-*' | sort | tail -n 1)
test -n "$checkpoint"
evog infer --config configs/evog_7b.yaml --adapter "$checkpoint/adapter" --manifest "$MANIFEST_ROOT/revos_valid.jsonl" --output "$OUTPUT_ROOT/predictions"
evog evaluate --manifest "$MANIFEST_ROOT/revos_valid.jsonl" --root "$DATA_ROOT" --predictions "$OUTPUT_ROOT/predictions" --output "$OUTPUT_ROOT/revos_metrics.json"
```

For custom data, `--media` accepts an image, video file, or directory of ordered frames. Supply `--query` and `--adapter`; add `--overlays` to save color visualizations. `evog infer --help` lists all arguments. 

```bash
evog infer --adapter "$checkpoint/adapter" --media /mnt/dataset/demo/input.mp4 --query 'the tool used to cut the tomato' --output "$OUTPUT_ROOT/demo" --overlays
```

## Citation

```bibtex
@inproceedings{huang2026beyond,
  title={Beyond Token Representations: Explicit Visual Object Grounding for Video Reasoning Segmentation},
  author={Huang, Jun and Zhang, Junyang and Yue, Zijie and Wang, Bingkun and Luo, Yong and Shi, Miaojing},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgements

We thank [UniPixel](https://github.com/PolyU-ChenLab/UniPixel), [VRS-HQ](https://github.com/SitongGong/VRS-HQ), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [SAM2](https://github.com/facebookresearch/sam2), and [Cutie](https://github.com/hkchengrex/Cutie). 
