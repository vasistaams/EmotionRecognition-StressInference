# Swin-FANE + BiLSTM-ResNet: Emotion Recognition & Temporal Stress Estimation

> **Paper:** "A Two-Phase Swin–FANE and BiLSTM–ResNet Framework for Emotion Recognition and Temporal Stress Estimation"

---

## Overview

This repository provides the **full PyTorch implementation** of the two-phase spatial–temporal framework described in the paper:

| Phase | Component | Purpose |
|-------|-----------|---------|
| I | **Swin-FANE** (Swin Transformer + Facial Attention Network Embedding) | Frame-level emotion recognition |
| II | **BiLSTM-ResNet** (Bidirectional LSTM + Residual Refinement) | Temporal stress estimation |

The pipeline converts facial video frames → emotion probability vectors → stress index, without requiring explicit stress annotations.

---

## Repository Structure

```
swin_fane/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths
│
├── data/
│   ├── fane_dataset.py          # FANE dataset loader (primary)
│   ├── fer2013_dataset.py       # FER2013 dataset loader (alternative)
│   ├── video_dataset.py         # Video sequence dataset loader
│   └── augmentations.py         # Custom augmentation pipeline
│
├── models/
│   ├── swin_transformer.py      # Swin Transformer backbone
│   ├── fane_module.py           # Facial Attention Network Embedding
│   ├── swin_fane.py             # Combined Swin-FANE spatial encoder
│   ├── bilstm_resnet.py         # BiLSTM + Residual temporal module
│   └── full_framework.py        # End-to-end two-phase framework
│
├── utils/
│   ├── metrics.py               # Evaluation metrics (ETSI, TVE, SCS, etc.)
│   ├── stress_formulation.py    # Deterministic stress index computation
│   ├── visualisation.py         # Grad-CAM, trajectory plots, confusion matrix
│   └── logger.py                # Training logger
│
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   ├── infer_video.py           # Single video inference
│   └── run_ablation.py          # Ablation study runner
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-username/swin-fane-stress.git
cd swin-fane-stress
pip install -r requirements.txt
```

### 2. Dataset Preparation

**Option A — FANE Dataset (Primary)**
```
data/
└── FANE/
    ├── images/
    │   ├── anger/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happiness/
    │   ├── sadness/
    │   ├── surprise/
    │   └── neutral/
    └── masks/          # Expressive region masks (optional)
```

**Option B — FER2013 (Alternative)**
Download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
```
data/
└── FER2013/
    ├── train/
    └── test/
```

### 3. Train

```bash
python scripts/train.py --config configs/config.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

### 5. Infer on Video

```bash
python scripts/infer_video.py \
  --video path/to/video.mp4 \
  --checkpoint checkpoints/fer2013_arwm/best_model.pth \
  --config configs/config_arwm_50.yaml \
  --face-mode all \
  --output results/video_output
```

Video inference detects faces on a downscaled frame, maps boxes back to the source
frame, and preprocesses each detected crop with the FER2013 grayscale-to-RGB and
ImageNet-normalization pipeline. `--face-mode` accepts `largest`, `selected`, or
`all`; `--face-index` selects a zero-based area-ranked face in `selected` mode.
Inference writes a per-frame JSON log, per-class CSV probabilities, and annotated
sample frames to the output directory.

---

## ARWM Ablation Reproduction

ARWM can be explicitly selected as `off`, `fixed`, or `adaptive`. The mode-specific
configs inherit the shared 50-epoch FER2013 setup and write checkpoints and logs to
separate `arwm_variants` directories. `fixed` uses equal one-third weights and does
not use the adaptive weighting MLP. `adaptive` uses the trained weighting MLP.
The masks remain fixed geometric priors in both modes; the adaptive weights are
input-dependent, not landmark-aligned.

Train missing variants separately. These commands do not overwrite the legacy
checkpoints:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config configs\arwm\off.yaml --arwm-mode off
.\.venv\Scripts\python.exe scripts\train.py --config configs\arwm\fixed.yaml --arwm-mode fixed
.\.venv\Scripts\python.exe scripts\train.py --config configs\arwm\adaptive.yaml --arwm-mode adaptive
```

Explicit ARWM-mode training refuses to reuse a variant directory containing
checkpoints. To repeat a run, copy its mode config and set fresh checkpoint, log,
and result paths first.

The full official-test evaluation is run once per available checkpoint mode and
writes `metrics.json`, `metrics.csv`, `confusion_matrix.png`, and
`evaluation_summary.md` under `results\arwm`. Missing checkpoints are listed as
unavailable rather than replaced by validation metrics. A small smoke test can be
run before the full CPU evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_arwm_variants.py --config configs\config_arwm_50.yaml --max-samples 64 --warmup-batches 1 --output results\arwm\smoke
.\.venv\Scripts\python.exe scripts\evaluate_arwm_variants.py --config configs\config_arwm_50.yaml --off-checkpoint checkpoints\arwm_variants\off\best_model.pth --fixed-checkpoint checkpoints\arwm_variants\fixed\best_model.pth --adaptive-checkpoint checkpoints\arwm_variants\adaptive\best_model.pth --output results\arwm\final
```

The existing baseline and adaptive checkpoints may be evaluated, but they do not
by themselves prove a controlled ablation: the fixed-mode checkpoint is not
available, and legacy checkpoint metadata cannot establish identical training
conditions. The final command above is for after all three new variants have been
trained. Results and confidence are not claims about movie-footage accuracy.

Run multi-face video inference with the existing adaptive checkpoint and display
the actual sample-dependent region weights:

```powershell
.\.venv\Scripts\python.exe scripts\infer_video.py --video data\videos\testvideo2.mp4 --checkpoint checkpoints\fer2013_arwm\best_model.pth --config configs\config_arwm_50.yaml --arwm-mode adaptive --face-mode all --frame-stride 3 --show-region-weights --output results\testvideo2\arwm_adaptive_live
```

Benchmark first without overlays, then with region-weight overlay drawing enabled.
These runs measure on the current hardware; overlay mode measures OpenCV drawing
overhead in memory and does not include a GUI display or encoded video writing.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_inference.py --video data\videos\testvideo2.mp4 --checkpoint checkpoints\fer2013_arwm\best_model.pth --config configs\config_arwm_50.yaml --arwm-mode adaptive --frame-stride 3 --timed-frames 120 --warmup-iterations 5 --batch-size 4 --output results\arwm\benchmark_adaptive_no_overlay.json
.\.venv\Scripts\python.exe scripts\benchmark_inference.py --video data\videos\testvideo2.mp4 --checkpoint checkpoints\fer2013_arwm\best_model.pth --config configs\config_arwm_50.yaml --arwm-mode adaptive --frame-stride 3 --timed-frames 120 --warmup-iterations 5 --batch-size 4 --show-region-weights --output results\arwm\benchmark_adaptive_overlay.json
```

The benchmark JSON distinguishes model-only forward timing from end-to-end video
decode, face detection, crop preprocessing, inference, and optional overlay
drawing. GPU timing is synchronized when CUDA is available. Compare runs only
when their hardware, device, image size, frame stride, batch size, warm-up, and
timed-frame count match.

### Google Colab GPU workflow

Open [SWIN_ARWM_Colab_Experiments.ipynb](./SWIN_ARWM_Colab_Experiments.ipynb) in
Google Colab, select a GPU runtime, and follow its preflight cells. Upload the
sanitized source archive and FER2013 dataset when prompted. The notebook checks
the source-checkpoint hashes, dataset counts, configuration parity, unit tests,
GPU smoke step, memory, and measured runtime before training. Training is
approval-gated and does not start until `APPROVE_TRAINING = True` is explicitly
set after reviewing the estimate.

The notebook writes each mode's best and resumable checkpoints, resolved config,
logs, evaluation metrics, reports, and confusion matrix to the Drive experiment
directory shown in the final export cell. Download that experiment directory
from Google Drive and copy its `off`, `fixed`, and `adaptive` subdirectories into
a **new** local results location; keep the existing baseline and adaptive
checkpoints untouched. The Drive copy is the durable source of the run, and the
exported report records each checkpoint hash and path.

---

## Results

| Model | Accuracy (%) | Precision (%) | F1-score (%) | ROC-AUC |
|-------|-------------|--------------|-------------|---------|
| Mujiyanto et al. | 88.4 | 87.9 | 88.1 | 0.924 |
| Xue et al. | 89.5 | 89.0 | 89.2 | 0.931 |
| Chen et al. | 90.1 | 90.0 | 90.0 | 0.938 |
| Bie et al. | 89.2 | 89.0 | 89.1 | 0.933 |
| **Swin-FANE (Ours)** | **90.7 ± 0.6** | **91.2** | **90.8** | **0.947** |

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3×10⁻⁴ |
| Batch Size | 32 |
| Epochs | 50 |
| Sequence Length | 16 frames |
| Stress α | 0.7 |
| Stress β | 0.3 |

---

## Citation

```bibtex
@article{swin_fane_2025,
  title={A Two-Phase Swin–FANE and BiLSTM–ResNet Framework for 
         Emotion Recognition and Temporal Stress Estimation},
  year={2025}
}
```

---

## License

MIT License
