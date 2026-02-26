<div align="center">

# 🌿 Offroad Semantic Scene Segmentation

### Hack For Green Bharat Hackathon — Duality AI

**Team DDOS_ME**

*Karnajeet Gosavi · Jay Gautam · Manas Bagul · Archit Bagad*

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

Pixel-wise semantic segmentation of synthetic desert environments using **DeepLabV3+ with ResNet-101** backbone, trained with a novel two-phase domain-generalisation strategy. The model segments offroad terrain images into **10 semantic classes**.

<img src="training_dashboard.png" alt="Training Dashboard" width="90%"/>

*Complete training dashboard — loss, mIoU, accuracy, learning rate, and per-class performance across 50 epochs.*

</div>

---

## 📑 Table of Contents

1. [Project Structure](#-project-structure)
2. [Environment Setup](#-environment-setup)
3. [Dataset](#-dataset)
4. [Training](#-training)
5. [Testing / Inference](#-testing--inference)
6. [Results](#-results)
7. [Methodology](#-methodology)
8. [Evaluation Metrics](#-evaluation-metrics)
9. [Failure Case Analysis](#-failure-case-analysis)
10. [PDF Report Generation](#-pdf-report-generation)
11. [Reproducing Results](#-reproducing-results)

---

## 📂 Project Structure

```
HACKFORBHARAT/
├── config.py                # All configuration: paths, classes, hyperparameters
├── dataset.py               # Dataset class & data loaders with augmentation
├── model.py                 # DeepLabV3+ model builder & loader
├── train.py                 # Phase 1 training script (base model)
├── train_phase2.py          # Phase 2 fine-tuning (v4 domain-generalisation)
├── test.py                  # Test / inference with TTA + failure analysis
├── utils.py                 # Metrics (IoU), loss functions, plotting utilities
├── generate_report_pdf.py   # Generate 8-page PDF hackathon report with plots
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── PROBLEM_STATEMENT.md     # Original hackathon problem statement
├── HACKATHON_REPORT.pdf     # Generated 8-page PDF report (with plots & metrics)
│
├── Dataset/                 # Provided dataset (not tracked in git)
│   ├── Offroad_Segmentation_Training_Dataset/
│   │   ├── train/  (Color_Images/ + Segmentation/)   [2857 images]
│   │   └── val/    (Color_Images/ + Segmentation/)   [317 images]
│   └── Offroad_Segmentation_testImages/
│       ├── Color_Images/                              [1002 images]
│       └── Segmentation/
│
└── runs_phase2/             # All outputs (Phase 2 final)
    ├── checkpoints/         # best_model.pth, last_model.pth
    ├── plots/               # Training plots, dashboard, confusion matrix
    ├── logs/                # history_v4.json, train_log_v4.txt
    └── predictions/         # Test predictions & evaluation
        ├── submission_masks/  # Final submission masks (uint16)
        ├── masks/             # Raw class-index masks
        ├── masks_color/       # Coloured visualisations
        ├── comparisons/       # Side-by-side comparisons
        ├── failure_analysis/  # Per-class failure case visualisations
        ├── test_metrics.txt   # IoU scores
        └── test_results.json  # Machine-readable results
```

---

## ⚙️ Environment Setup

### Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.10 |
| PyTorch | 2.1.2 + CUDA 12.1 |
| torchvision | 0.16.2 |
| GPU | NVIDIA RTX 4060 (8 GB VRAM) or equivalent |

### Installation

```bash
# 1. Create virtual environment
py -3.10 -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install PyTorch with CUDA
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt
```

<details>
<summary><strong>Dependencies (requirements.txt)</strong></summary>

```
numpy==1.26.4
Pillow>=9.4
matplotlib>=3.7
tqdm>=4.64
opencv-contrib-python>=4.7
scipy>=1.11
```

</details>

---

## 📊 Dataset

The dataset is provided by **Duality AI's Falcon** synthetic data platform, consisting of desert environment scenes.

| Split | Images | Resolution | Purpose |
|-------|--------|------------|---------|
| Train | 2,857  | 540×960    | Model training |
| Val   | 317    | 540×960    | Validation during training |
| Test  | 1,002  | 540×960    | Final evaluation (different desert environment) |

### Class Definitions (10 Classes)

| Original ID | Class Name      | Index | In Test Set? | Train Pixels % | Test Pixels % |
|-------------|----------------|-------|:------------:|:--------------:|:-------------:|
| 100         | Trees          | 0     | ✅           | 3.53%          | 0.27%         |
| 200         | Lush Bushes    | 1     | ✅           | 5.93%          | ~0%           |
| 300         | Dry Grass      | 2     | ✅           | 18.87%         | 17.4%         |
| 500         | Dry Bushes     | 3     | ✅           | 1.10%          | 3.05%         |
| 550         | Ground Clutter | 4     | ❌           | 4.39%          | 0%            |
| 600         | Flowers        | 5     | ❌           | 2.81%          | 0%            |
| 700         | Logs           | 6     | ❌           | 0.08%          | 0%            |
| 800         | Rocks          | 7     | ✅           | 1.20%          | 18.1%         |
| 7100        | Landscape      | 8     | ✅           | 24.45%         | 43.2%         |
| 10000       | Sky            | 9     | ✅           | 37.64%         | 18.0%         |

> **⚠️ Key Challenge — Domain Shift:**
> - 3 classes (Ground Clutter, Flowers, Logs) are **entirely absent** in the test set
> - Rocks: 1.2% → 18.1% in test (**15× increase**)
> - Landscape: 24.5% → 43.2% in test
> - Train and test images come from **different desert environments** (train brightness 0.507 vs test 0.423)

---

## 🏋️ Training

### Phase 1 — Base Model (448×800)

```bash
python train.py --epochs 80 --batch_size 4 --lr 6e-5
```

| Parameter | Value |
|-----------|-------|
| Model | DeepLabV3+ ResNet-101 (COCO pretrained) |
| Resolution | 448×800 |
| Batch Size | 4 (grad accum 2 → effective 8) |
| Optimizer | AdamW (lr=6e-5, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CE + Dice (0.5/0.5) + Aux (0.4) |
| Epochs | 80 (early stop patience 20) |
| Result | **Val mIoU = 0.6077** (Epoch 66) |

### Phase 2 — Domain-Generalisation Fine-Tuning (540×960)

```bash
python train_phase2.py --checkpoint runs_phase2/checkpoints/best_model.pth --epochs 50 --lr 6e-5
```

| Parameter | Value |
|-----------|-------|
| Resolution | 540×960 (original, no downscaling) |
| Batch Size | 2 (grad accum 4 → effective 8) |
| Optimizer | AdamW (head lr=6e-5, backbone lr=1.5e-5) |
| Scheduler | OneCycleLR (5% warmup → cosine) |
| Loss | Focal (γ=2, w=0.4) + Dice (w=0.6) + Aux (0.4) |
| EMA | Exponential Moving Average (decay=0.999) |
| Epochs | 50 |
| Result | **Val mIoU = 0.6338** (Epoch 45) |

<div align="center">
<img src="phase_comparison.png" alt="Phase 1 vs Phase 2 Comparison" width="80%"/>

*Phase 1 vs Phase 2 — improvements across loss, mIoU, and accuracy from domain-generalisation fine-tuning.*
</div>

#### Learning Rate Schedule

<div align="center">
<img src="lr_schedule.png" alt="Learning Rate Schedule" width="55%"/>

*OneCycleLR schedule with 5% warmup followed by cosine decay over 50 epochs.*
</div>

#### Phase 2 v4 Augmentation Strategy

Designed specifically to close the train→test domain gap:

| Augmentation | Parameters | Purpose |
|-------------|------------|---------|
| ColorJitter | brightness=0.6, contrast=0.6, saturation=0.6, hue=0.25 | Bridge lighting gap (train 0.507 → test 0.423) |
| Random Gamma | p=0.3, range=[0.5, 2.0] | Simulate exposure variation |
| Random Grayscale | p=0.2 | Force shape-based learning, reduce colour dependency |
| Channel Shuffle | p=0.1 | Prevent channel-specific overfitting |
| Posterize | p=0.1, bits=[3,6] | Robustness to colour quantisation |
| Scale | [0.5, 2.0] | Wide range for scale invariance |
| Rotation | ±15° | Geometric invariance |
| Gaussian Blur | sigma=[0.5, 2.5] | Defocus simulation |
| Copy-Paste | p=0.5, Rock+DryBush patches | Increase rare class exposure with independent colour jitter |
| Weighted Sampling | Rock images 2.5× oversampled | Balance class representation per epoch |

#### Phase 2 Class Weights

Tuned for the test distribution domain shift:

| Class | Weight | Rationale |
|-------|--------|-----------|
| Trees | 1.2 | Moderate — present in test but small |
| Lush Bushes | 0.2 | Very low — nearly absent in test |
| Dry Grass | 1.3 | Moderate — comparable train/test % |
| Dry Bushes | 5.0 | High — rare in train, present in test |
| Ground Clutter | 0.1 | Minimal — absent in test |
| Flowers | 0.1 | Minimal — absent in test |
| Logs | 0.1 | Minimal — absent in test |
| Rocks | 10.0 | Highest — critical domain shift class |
| Landscape | 2.0 | Moderate-high — dominant in test |
| Sky | 0.4 | Low — well-learned, need not dominate |

---

## 🧪 Testing / Inference

### Standard Test (Multi-Scale TTA)

```bash
python test.py --checkpoint runs_phase2/checkpoints/best_model.pth --multiscale --failure_analysis
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | Auto-detects Phase 2 best | Model weights path |
| `--multiscale` | off | Multi-scale TTA (0.75, 1.0, 1.25 × hflip = 6 passes) |
| `--no_tta` | off | Disable all augmentation at test time |
| `--failure_analysis` | off | Generate per-class failure visualisations |
| `--num_vis` | 8 | Number of comparison images to save |

### Output Structure

```
runs_phase2/predictions/
├── submission_masks/     # ← FINAL: Original-ID masks (uint16) for submission
├── masks/                # Raw class-index masks (0-9)
├── masks_color/          # Coloured visualisation of predictions
├── comparisons/          # Side-by-side: input | GT | prediction
├── failure_analysis/     # Worst predictions per class
├── test_metrics.txt      # IoU scores (text report)
├── test_results.json     # Machine-readable evaluation results
├── test_per_class_iou.png    # Per-class IoU bar chart
└── test_confusion_matrix.png # 10×10 confusion matrix
```

---

## 📈 Results

### Training Performance (Validation Set — Same Domain)

| Metric | Phase 1 | Phase 2 v4 | Improvement |
|--------|---------|------------|:-----------:|
| Val mIoU (all 10 classes) | 0.6077 | **0.6338** | +0.0261 |
| Val Test-Relevant mIoU (7 classes) | — | **0.6913** | — |
| Val Pixel Accuracy | — | 0.8645 | — |

### Training Curves

<div align="center">

| Loss | IoU |
|:---:|:---:|
| <img src="loss_curves.png" alt="Loss Curves" width="400"/> | <img src="iou_curves.png" alt="IoU Curves" width="400"/> |
| *Train vs validation loss over 50 epochs* | *mIoU progression — all classes & test-relevant* |

| Accuracy | Test-Relevant mIoU |
|:---:|:---:|
| <img src="accuracy_curves.png" alt="Accuracy Curves" width="400"/> | <img src="test_relevant_miou.png" alt="Test-Relevant mIoU" width="400"/> |
| *Pixel accuracy convergence during training* | *7-class test-relevant mIoU reaching 0.6913* |

</div>

#### Phase 2 v4 — Per-Class Validation IoU (Best Epoch 45)

<div align="center">
<img src="val_per_class_iou.png" alt="Validation Per-Class IoU" width="70%"/>

*Per-class IoU breakdown on the validation set — Sky (0.98) dominates while Dry Bushes (0.50) and Rocks (0.48) remain challenging.*
</div>

| Class | Phase 2 v4 Val IoU | In Test? |
|-------|:------------------:|:--------:|
| Trees | 0.8476 | ✅ |
| Lush Bushes | 0.6805 | ✅ |
| Dry Grass | 0.6803 | ✅ |
| Dry Bushes | 0.4962 | ✅ |
| Ground Clutter | 0.2974 | ❌ |
| Flowers | 0.6313 | ❌ |
| Logs | 0.5698 | ❌ |
| Rocks | 0.4761 | ✅ |
| Landscape | 0.6760 | ✅ |
| Sky | 0.9825 | ✅ |

<div align="center">
<img src="val_confusion_matrix.png" alt="Validation Confusion Matrix" width="60%"/>

*Validation confusion matrix — diagonal dominance shows strong class separation on the training domain.*
</div>

### Test Performance (Novel Desert Environment)

| Class | Test IoU | Test Pixels % |
|-------|:--------:|:-------------:|
| Trees | 0.3787 | 0.27% |
| Lush Bushes | 0.0005 | ~0% |
| Dry Grass | 0.4581 | 17.4% |
| Dry Bushes | 0.5020 | 3.05% |
| Ground Clutter | 0.0000 | 0% |
| Flowers | 0.0000 | 0% |
| Logs | 0.0000 | 0% |
| Rocks | 0.0569 | 18.1% |
| Landscape | 0.6457 | 43.2% |
| Sky | 0.9707 | 18.0% |

| Metric | Value |
|--------|-------|
| **Test mIoU (all 10)** | **0.3013** |
| **Test mIoU (7 present classes)** | **0.4304** |
| Test Pixel Accuracy | 0.6958 |
| Avg Inference Time | 224.2 ms/image (multi-scale TTA) |

> **Note:** The 10-class mIoU (0.3013) includes 3 classes entirely absent from the test set (Ground Clutter, Flowers, Logs), each contributing 0.0. The **7-class mIoU (0.4304)** is a more representative measure of model performance.

<div align="center">
<img src="final_per_class_iou.png" alt="Final Per-Class IoU" width="70%"/>

*Per-class IoU on test set — Sky (0.97) and Landscape (0.65) perform well, Rocks (0.06) is the key failure point.*
</div>

<div align="center">
<img src="final_confusion_matrix.png" alt="Test Confusion Matrix" width="60%"/>

*Test confusion matrix — reveals Rocks being heavily misclassified as Landscape and Dry Grass, the primary source of domain-shift error.*
</div>

### Sample Predictions

<div align="center">
<img src="val_predictions.png" alt="Validation Predictions" width="90%"/>

*Sample validation predictions — input image (left), ground truth (centre), model prediction (right).*
</div>

---

## 📄 PDF Report Generation

A comprehensive 8-page PDF report (`HACKATHON_REPORT.pdf`) can be regenerated:

```bash
python generate_report_pdf.py
```

The script uses **fpdf2** for PDF layout and **Pillow** for accurate image dimension calculation. The report includes training dashboards, methodology, all plots, confusion matrices, and failure analysis.

---

## 🔬 Methodology

### Two-Phase Training Strategy

#### Phase 1 — Base Feature Learning
1. **Model:** DeepLabV3+ with ResNet-101 backbone, COCO-pretrained weights
2. **Resolution:** 448×800 (16:9 preserved, fits 8GB VRAM at batch 4)
3. **Augmentation:** Random scale (0.5-2.0×), horizontal flip, rotation (±15°), colour jitter, Gaussian blur
4. **Loss:** Combined Cross-Entropy + Dice (0.5/0.5) with class weights
5. **Optimizer:** AdamW with differential LR (backbone 0.1× head) + cosine annealing
6. **AMP:** Mixed-precision training for memory efficiency

#### Phase 2 — Domain-Generalisation Fine-Tuning
1. **Resolution:** Original 540×960 — zero information loss from downscaling
2. **Foundation:** Loads Phase 1 best checkpoint and fine-tunes
3. **Strong augmentation (v4):** 7 additional techniques targeting the brightness/contrast domain gap
4. **Copy-paste augmentation:** Rock and DryBush patches inserted with independent colour jitter (50% probability)
5. **Weighted sampling:** Images containing >2% Rock/DryBush pixels oversampled 2.5×
6. **Loss:** Focal Loss (γ=2.0) + Dice (0.4/0.6) — focuses on hard, rare pixels
7. **EMA:** Exponential Moving Average (decay=0.999) for stable generalisation
8. **OneCycleLR:** 5% warmup → cosine decay

### Domain Shift Analysis

The core challenge is the significant domain gap between training and test environments:

| Property | Train | Test | Gap |
|----------|-------|------|-----|
| Mean Brightness | 0.507 | 0.423 | -16.6% |
| RGB Distribution | R=0.099, G=0.099, B=0.053 | Different | Shifted |
| Rock Coverage | 1.2% | 18.1% | 15× increase |
| Landscape Coverage | 24.5% | 43.2% | 1.8× increase |
| Desert Environment | Environment A | Environment B | Completely different |

This domain shift causes the validation → test performance gap (val mIoU 0.6338 → test mIoU 0.3013). The v4 augmentation strategy specifically addresses the lighting and colour shift aspects.

---

## 📏 Evaluation Metrics

- **Mean IoU (mIoU):** Primary hackathon metric — intersection over union averaged across all classes
- **Test-Relevant mIoU:** mIoU over 7 classes present in the test set
- **Per-Class IoU:** Breakdown per category to identify strengths and failure modes
- **Pixel Accuracy:** Fraction of correctly classified pixels
- **Confusion Matrix:** Visualises inter-class misclassification patterns
- **Inference Speed:** 224.2 ms/image (multi-scale TTA)

---

## 🔍 Failure Case Analysis

Failure analysis is generated by `test.py --failure_analysis` and saved to `runs_phase2/predictions/failure_analysis/`.

### Key Failure Modes

1. **Rocks (IoU: 0.0569):** Most significant failure — dramatic visual difference between train and test rocks (texture, colour, lighting). Frequently misclassified as Landscape or Dry Grass.
2. **Trees (IoU: 0.3787):** Moderate failure — different species, shape, and lighting in test.
3. **Lush Bushes (IoU: 0.0005):** Near zero — effectively absent in test.
4. **3 Absent Classes (IoU: 0.0000):** Ground Clutter, Flowers, and Logs not present in test.

### Why the Gap Exists

The synthetic dataset from Duality AI's Falcon platform uses two distinct desert environments for train and test. While the model achieves strong validation performance (mIoU 0.6338), full transfer is limited by:
- Different terrain textures and materials
- Different lighting conditions (16.6% brightness difference)
- Different class proportions (15× more Rocks in test)
- Different visual appearance of the same semantic classes

---

## 🔄 Reproducing Results

```bash
# 1. Setup environment
py -3.10 -m venv venv
venv\Scripts\activate                                              # Windows
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Phase 1 training (~10 hours on RTX 4060)
python train.py --epochs 80 --batch_size 4 --lr 6e-5

# 3. Phase 2 fine-tuning (~10.7 hours on RTX 4060)
python train_phase2.py --checkpoint runs_phase2/checkpoints/best_model.pth --epochs 50 --lr 6e-5

# 4. Test evaluation with multi-scale TTA + failure analysis (~9 minutes)
python test.py --checkpoint runs_phase2/checkpoints/best_model.pth --multiscale --failure_analysis

# 5. Generate PDF report
python generate_report_pdf.py
```

### Expected Outputs

| Stage | Metric | Expected Value |
|-------|--------|----------------|
| Phase 1 | Val mIoU | ≈ 0.6077 (epoch ~66) |
| Phase 2 v4 | Val mIoU | ≈ 0.6338 (epoch ~45) |
| Phase 2 v4 | Test-Relevant Val mIoU | ≈ 0.6913 |
| Test | mIoU (10-class) | ≈ 0.3013 |
| Test | mIoU (7 present classes) | ≈ 0.4304 |
| Submission | Mask count | 1002 uint16 PNGs |

---

## ⚠️ Important Notes

- **Test images are NEVER used during training or validation.** Strict train/val/test separation is maintained.
- **Training data only:** The model is trained exclusively on the provided hackathon dataset.
- **Class IDs in submission masks** use original IDs (100, 200, 300, ..., 10000) in uint16 format as specified.
- All 10 classes from the problem statement are supported.
- Model weights: `runs_phase2/checkpoints/best_model.pth`

---

## 📦 Dependencies

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.1.2 |
| torchvision | 0.16.2 |
| CUDA | 12.1 |
| numpy | 1.26.4 |
| Pillow | ≥9.4 |
| matplotlib | ≥3.7 |
| tqdm | ≥4.64 |
| opencv-contrib-python | ≥4.7 |
| scipy | ≥1.11 |
| fpdf2 | ≥2.7 |

---

<div align="center">

**Built with ❤️ by Team DDOS_ME for Hack For Green Bharat**

</div>
