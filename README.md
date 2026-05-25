
# Scientific Image Forgery Detection

A deep learning pipeline for pixel-level detection of forged regions in scientific images, built with PyTorch and Segmentation Models PyTorch (SMP). Trained and evaluated on the [RECODAI dataset](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection) via Kaggle.

---

##  Overview

Scientific image manipulation such as copy-move forgery, splicing, and region duplication in research papers — is a growing concern for academic integrity. This project tackles the problem as a **binary semantic segmentation** task, where the model predicts a pixel-level mask identifying forged regions within an image.

---

## Dataset — RECODAI

| Split | Description |
|---|---|
| `train_images/forged/` | Forged scientific images with ground truth masks |
| `train_images/authentic/` | Authentic images (no masks) |
| `train_masks/` | Binary `.npy` mask files per forged image |
| `supplemental_images/` | Additional images for training |
| `supplemental_masks/` | Corresponding supplemental masks |
| `test_images/` | Unlabelled test images for submission |

Only forged images have corresponding masks. Image-mask pairs are matched by filename stem before training.

---

##  Architecture

```
Input Image (256×256×3)
        ↓
  EfficientNet-B0 Encoder  ← ImageNet pretrained
        ↓
    UNet Decoder
        ↓
  Output Mask (256×256×1)  ← Binary sigmoid output
```

- Encoder: `EfficientNet-B0` (ImageNet pretrained weights)
- Decoder: `UNet`
- Framework: `segmentation-models-pytorch`
- Output: Binary segmentation mask (forged = 1, authentic = 0)

---

##  Pipeline

```
Raw Images & .npy Masks
        ↓
  Filename Stem Matching
        ↓
  Train / Val Split (80/20)
        ↓
  Albumentations Augmentations
  (Resize 256×256, HFlip, VFlip, Normalize)
        ↓
  ForgeryDataset + DataLoader (batch=8)
        ↓
  UNet + EfficientNet-B0
        ↓
  Dice Loss + Adam (lr=1e-4)
        ↓
  Training (5 Epochs)
        ↓
  Evaluation + RLE Submission
```

---

##  Results

Evaluated on the held-out validation set after 5 epochs of training:

| Metric | Score |
|---|---|
| IoU (Jaccard) | 0.3801 |
| Dice / F1 | 0.5508 |
| Precision | 0.4996 |
| Recall | 0.6138 |
| Pixel Accuracy | 0.9449 |

> Pixel accuracy is high due to class imbalance (most pixels are authentic background). IoU and Dice are the primary metrics for segmentation quality.

---

##  Project Structure

```
├── notebook.ipynb          # Main Kaggle notebook
├── submission.csv          # RLE-encoded predictions on test set
└── README.md
```

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| PyTorch | Model training |
| Segmentation Models PyTorch | UNet + EfficientNet |
| Albumentations | Image augmentation |
| OpenCV | Image loading |
| NumPy | Mask processing |
| scikit-learn | Train/val split |
| Pandas | Submission CSV |
| Kaggle GPU (T4) | Training environment |

---

## Future Improvements

- Train for more epochs (20–50) with a cosine LR scheduler
- Try stronger encoders: `EfficientNet-B3`, `ResNet50`, `MiT-B2`
- Add supplemental images to increase training data
- Use combined loss: `Dice + BCE` for better convergence
- Apply test-time augmentation (TTA) for better predictions
- Tune decision threshold (currently 0.5) using F1 curve

---

##  License

This project is for academic and educational purposes. Dataset usage is subject to the [RECODAI competition rules](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection).
