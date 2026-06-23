# Mob Radar: Real-Time Minecraft Object Detection

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset/data)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26n-orange.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-lightgrey.svg)](https://onnxruntime.ai/)

**Mob Radar** is an end-to-end computer vision pipeline for detecting and tracking Minecraft mobs in real-time. This project uses a fine-tuned **YOLO26 Nano (`yolo26n`)** model to keep latency low. Paired with a custom data extraction pipeline and an ONNX inference script, the model runs at roughly 60 FPS on a GPU and maintains around 20 FPS on a standard CPU.

<p align="center">
  <video src="https://github.com/user-attachments/assets/57bc4cf5-27f5-47d9-97eb-a94b6528c02a" autoplay loop muted playsinline width="100%"></video>
  <br>
  <i>YOLO26n tracking mobs across different game biomes and lighting conditions.</i>
</p>

## Key Features

- **Edge Deployment:** Uses the lightweight YOLO26n model to balance compute constraints with accuracy.
- **Cross-Platform Inference:** Runs at 60 FPS via GPU execution providers and falls back to 20 FPS on CPU.
- **End-to-End Pipeline:** Automated data extraction, FFmpeg SDR color correction, and train/val splitting.
- **Dataset Strategy:** Trained on a 2,585-image custom dataset, with 24% background images included as hard negatives to reduce false positives.

## Tech Stack

- **Computer Vision & ML:** Ultralytics (YOLO), ONNX Runtime (GPU & CPU), OpenCV
- **Data Engineering:** FFmpeg (Video Filtering), Python (Numpy)
- **Tooling:** `uv` for dependency management, Makefile for pipeline automation

## Dataset: Minecraft Mobs YOLO

The model groups morphological mob variants into 5 main macro-classes. The dataset is publicly available on Kaggle.

- **Classes:** `Creeper`, `Skeleton` (incl. Wither/Stray/Bogged), `Spider` (incl. Cave), `Zombie` (incl. Drowned/Husk), `Enderman`.
- [**Download the Dataset on Kaggle**](https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset/data)

## Performance Metrics

Evaluated on a 20% holdout validation set:

| Metric        | Score   | Description                                        |
| :------------ | :------ | :------------------------------------------------- |
| **mAP50**     | `0.925` | Mean Average Precision at 50% IoU.                 |
| **mAP50-95**  | `0.758` | Mean Average Precision across 50-95% IoU.          |
| **Precision** | `0.951` | Low rate of false positive detections.             |
| **Recall**    | `0.859` | High rate of successfully identifying actual mobs. |

_Note: These metrics include evaluations on the 24% background image split to test robustness against false positives._

## Quick Start

This project uses a `Makefile` to manage the pipeline.

**1. Data Pipeline**

```bash
make preprocess_videos  # Standardizes raw .webm to SDR .mp4 via FFmpeg
make extract_frames     # Extracts 1 FPS frames from raw video
make split_data         # Automates 80/20 train/val dataset splits

```

**2. Model Training & Export**

```bash
make train              # Fine-tunes the YOLO model
make export_model       # Exports best.pt to an optimized ONNX graph

```

**3. Real-Time Inference**

```bash
make deploy             # Launches the ONNX-backed real-time tracker

```

## Architecture Highlights

- `src/data/`: Scripts for raw video ingestion, normalization, and dataset splitting.
- `src/model/`: Training configurations (Cosine LR, Mosaic augmentation, layer freezing).
- `src/inference/`: ONNX Runtime scripts for real-time tracking without PyTorch overhead.

---

_Created by Draco TLW - Feel free to reach out for collaborations!_
