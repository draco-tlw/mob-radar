# Mob Radar: Real-Time Minecraft Object Detection

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset/data)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26n-orange.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-lightgrey.svg)](https://onnxruntime.ai/)

**Mob Radar** is an end-to-end computer vision pipeline that detects and tracks Minecraft mobs in real-time. To ensure ultra-low latency and edge-device compatibility, this project fine-tunes the highly lightweight **YOLO26 Nano (`yolo26n`)** architecture. Coupled with a custom data extraction pipeline and an optimized ONNX inference engine, the model achieves a fluid 60 FPS utilizing GPU execution providers, and a highly viable 20 FPS fallback on standard CPUs.

<p align="center">
  <video src="assets/final_demo.mp4" autoplay loop muted playsinline width="100%"></video>
  <br>
  <i>YOLO26n model dynamically generalizing tracking across diverse game biomes and varying environmental color palettes.</i>
</p>

## Key Features

- **Edge-Optimized Deployment:** Fine-tuned the lightweight **YOLO Nano** variant, balancing tight computational constraints with high detection accuracy.
- **Cross-Platform Performance:** Achieves a fluid 60 FPS utilizing GPU execution providers, while remaining highly viable for edge devices with a **20 FPS fallback on standard CPUs**.
- **End-to-End ML Pipeline:** Fully automated data extraction, preprocessing (FFmpeg SDR color correction), and train/val splitting.
- **Custom Dataset Strategy:** Trained on a custom 2,585-image dataset with a 24% hard-negative background mining ratio to aggressively reduce false positives.

## Tech Stack

- **Computer Vision & ML:** Ultralytics (YOLO), ONNX Runtime (GPU), OpenCV
- **Data Engineering:** FFmpeg (Video Filtering), Python (Numpy)
- **Tooling:** `uv` for dependency management, Makefile for pipeline automation

## Dataset: Minecraft Mobs YOLO

The model is trained to identify 5 "macro-classes" consolidating various morphological mob variants. The dataset is publicly available on Kaggle.

- **Classes:** `Creeper`, `Skeleton` (incl. Wither/Stray/Bogged), `Spider` (incl. Cave), `Zombie` (incl. Drowned/Husk), `Enderman`.
- [**Download the Dataset on Kaggle**](https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset/data)

## Quick Start

This project uses a `Makefile` to orchestrate the entire ML lifecycle.

**1. Data Pipeline**

```bash
make preprocess_videos  # Standardizes raw .webm to SDR .mp4 via FFmpeg
make extract_frames     # Extracts 1 FPS frames from raw video
make split_data         # Automates 80/20 train/val dataset splits
```

**2. Model Training & Export**

```bash
make train              # Fine-tunes the YOLO model using cosine learning rates
make export_model       # Exports best.pt to an optimized, half-precision ONNX graph

```

**3. Real-Time Inference**

```bash
make deploy             # Launches the ONNX-backed real-time webcam/video tracker

```

## Architecture Highlights

- `src/data/`: Scripts for handling raw video ingestion, normalization, and strict dataset splitting.
- `src/model/`: Training configurations featuring optimized hyperparameters (Cosine LR, Mosaic augmentation, frozen layers).
- `src/inference/`: Production-ready scripts bypassing PyTorch overhead in favor of lightweight ONNX Runtime execution.

---

_Created by Draco TLW - Feel free to reach out for collaborations!_
