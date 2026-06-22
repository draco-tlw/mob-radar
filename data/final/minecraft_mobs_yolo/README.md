# Minecraft Mobs Object Detection Dataset (YOLO Format)

## Overview

This dataset contains 2,585 images of Minecraft mobs annotated with bounding boxes. It is pre-split into Train (80%) and Validation (20%) sets and is formatted for YOLO object detection models.

**Note on Background Images:** Approximately 24% of the dataset (625 images) consists of pure background images with zero bounding boxes. This data is included for hard negative mining to help the model reduce false positive detections.

## Class Taxonomy (Macro-Classes)

Morphological variants are consolidated into five primary macro-classes to group similar shapes together.

The dataset contains the following distribution:

- **`0: creeper`** (594 instances)
  - _Includes:_ <img src="assets/Creeper.webp" style="vertical-align: middle; height: 48px"> Standard Creeper
- **`1: skeleton`** (959 instances)
  - _Includes:_ <img src="assets/Skeleton.webp" style="vertical-align: middle; height: 48px"> Standard Skeleton, <img src="assets/Wither_Skeleton.webp" style="vertical-align: middle; height: 48px"> Wither Skeleton, <img src="assets/Bogged.png" style="vertical-align: middle; height: 48px"> Bogged, <img src="assets/Stray.webp" style="vertical-align: middle; height: 48px"> Stray
- **`2: spider`** (491 instances)
  - _Includes:_ <img src="assets/Spider.webp" style="vertical-align: middle; height: 48px"> Standard Spider, <img src="assets/Cave_Spider.webp" style="vertical-align: middle; height: 48px"> Cave Spider
- **`3: zombie`** (1,115 instances)
  - _Includes:_ <img src="assets/Zombie.webp" style="vertical-align: middle; height: 48px"> Standard Zombie, <img src="assets/Drowned.webp" style="vertical-align: middle; height: 48px"> Drowned, <img src="assets/Husk.webp" style="vertical-align: middle; height: 48px"> Husk
- **`4: enderman`** (177 instances)
  - _Includes:_ <img src="assets/Enderman.png" style="vertical-align: middle; height: 76px; margin: -12px -24px"> Standard Enderman
