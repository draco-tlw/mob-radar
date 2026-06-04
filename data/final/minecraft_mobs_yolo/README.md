# Minecraft Mobs Object Detection Dataset (YOLO Format)

## Overview

This dataset contains 1,462 images of Minecraft mobs (Creeper, Skeleton, Spider, Zombie) annotated with bounding boxes. It is pre-split into Train (80%) and Validation (20%) sets and is formatted specifically for YOLOv8/YOLO11 object detection models.

## Classes

To optimize the model's bounding-box accuracy and ensure robust detection across various Minecraft biomes and lighting conditions, morphological variants were consolidated into four primary macro-classes.

The dataset contains the following distribution:

- **`0: creeper`** (316 instances)
  - _Includes:_ <img src="assets/Creeper.webp" width="20" align="top"> Standard Creeper
- **`1: skeleton`** (650 instances)
  - _Includes:_ <img src="./assets/Skeleton.webp" width="20" align="top"> Standard Skeleton, <img src="./assets/Wither_Skeleton.webp" width="20" align="top"> Wither Skeleton, <img src="./assets/Bogged.png" width="20" align="top"> Bogged, <img src="./assets/Stray.webp" width="20" align="top"> Stray
- **`2: spider`** (346 instances)
  - _Includes:_ <img src="./assets/Spider.webp" width="20" align="top"> Standard Spider, <img src="./assets/Cave_Spider.webp" width="20" align="top"> Cave Spider
- **`3: zombie`** (721 instances)
  - _Includes:_ <img src="./assets/Zombie.webp" width="20" align="top"> Standard Zombie, <img src="./assets/Drowned.webp" width="20" align="top"> Drowned, <img src="./assets/Husk.webp" width="20" align="top"> Husk
