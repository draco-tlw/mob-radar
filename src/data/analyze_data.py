from collections import Counter
from pathlib import Path

import yaml

from src.utils.logger import logging, setup_logger

setup_logger()

DATASET_DIR = Path("data/final/minecraft_mobs_yolo")
YAML_FILE = DATASET_DIR / "data.yaml"


def analyze_subset(subset_name: str, class_names: dict):
    images_dir = DATASET_DIR / subset_name / "images"
    labels_dir = DATASET_DIR / subset_name / "labels"

    if not images_dir.exists():
        logging.warning(f"Subset directory not found: {images_dir}")
        return

    total_images = 0
    background_images = 0
    class_counts = Counter()

    for img_path in images_dir.glob("*.jpg"):
        total_images += 1
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists() or label_path.stat().st_size == 0:
            background_images += 1
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()
            if not lines:
                background_images += 1
            else:
                for line in lines:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

    if total_images == 0:
        logging.warning(f"No images found in the '{subset_name}' set.")
        return

    report = []
    report.append(f"--- [{subset_name.upper()} SET PROFILE] ---")
    report.append(f"Total Images:      {total_images}")
    report.append(
        f"Background Images: {background_images} ({(background_images/total_images)*100:.1f}%)"
    )
    report.append("Bounding Boxes:")

    for class_id, count in sorted(class_counts.items()):
        name = class_names.get(class_id, f"Unknown (ID: {class_id})")
        report.append(f"  - {name.capitalize()}: {count}")

    logging.info("\n" + "\n".join(report))


def main():
    logging.info("Starting dataset analysis pipeline...")

    if not YAML_FILE.exists():
        logging.error(f"Dataset configuration not found: {YAML_FILE}")
        return

    with open(YAML_FILE, "r") as f:
        data_config = yaml.safe_load(f)

    names = data_config.get("names", {})
    if isinstance(names, list):
        class_names = {i: name for i, name in enumerate(names)}
    else:
        class_names = names

    analyze_subset("train", class_names)
    analyze_subset("val", class_names)

    logging.info("Dataset analysis complete.")


if __name__ == "__main__":
    main()
