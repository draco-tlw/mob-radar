import random
import shutil
from pathlib import Path

from src.utils.logger import logging, setup_logger

setup_logger()

SOURCE_DIR = Path("data/processed/minecraft_mobs_yolo_hbb")
OUTPUT_DIR = Path("data/final/minecraft_mobs_yolo")

SPLIT_RATIO = 0.8


def copy_files(imgs: list[Path], subset: str):
    for img in imgs:
        img_stem = img.stem
        txt_name = Path(img_stem + ".txt")
        txt = SOURCE_DIR / txt_name

        img_dst = OUTPUT_DIR / subset / "images" / img.name
        txt_dst = OUTPUT_DIR / subset / "labels" / txt_name

        shutil.copy(img, img_dst)
        shutil.copy(txt, txt_dst)


def main():
    logging.info("Starting dataset split & sync pipeline...")

    for subset in ["train", "val"]:
        for cat in ["images", "labels"]:
            dir_path = OUTPUT_DIR / subset / cat
            dir_path.mkdir(parents=True, exist_ok=True)

    source_imgs = list(SOURCE_DIR.glob("*.jpg"))
    if not source_imgs:
        logging.error(f"No images found in {SOURCE_DIR}! Please check the path.")
        return

    source_img_names = {img.name for img in source_imgs}

    seen_imgs: dict[str, str] = {}  # filename: train/val
    purged_count = 0

    for subset in ["train", "val"]:
        dst_image_dir = OUTPUT_DIR / subset / "images"
        dst_label_dir = OUTPUT_DIR / subset / "labels"

        dst_imgs = dst_image_dir.glob("*.jpg")
        for img in dst_imgs:
            if img.name in source_img_names:
                seen_imgs[img.name] = subset
            else:
                img.unlink()
                label = dst_label_dir / (img.stem + ".txt")
                label.unlink(missing_ok=True)
                purged_count += 1

    if purged_count > 0:
        logging.warning(
            f"Cleaned up {purged_count} deleted images/labels before processing."
        )

    existing_imgs = [img for img in source_imgs if img.name in seen_imgs]
    new_imgs = [img for img in source_imgs if img.name not in seen_imgs]

    if existing_imgs:
        logging.info(
            f"Syncing {len(existing_imgs)} existing images to update labels/brightness while preserving split freeze..."
        )
        for img in existing_imgs:
            target_subset = seen_imgs[img.name]
            copy_files([img], target_subset)
        logging.info("Existing images and labels successfully updated!")

    if not new_imgs:
        logging.info("No new images to split. Dataset sync is complete!")
        return

    random.seed(42)
    random.shuffle(new_imgs)

    split_index = int(len(new_imgs) * SPLIT_RATIO)

    train_imgs = new_imgs[:split_index]
    val_imgs = new_imgs[split_index:]

    logging.info(f"Discovered {len(new_imgs)} completely new images to process.")
    logging.info(
        f"Splitting {len(train_imgs)} new images into 'train' ({SPLIT_RATIO*100:.0f}%)..."
    )
    copy_files(train_imgs, "train")

    logging.info(
        f"Splitting {len(val_imgs)} new images into 'val' ({(1-SPLIT_RATIO)*100:.0f}%)..."
    )
    copy_files(val_imgs, "val")

    logging.info(f"Pipeline complete! Dataset is perfectly synced at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
