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
    logging.info("Starting dataset split & update pipeline...")

    for subset in ["train", "val"]:
        for cat in ["images", "labels"]:
            dir_path = OUTPUT_DIR / subset / cat
            dir_path.mkdir(parents=True, exist_ok=True)

    source_imgs = list(SOURCE_DIR.glob("*.jpg"))
    if not source_imgs:
        logging.error(f"No images found in {SOURCE_DIR}! Please check the path.")
        return

    source_img_names = {img.name for img in source_imgs}

    seen_imgs: set[str] = set()
    purged_count = 0

    for subset in ["train", "val"]:
        dst_image_dir = OUTPUT_DIR / subset / "images"
        dst_label_dir = OUTPUT_DIR / subset / "labels"

        dst_imgs = dst_image_dir.glob("*.jpg")
        for img in dst_imgs:
            if img.name in source_img_names:
                seen_imgs.add(img.name)
            else:
                img.unlink()
                label = dst_label_dir / (img.stem + ".txt")
                label.unlink(missing_ok=True)
                purged_count += 1

    if purged_count > 0:
        logging.warning(
            f"Cleaned up {purged_count} deleted images/labels before splitting."
        )

    new_imgs = [img for img in source_imgs if img.name not in seen_imgs]

    if not new_imgs:
        logging.info("No new images to split. Dataset is already up-to-date!")
        return

    random.seed(42)
    random.shuffle(new_imgs)

    split_index = int(len(new_imgs) * SPLIT_RATIO)

    train_imgs = new_imgs[:split_index]
    val_imgs = new_imgs[split_index:]

    logging.info(f"Discovered {len(new_imgs)} new images to process.")
    logging.info(
        f"Splitting {len(train_imgs)} images into 'train' ({SPLIT_RATIO*100:.0f}%)..."
    )
    copy_files(train_imgs, "train")

    logging.info(
        f"Splitting {len(val_imgs)} images into 'val' ({(1-SPLIT_RATIO)*100:.0f}%)..."
    )
    copy_files(val_imgs, "val")

    logging.info(f"Splitting complete! Dataset is ready at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
