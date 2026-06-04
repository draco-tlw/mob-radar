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

        img_dst = OUTPUT_DIR / Path(subset) / "images" / img.name
        txt_dst = OUTPUT_DIR / Path(subset) / "labels" / txt_name

        shutil.copy(img, img_dst)
        shutil.copy(txt, txt_dst)


def main():
    logging.info("Starting dataset split pipeline...")
    for subset in ["train", "val"]:
        for cat in ["images", "labels"]:
            dir_path = OUTPUT_DIR / subset / cat

            if dir_path.exists():
                shutil.rmtree(dir_path)

            dir_path.mkdir(parents=True, exist_ok=True)

    imgs = list(SOURCE_DIR.glob("*.jpg"))
    if not imgs:
        logging.error(f"No images found in {SOURCE_DIR}! Please check the path.")
        return

    random.seed(42)
    random.shuffle(imgs)

    split_index = int(len(imgs) * SPLIT_RATIO)

    train_imgs = imgs[:split_index]
    val_imgs = imgs[split_index:]

    logging.info(f"Total images found: {len(imgs)}")
    logging.info(
        f"Copying {len(train_imgs)} images to 'train' ({SPLIT_RATIO*100:.2f}%)..."
    )
    copy_files(train_imgs, "train")

    logging.info(
        f"Copying {len(val_imgs)} images to 'val' ({(1-SPLIT_RATIO)*100:.2f}%)..."
    )
    copy_files(val_imgs, "val")

    logging.info(f"Splitting complete! Dataset is ready at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
