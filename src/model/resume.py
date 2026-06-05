from ultralytics.models import YOLO

from src.utils.logger import logging, setup_logger

setup_logger()

CHECKPOINT_PATH = "runs/detect/mob_radar_26n_dv4_1024_f11/weights/last.pt"


def main():
    logging.info(f"Initializing YOLO recovery from checkpoint: {CHECKPOINT_PATH}")

    try:
        model = YOLO(CHECKPOINT_PATH)

        logging.info("Resuming training pipeline...")
        model.train(resume=True)

        logging.info("Training pipeline recovered and finished successfully!")

    except FileNotFoundError:
        logging.error(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Did the path change?"
        )
    except Exception as e:
        logging.error(f"Failed to resume training: {e}")


if __name__ == "__main__":
    main()
