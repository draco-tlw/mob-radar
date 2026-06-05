from ultralytics.models import YOLO

DATASET_DIR = "data/final/minecraft_mobs_yolo/data.yaml"

HYPERPARAMETERS = {
    "epochs": 200,
    "imgsz": 1024,
    "rect": True,
    "patience": 30,
    "freeze": 11,
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "batch": 16,
    "deterministic": False,
}

# MODEL_PATH = "yolo26n.pt"
MODEL_PATH = "runs/detect/mob_radar_26n_dv4_1024_f11/weights/best.pt"


def main():

    model = YOLO(MODEL_PATH)

    _ = model.train(
        data=DATASET_DIR,
        name="mob_radar_26n_dv4_1024_f11",
        **HYPERPARAMETERS,
    )


if __name__ == "__main__":
    main()
