from ultralytics.models import YOLO

DATASET_DIR = "data/final/minecraft_mobs_yolo/data.yaml"

HYPERPARAMETERS = {
    "epochs": 200,
    "imgsz": 1024,
    "rect": True,
    "patience": 30,
    "freeze": 11,
    "optimizer": "AdamW",
    "lr0": 1e-2,
    "lrf": 1e-2,
    "cos_lr": True,
    "batch": 16,
    "deterministic": False,
}

MODEL_VERSION = 26
MODEL_SIZE = "n"
MODEL_PATH = f"yolo{MODEL_VERSION}{MODEL_SIZE}.pt"

DATASET_VERSION = 5


def generate_run_name() -> str:
    """Generates a standardized, readable run name for tracking experiments."""
    img_size = HYPERPARAMETERS["imgsz"]
    frozen_layers = HYPERPARAMETERS["freeze"]

    lr_val = HYPERPARAMETERS["lr0"]
    lr_str = f"{lr_val:.0e}".replace("0", "")
    lrf_val = HYPERPARAMETERS["lrf"]
    lrf_str = f"{lrf_val:.0e}".replace("0", "")

    scheduler = "cos" if HYPERPARAMETERS.get("cos_lr") else "lin"

    return (
        f"mob_radar_"
        f"v{MODEL_VERSION}{MODEL_SIZE}_"
        f"dv{DATASET_VERSION}_"
        f"sz{img_size}_"
        f"f{frozen_layers}_"
        f"lr{lr_str}_"
        f"lrf{lrf_str}_"
        f"{scheduler}"
    )


def main():
    model = YOLO(MODEL_PATH)

    run_name = generate_run_name()

    _ = model.train(
        data=DATASET_DIR,
        name=run_name,
        **HYPERPARAMETERS,
    )


if __name__ == "__main__":
    main()
