from ultralytics.models import YOLO

DATASET_DIR = "data/final/minecraft_mobs_yolo/data.yaml"

HYPERPARAMETERS = {
    "epochs": 200,
    "imgsz": 1024,
    "patience": 30,
    "freeze": 11,
    "optimizer": "AdamW",
    "lr0": 1e-3,
    "batch": 16,
}


def main():

    model = YOLO("yolo26n.pt")

    _ = model.train(
        data=DATASET_DIR,
        lr0=HYPERPARAMETERS["lr0"],
        imgsz=HYPERPARAMETERS["imgsz"],
        rect=True,
        optimizer=HYPERPARAMETERS["optimizer"],
        batch=HYPERPARAMETERS["batch"],
        epochs=HYPERPARAMETERS["epochs"],
        freeze=HYPERPARAMETERS["freeze"],
        patience=HYPERPARAMETERS["patience"],
        deterministic=False,
    )


if __name__ == "__main__":
    main()
