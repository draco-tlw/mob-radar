from ultralytics.models import YOLO


def main():
    MODEL_PATH = (
        "runs/detect/mob_radar_v26n_dv5_sz1024_f11_lr1e-2_lrf1e-2_cos/weights/best.pt"
    )

    model = YOLO(MODEL_PATH)

    model.export(format="onnx", imgsz=[576, 1024], half=True, simplify=True)


if __name__ == "__main__":
    main()
