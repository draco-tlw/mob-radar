import cv2
from ultralytics.models import YOLO

MODEL_PATH = (
    "runs/detect/mob_radar_v26n_dv5.3_sz1024_f11_lr1e-2_lrf1e-2_cos/weights/best.pt"
)


def main():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(2)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.predict(frame, imgsz=1024, conf=0.5, verbose=False)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLO26 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
