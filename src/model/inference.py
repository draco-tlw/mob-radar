import cv2
from ultralytics.models import YOLO


def main():

    model = YOLO("runs/detect/mob_radar_26n_dv4_1024_f11/weights/best.pt")

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
