import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.logger import logging, setup_logger

setup_logger()


MODEL_PATH = (
    "runs/detect/mob_radar_v26n_dv5_sz1024_f11_lr1e-2_lrf1e-2_cos/weights/best.onnx"
)
CLASSES = ["Creeper", "Skeleton", "Spider", "Zombie", "Enderman"]
COLORS = [(0, 255, 0), (200, 200, 200), (0, 0, 150), (0, 100, 0), (150, 0, 150)]
IMGSZ = (1024, 576)
CONF_THRESHOLD = 0.5
TARGET_FPS = 60
CAMERA_I = 2


def initialize_onnx_session(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def initialize_camera(camera_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    return cap


def crop_to_16_9(frame: np.ndarray) -> np.ndarray:
    """Cuts the largest possible 16:9 rectangle out of the center of ANY frame."""
    h, w = frame.shape[:2]
    target_ratio = 16 / 9
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x_offset = (w - new_w) // 2
        return frame[:, x_offset : x_offset + new_w]
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        y_offset = (h - new_h) // 2
        return frame[y_offset : y_offset + new_h, :]

    return frame


def preprocess(cropped_frame: np.ndarray) -> np.ndarray:
    """Shrinks the 16:9 crop to ONNX size and converts to FP16 tensor."""
    img = cv2.resize(cropped_frame, IMGSZ)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return (img / 255.0).astype(np.float32)


def postprocess(predictions: np.ndarray) -> List[Dict]:
    """Filters baked-in ONNX NMS outputs (1, 300, 6) into clean boxes."""
    preds = predictions[0]

    scores = preds[:, 4]
    mask = scores > CONF_THRESHOLD
    valid_preds = preds[mask]

    if len(valid_preds) == 0:
        return []

    results = []
    for pred in valid_preds:
        x1, y1, x2, y2, score, class_id = pred

        x_min = int(x1)
        y_min = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)

        c_id = int(class_id)

        if c_id < 0 or c_id >= len(CLASSES):
            continue

        results.append(
            {
                "box": [x_min, y_min, w, h],
                "score": float(score),
                "class_id": c_id,
            }
        )
    return results


def draw_overlay(
    cropped_frame: np.ndarray, detections: List[Dict], fps: float
) -> np.ndarray:
    """Draws boxes directly onto the 16:9 cropped frame."""
    display_frame = cropped_frame.copy()
    h, w = display_frame.shape[:2]

    scale_x = w / IMGSZ[0]
    scale_y = h / IMGSZ[1]

    for det in detections:
        x, y, bw, bh = det["box"]

        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + bw) * scale_x)
        y2 = int((y + bh) * scale_y)

        class_id = det["class_id"]
        label = f"{CLASSES[class_id]} {det['score']:.2f}"
        color = COLORS[class_id]

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return display_frame


def cleanup_resources(cap: Optional[cv2.VideoCapture]) -> None:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def run_pipeline(camera_id: int) -> None:
    cap = None
    try:
        session = initialize_onnx_session(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        cap = initialize_camera(camera_id)

        logging.info("Pipeline Online. Press 'q' to terminate.")
        FRAME_TARGET_TIME = 1.0 / float(TARGET_FPS)

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                continue

            cropped_frame = crop_to_16_9(frame)
            tensor = preprocess(cropped_frame)

            raw_outputs = session.run(None, {input_name: tensor})
            output_tensor = np.asarray(raw_outputs[0])
            detections = postprocess(output_tensor)

            elapsed_time = time.time() - start_time

            time_left_in_budget = FRAME_TARGET_TIME - elapsed_time
            if time_left_in_budget > 0:
                time.sleep(time_left_in_budget)

            final_fps = 1.0 / (time.time() - start_time + 1e-9)
            final_frame = draw_overlay(cropped_frame, detections, final_fps)

            cv2.imshow("Mob Radar", final_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Crash: {e}")
    finally:
        cleanup_resources(cap)


if __name__ == "__main__":
    run_pipeline(CAMERA_I)
