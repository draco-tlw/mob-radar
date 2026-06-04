from pathlib import Path

import cv2

from src.utils.logger import logging, setup_logger

setup_logger()


VIDEOS_DIR = Path("data/raw/videos")
OUTPUT_DIR = Path("data/raw/frames")


def extract_frames_from_video(video_path: Path, output_path: Path):
    video_name_stem = video_path.stem
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path.name}")
        return

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = round(raw_fps) if raw_fps > 0 else 30

    logging.info(f"Processing '{video_path.name}' | Detected FPS: {fps}")

    frame_index = 0
    saved_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Extract 1 frames per second
        if frame_index % fps == 0:
            file_name = f"{video_name_stem}_frame_{saved_count:04d}.jpg"
            target_file_path = output_path / file_name

            cv2.imwrite(str(target_file_path), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    logging.info(f"Extracted {saved_count} frames from {video_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEOS_DIR.exists():
        logging.error(f"Input video directory not found: {VIDEOS_DIR}")
        return

    video_files = list(VIDEOS_DIR.glob("*.mp4"))

    if not video_files:
        logging.warning(f"No '.mp4' video files found in {VIDEOS_DIR}")
        return

    logging.info(f"Beginning frame extraction batch for {len(video_files)} video(s)...")
    logging.info(f"Output target directory: {OUTPUT_DIR.resolve()}")

    for video_file in video_files:
        extract_frames_from_video(video_file, OUTPUT_DIR)

    logging.info(f"Batch processing finished. All frames saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
