import subprocess
from pathlib import Path

from src.utils.logger import logging, setup_logger

setup_logger()

VIDEOS_DIR = Path("data/raw/videos")

FFMPEG_VIDEO_FILTER = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
    "eq=saturation=1.25:contrast=1.05"
)


def check_ffmpeg_installed():
    try:
        subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except FileNotFoundError:
        return False


def convert_video(input_path: Path):
    output_path = input_path.with_suffix(".mp4")

    if output_path.exists():
        logging.info(f"Skipped: '{output_path.name}' already exists.")
        return

    logging.info(f"Processing: Converting '{input_path.name}' to SDR MP4...")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        FFMPEG_VIDEO_FILTER,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-c:a",
        "aac",
        str(output_path),
    ]

    try:
        subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True
        )
        logging.info(f"Saved formatted video to '{output_path.name}'")

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed on '{input_path.name}'")
        logging.error(f"Details: {e}")


def main():
    if not check_ffmpeg_installed():
        logging.critical("FFmpeg is not installed. Halting execution.")
        return

    if not VIDEOS_DIR.exists():
        logging.error(f"Video directory not found: {VIDEOS_DIR}")
        return

    webm_files = list(VIDEOS_DIR.glob("*.webm"))

    if not webm_files:
        logging.info(f"No raw .webm files found in {VIDEOS_DIR}")
        return

    logging.info(f"Found {len(webm_files)} video(s) for pre-processing.")

    for video_file in webm_files:
        convert_video(video_file)

    logging.info("All videos have been standardized to SDR MP4.")


if __name__ == "__main__":
    main()
