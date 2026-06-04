.PHONY: preprocess_videos, extract_frames, split_data, train, track, predict

preprocess_videos:
	uv run python -m src.data.preprocess_videos

extract_frames:
	uv run python -m src.data.extract_frames

split_data:
	uv run python -m src.data.split_data

train:
	uv run python -m src.model.train

track:
	uv run python -m src.model.track

predict:
	uv run python -m src.model.predict
