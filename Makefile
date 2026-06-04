.PHONY: preprocess_videos, extract_frames, split_data, analyze_data, train, inference

preprocess_videos:
	uv run python -m src.data.preprocess_videos

extract_frames:
	uv run python -m src.data.extract_frames

split_data:
	uv run python -m src.data.split_data

analyze_data:
	uv run python -m src.data.analyze_data

train:
	uv run python -m src.model.train

inference:
	uv run python -m src.model.inference
