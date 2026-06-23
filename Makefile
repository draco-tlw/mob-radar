.PHONY: preprocess_videos, extract_frames, split_data, analyze_data, train, resume, prototype, deploy

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

resume:
	uv run python -m src.model.resume

export_model:
	uv run python -m src.model.export

prototype:
	uv run python -m src.inference.torch_predict

deploy:
	uv run python -m src.inference.onnx_radar
