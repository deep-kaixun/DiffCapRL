# DiffCap-RL

Image/video captioning with [DiffCap-RL 4B](https://huggingface.co/fengyao123/DiffCapRL-4B), Offline QA construction, and [VideoCapArena](https://huggingface.co/datasets/fengyao123/VideoCapArena) evaluation.

## Caption inference

```bash
pip install -r requirements-infer.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('fengyao123/diffcaprl4b', local_dir='models/4B')"

python -m diffcap.infer --model models/4B --input /path/to/image.jpg --output runs/image
python -m diffcap.infer --model models/4B --input /path/to/video.mp4 --output runs/video
```

Batch input is JSONL; media paths are relative to the manifest:

```json
{"id":"image_001","media":["images/001.jpg"]}
{"id":"video_001","media":["videos/001.mp4"]}
```

Outputs are `captions.jsonl`, `run.json`, and `status.json`. Repeating the same command resumes missing samples. `status.json` reports captions that reached the output token limit; inspect their `finish_reason` before evaluation.

## Offline QA

This release includes an updated Offline QA pipeline.

```bash
pip install -r requirements.txt
python offline_qa/run.py --input /path/to/media.jsonl \
  --config /path/to/config.json --run-dir runs/offline_qa --preflight
```

Workflow: prepare → generate → validate → audit and blind answer → visual verification → 8-caption calibration → 6–8 question sampling → export. See [Offline QA](offline_qa/README.md).

## VideoCapArena

Choose the better caption for each video and save JSONL predictions with `id` and `best_idx` (0 for A, 1 for B):

```bash
bash videocaparena/run.sh --data-root /path/to/VideoCapArena \
  --predictions predictions.jsonl --output runs/arena_summary.json
```

The scorer compares predictions with human preferences. Missing predictions and ties count as incorrect. Download and format details are in [VideoCapArena](videocaparena/README.md).

## TODO

- [x] Release DiffCap-RL weights.
- [x] Provide image/video caption inference.
- [x] Provide the Offline QA pipeline.
- [x] Release VideoCapArena benchmark and provide scoring.
- [ ] Release training data & script

## License

Code and 4B weights: Apache-2.0. The model is fine-tuned from [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct); retain its license and attribution.
