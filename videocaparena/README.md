# VideoCapArena

[Download VideoCapArena](https://huggingface.co/datasets/lalahaha1/VideoCapArena): 100 English caption pairs with human preferences.

```bash
pip install huggingface-hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('lalahaha1/VideoCapArena', repo_type='dataset', local_dir='datasets/videocaparena')"
```

`data/test.jsonl` contains the video `media_path`, two `captions`, and human `best_idx`. Choose a caption without exposing the human label to your method: **0 = captions[0] (A), 1 = captions[1] (B)**.

Save choices to `predictions.jsonl`, preserving the string IDs:

```json
{"id":"001","best_idx":0}
{"id":"002","best_idx":1}
```

Score with Python's standard library:

```bash
bash videocaparena/run.sh --data-root datasets/videocaparena \
  --predictions predictions.jsonl --output runs/arena_summary.json
```

Accuracy = matching choices / total ground-truth pairs. Missing predictions and `best_idx: null` (tie or abstention) count as incorrect. Duplicate IDs, unknown IDs, and invalid indices are rejected. The summary includes overall and per-category accuracy. Add `--check-media` to verify local videos.
