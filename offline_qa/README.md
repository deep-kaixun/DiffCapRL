# Offline QA

This release includes an updated Offline QA pipeline.

Run commands from the repository root. Install `requirements.txt`; frame-based video input also requires `ffmpeg` and `ffprobe`.

## Workflow

| Stage | Function |
|---|---|
| 1 · Prepare | Validate media and deduplicate by content. |
| 2 · Generate | Generate up to 10 visual QA candidates. |
| 3 · Validate | Check four choices and references; shuffle choices consistently. |
| 4 · Audit and blind answer | Check every question for stem leakage and duplicate facts; remove leakage sources and filter questions answerable without media. |
| 5 · Verify | Check answers, uniqueness, and localized visual evidence using distinct models. |
| 6 · Calibrate | Generate 8 captions and answer QA using captions alone. |
| 7 · Sample | Select 6–8 questions with difficulty and category quotas. |
| 8 · Export | Write `qa_offline` records and a separate review queue. |

Default `full_teacher` verification requires Weak + Teacher 1 or both Teachers to support the original answer. Configure three distinct verifier model IDs. Unresolved questions enter review; references are not automatically relabeled.

Difficulty is easy when all 8 answers are correct, hard when none are correct, and medium otherwise. Sampling targets easy/medium/hard at 20%/60%/20%, redistributing when a bucket lacks questions. Difficulty reflects the captioner and solver combination.

## Inputs and configuration

Local paths are relative to the input JSONL:

```json
{"id":"image_001","media_type":"image","media_path":"media/001.jpg","source":"my_dataset"}
{"id":"video_001","media_type":"video","media_path":"media/001.mp4","source":"my_dataset"}
```

Copy [config.example.json](config.example.json) and set supported model IDs. Configure endpoints and credentials through environment variables:

| Role | Endpoint | Credential |
|---|---|---|
| QA maker | `QA_MAKER_BASE_URL` | `QA_MAKER_API_KEY` |
| Blind / audit / QA solver | `QA_TEXT_BASE_URL` | `QA_TEXT_API_KEY` |
| Captioner | `QA_CAPTIONER_BASE_URL` | `QA_CAPTIONER_API_KEY` |
| Weak verifier | `QA_WEAK_VERIFIER_BASE_URL` | `QA_WEAK_VERIFIER_API_KEY` |
| Teacher 1 | `QA_TEACHER_1_BASE_URL` | `QA_TEACHER_1_API_KEY` |
| Teacher 2 | `QA_TEACHER_2_BASE_URL` | `QA_TEACHER_2_API_KEY` |

Teacher 2 reads its model ID from `QA_TEACHER_2_MODEL`. Services use OpenAI-compatible Chat Completions. Configuration is loaded from `.env` or a private dotenv file selected by `QA_ENV_FILE`; existing environment variables take priority.

Video input supports `url`, `inline` MP4, or timestamped `frames`, configurable globally or per client. Frames use 2 FPS with a 48-frame cap. URL input requires a JSONL mapping supplied with `--url-map`:

```json
{"id":"video_001","url":"https://your-media-host/001.mp4"}
```

Use audio-free source videos for strictly visual QA. Check generated evidence against the source before using QA as rewards.

## Run

```bash
python offline_qa/run.py --input /path/to/media.jsonl \
  --config /path/to/config.json --run-dir runs/offline_qa --preflight
```

Remove `--preflight` to execute. Add `--resume` for the same inputs and settings; use a new output directory after changes. Each numbered script also runs independently.

Outputs include stage JSONL files, `train.jsonl`, `train.json`, `summary.json`, and unresolved questions in `5_review_queue.jsonl`.

An optional shared request budget counts network attempts, including retries:

```json
"request_budget": {"ledger":"runs/request_budget.sqlite3","limit":80}
```
