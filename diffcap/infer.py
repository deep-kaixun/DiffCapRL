"""Generate image/video captions from local media or a JSONL manifest."""
import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

from .common import digest, read_jsonl, read_resumable_output, sha256, write_json

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
VIDEO_SUFFIXES = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v'}
PROMPTS = {'image': 'Describe this image in detail.', 'video': 'Describe this video in detail.'}


def check_video_backend():
    os.environ.setdefault('VLLM_VIDEO_LOADER_BACKEND', 'opencv')
    if os.environ.get('VLLM_VIDEO_LOADER_BACKEND', 'opencv') in ('opencv', 'opencv_dynamic'):
        import cv2
        from cv2 import videoio_registry
        if not hasattr(videoio_registry, 'getStreamBufferedBackends'):
            raise RuntimeError('Video inference requires OpenCV >=4.11. Install requirements-infer.txt '
                               'in a clean environment; older opencv-python packages can conflict.')
        return getattr(cv2, '__version__', 'unknown')
    return None


def install_video_decoder_fix():
    """Keep vLLM 0.11 OpenCV byte streams alive until decoding finishes."""
    import threading
    from vllm.multimodal import video

    if getattr(video, '_diffcap_stream_retention', False):
        return
    streams = threading.local()
    bytes_io = video.BytesIO

    def retained_bytes_io(*args, **kwargs):
        stream = bytes_io(*args, **kwargs)
        stack = getattr(streams, 'stack', [])
        if stack:
            stack[-1].append(stream)
        return stream

    video.BytesIO = retained_bytes_io
    registry = video.VIDEO_LOADER_REGISTRY.name2class
    for name in ('opencv', 'opencv_dynamic'):
        original = registry[name]

        def make_backend(base):
            class RetainedStreamBackend(base):
                @classmethod
                def load_bytes(cls, *args, **kwargs):
                    if not hasattr(streams, 'stack'):
                        streams.stack = []
                    streams.stack.append([])
                    try:
                        return super().load_bytes(*args, **kwargs)
                    except BaseException as exc:
                        # Tracebacks can retain an open capture after this call.
                        tb = exc.__traceback__
                        while tb:
                            capture = tb.tb_frame.f_locals.get('cap')
                            if capture is not None:
                                capture.release()
                            tb = tb.tb_next
                        raise
                    finally:
                        streams.stack.pop()
            return RetainedStreamBackend

        registry[name] = make_backend(original)
    video._diffcap_stream_retention = True


def load_inputs(source, media_root=None, prompt=None):
    """Validate all inputs before loading the model; never modify source records."""
    source = Path(source).expanduser().resolve(strict=True)
    root = Path(media_root).expanduser().resolve(strict=True) if media_root else source.parent
    raw = read_jsonl(source) if source.suffix.lower() == '.jsonl' else [
        {'id': source.stem, 'media': [str(source)]}]
    if not raw:
        raise ValueError('Input contains no samples')
    rows, seen = [], set()
    for row in raw:
        rid = row.get('id')
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError('Every sample needs a unique, nonempty string id')
        seen.add(rid)
        media = row.get('media')
        if not isinstance(media, list) or not media or not all(isinstance(p, str) and p for p in media):
            raise ValueError(f'{rid}: media must be a nonempty list of local paths')
        paths, kinds = [], set()
        for name in media:
            path = (root / Path(name).expanduser()).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f'{rid}: expected a media file')
            if path.suffix.lower() in IMAGE_SUFFIXES:
                kinds.add('image')
            elif path.suffix.lower() in VIDEO_SUFFIXES:
                kinds.add('video')
            else:
                raise ValueError(f'{rid}: unsupported media extension {path.suffix}')
            paths.append(path)
        if len(kinds) != 1 or ('video' in kinds and len(paths) != 1):
            raise ValueError(f'{rid}: use one video or one or more images per sample')
        kind = kinds.pop()
        instruction = prompt if prompt is not None else row.get('prompt', PROMPTS[kind])
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f'{rid}: prompt must contain non-whitespace text')
        rows.append(dict(id=rid, media=[str(p) for p in paths], media_type=kind,
                         prompt=instruction, media_sha256=[sha256(p) for p in paths]))
    return rows


def caption_messages(row, system_prompt=None):
    kind = row['media_type'] + '_url'
    visuals = [{'type': kind, kind: {'url': Path(p).as_uri()}} for p in row['media']]
    text = {'type': 'text', 'text': row['prompt']}
    content = [text, *visuals] if row['media_type'] == 'image' else [*visuals, text]
    messages = [{'role': 'user', 'content': content}]
    if system_prompt:
        messages.insert(0, {'role': 'system', 'content': system_prompt})
    return messages


def apply_preset(args, rows, config, parser):
    """Resolve benchmark defaults while allowing explicit CLI overrides."""
    kinds = {row['media_type'] for row in rows}
    if args.preset != 'default':
        expected = 'image' if args.preset == 'captionqa' else 'video'
        if kinds != {expected}:
            parser.error(f'--preset {args.preset} requires {expected} inputs only')
    context = config.get('text_config', config).get('max_position_embeddings', 262144)
    defaults = {
        'default': dict(temperature=0.7, seed=0 if kinds == {'video'} else 1,
                        max_model_len=32768, system_prompt=None),
        'captionqa': dict(temperature=0.7, seed=1, max_model_len=context, system_prompt=None),
        'vidcapbench': dict(temperature=0.0, seed=0, max_model_len=32768, system_prompt=None),
        'vdc': dict(temperature=0.7, seed=0, max_model_len=128000,
                    system_prompt='You are a helpful assistant.'),
    }[args.preset]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Downloaded 4B checkpoint directory')
    parser.add_argument('--input', type=Path, required=True, help='Local image, video, or JSONL manifest')
    parser.add_argument('--media-root', type=Path, help='Base for manifest paths; default: manifest directory')
    parser.add_argument('--output', type=Path, required=True, help='Directory for captions.jsonl and run metadata')
    parser.add_argument('--prompt', help='Override the default or per-sample caption instruction')
    parser.add_argument('--preset', choices=('default', 'captionqa', 'vidcapbench', 'vdc'),
                        default='default', help='Benchmark prompt and sampling defaults; explicit flags override them')
    parser.add_argument('--system-prompt', help='Optional system instruction; empty string disables it')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel GPU count')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--video-frames', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.6)
    parser.add_argument('--limit', type=int, help='Process only the first N samples')
    parser.add_argument('--enforce-eager', action='store_true')
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve(strict=True)
    rows = load_inputs(args.input, args.media_root, args.prompt)
    native_configs = {
        name: json.loads((args.model / name).read_text())
        for name in ('config.json', 'preprocessor_config.json',
                     'video_preprocessor_config.json', 'generation_config.json')
        if (args.model / name).exists()
    }
    apply_preset(args, rows, native_configs.get('config.json', {}), parser)
    for name in ('tp', 'batch_size', 'max_model_len', 'video_frames', 'max_tokens', 'limit'):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.temperature < 0 or not 0 < args.gpu_memory_utilization < 1:
        parser.error('Temperature must be nonnegative and GPU memory utilization must be between 0 and 1')
    if args.max_tokens >= args.max_model_len:
        parser.error('--max-tokens must be smaller than --max-model-len')
    if args.prompt is not None and not args.prompt.strip():
        parser.error('--prompt must contain non-whitespace text')
    if args.limit:
        rows = rows[:args.limit]
    opencv_version = check_video_backend() if any(r['media_type'] == 'video' for r in rows) else None
    data_hash = digest(rows)
    model_manifest = args.model / 'checksums.json'
    weights = json.loads(model_manifest.read_text()) if model_manifest.exists() else {
        p.name: sha256(p) for p in sorted(args.model.iterdir()) if p.is_file()}
    mm_limits = {'image': max((len(r['media']) for r in rows if r['media_type'] == 'image'), default=0),
                 'video': int(any(r['media_type'] == 'video' for r in rows))}
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'vllm')}
    protocol = dict(model=str(args.model), weights=weights, versions=versions,
                    native_checkpoint_configs=native_configs, data_sha256=data_hash,
                    tensor_parallel_size=args.tp, batch_size=args.batch_size,
                    max_model_len=args.max_model_len, max_tokens=args.max_tokens,
                    video_num_frames=args.video_frames, temperature=args.temperature,
                    video_loader_backend=os.environ.get('VLLM_VIDEO_LOADER_BACKEND', 'opencv'),
                    opencv_version=opencv_version,
                    top_p=1.0, seed=args.seed, seed_mode='engine',
                    preset=args.preset, system_prompt=args.system_prompt,
                    top_k=-1, repetition_penalty=1.0, frequency_penalty=0.0,
                    max_num_seqs=32,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    generation_config='vllm', enforce_eager=args.enforce_eager,
                    limit_mm_per_prompt=mm_limits, source_sha256=sha256(Path(__file__)))
    run_id = digest(protocol)
    manifest = args.output / 'run.json'
    path = args.output / 'captions.jsonl'
    if manifest.exists():
        if json.loads(manifest.read_text())['run_id'] != run_id:
            raise ValueError('Output directory belongs to another protocol or input; choose a fresh directory')
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory or a previous run from this script')
    existing = read_resumable_output(path) if path.exists() else []
    by_id = {r['id']: r for r in rows}
    done = set()
    for row in existing:
        rid = row.get('id')
        if rid not in by_id or rid in done or row.get('run_id') != run_id or row.get('data_sha256') != data_hash:
            raise ValueError('Existing captions do not match this run')
        if any(row.get(key) != value for key, value in by_id[rid].items()) or not row.get('caption', '').strip():
            raise ValueError('Existing caption inputs or content have changed')
        done.add(rid)
    write_json(manifest, dict(run_id=run_id, protocol=protocol))
    pending = [r for r in rows if r['id'] not in done]
    errors = []
    if pending:
        if any(r['media_type'] == 'video' for r in pending):
            install_video_decoder_fix()
        from vllm import LLM, SamplingParams
        media_root = os.path.commonpath([str(Path(p).parent) for row in rows for p in row['media']])
        llm = LLM(model=str(args.model), dtype='bfloat16', tensor_parallel_size=args.tp,
                  gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
                  allowed_local_media_path=media_root, limit_mm_per_prompt=mm_limits,
                  media_io_kwargs={'video': {'num_frames': args.video_frames}}, seed=args.seed,
                  generation_config='vllm', enable_chunked_prefill=True,
                  max_num_seqs=32, enforce_eager=args.enforce_eager)
        sampling = SamplingParams(temperature=args.temperature, top_p=1.0, top_k=-1,
                                  repetition_penalty=1.0, frequency_penalty=0.0,
                                  max_tokens=args.max_tokens)

        def generate(batch, out):
            try:
                outputs = llm.chat(messages=[caption_messages(r, args.system_prompt) for r in batch],
                                   sampling_params=sampling, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise ValueError('Model returned an unexpected output count')
                records = []
                for row, result in zip(batch, outputs):
                    pred = result.outputs[0]
                    if not pred.text.strip():
                        raise ValueError('Model returned an empty caption')
                    records.append(dict(row, caption=pred.text.strip(), run_id=run_id,
                                        data_sha256=data_hash, prompt_tokens=len(result.prompt_token_ids),
                                        output_tokens=len(pred.token_ids), finish_reason=pred.finish_reason,
                                        stop_reason=getattr(pred, 'stop_reason', None)))
            except Exception as exc:
                if len(batch) > 1:
                    for row in batch:
                        generate([row], out)
                else:
                    errors.append(dict(id=batch[0]['id'], error_type=type(exc).__name__))
                return
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                done.add(record['id'])
            out.flush()

        with path.open('a', encoding='utf-8') as out:
            for start in range(0, len(pending), args.batch_size):
                generate(pending[start:start + args.batch_size], out)
                print(f'{len(done)}/{len(rows)} captions complete, {len(errors)} errors', flush=True)
    truncated = sum(r.get('finish_reason') == 'length' for r in read_jsonl(path)) if path.exists() else 0
    write_json(args.output / 'status.json', dict(expected=len(rows), completed=len(done), errors=errors,
                                               truncated=truncated,
                                               complete=len(done) == len(rows), timestamp=time.time()))
    if errors or len(done) != len(rows):
        raise RuntimeError(f'Incomplete captions ({len(done)}/{len(rows)}); rerun to retry')
    if truncated:
        print(f'{truncated} captions reached the output token limit; inspect finish_reason before evaluation.', flush=True)
    print(f'Captions saved to {path}', flush=True)


if __name__ == '__main__':
    main()
