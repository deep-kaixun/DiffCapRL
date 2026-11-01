"""Configuration, immutable stage manifests, resumable records and media inputs."""
import argparse
import base64
import fcntl
import hashlib
import json
import mimetypes
import math
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv

from . import PROTOCOL
from .client import Client, resolve_endpoint, resolve_model
from .schema import digest

PACKAGE = Path(__file__).resolve().parent
STAGES = {1: 'media', 2: 'generated', 3: 'validated', 4: 'blind_filtered',
          5: 'verified', 6: 'calibrated', 7: 'sampled', 8: 'exported'}
ROLES = {2: ['offline_qa_maker'], 4: ['blind_solver', 'dependency_checker'],
         5: ['weak_verifier', 'teacher_1', 'teacher_2'], 6: ['captioner', 'offline_qa_solver']}


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        start = f.read(1)
        f.seek(0)
        if start == '[':
            yield from json.load(f)
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def load_config(path):
    load_dotenv(os.getenv('QA_ENV_FILE', str(PACKAGE.parent / '.env')), override=False)
    config = json.loads(Path(path).read_text())
    if config.get('protocol') != PROTOCOL:
        raise ValueError('unsupported_protocol')
    for key in ('workers', 'request_attempts', 'timeout_seconds', 'captions_per_media',
                'blind_permutation_checks', 'min_questions', 'seed'):
        if type(config.get(key)) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError('invalid_config_' + key)
    if config['captions_per_media'] < 2:
        raise ValueError('calibration_requires_multiple_captions')
    if config['blind_permutation_checks'] > 3:
        raise ValueError('only_three_nonidentity_rotations_for_four_choices')
    if config.get('verification_strategy') not in ('cascade', 'full_teacher'):
        raise ValueError('invalid_verification_strategy')
    if config.get('retry_delay_seconds', -1) < 0:
        raise ValueError('invalid_retry_delay')
    weights = config['difficulty_weights']
    if set(weights) != {'easy', 'medium', 'hard'} or any(type(v) not in (int, float) or v <= 0 for v in weights.values()):
        raise ValueError('invalid_difficulty_weights')
    if abs(sum(weights.values()) - 1) > 1e-8:
        raise ValueError('difficulty_weights_must_sum_to_one')
    for media_type in ('image', 'video'):
        if type(config['target_questions'][media_type]) is not int or config['target_questions'][media_type] < config['min_questions']:
            raise ValueError('target_below_minimum')
    for role in {r for roles in ROLES.values() for r in roles}:
        client = config['clients'][role]
        if not (client.get('model') or client.get('model_env')) or not (client.get('base_url') or client.get('base_url_env')):
            raise ValueError('incomplete_client_' + role)
        if 'api_key' in client:
            raise ValueError('use_api_key_env_instead_of_literal_key')
        if type(client['max_tokens']) is not int or client['max_tokens'] < 1 or client['temperature'] < 0:
            raise ValueError('invalid_sampling_' + role)
    if type(config.get('expand_sources')) is not bool:
        raise ValueError('expand_sources_must_be_boolean')
    if type(config.get('pool_audit', True)) is not bool:
        raise ValueError('pool_audit_must_be_boolean')
    if config.get('video_input', 'url') not in ('url', 'inline', 'frames'):
        raise ValueError('video_input_must_be_url_inline_or_frames')
    reserve = config.get('candidate_reserve', 2)
    if type(reserve) is not int or not 0 <= reserve <= 4:
        raise ValueError('candidate_reserve_must_be_zero_to_four')
    frame_fps = config.get('video_fps', 2)
    if type(frame_fps) not in (int, float) or not math.isfinite(frame_fps) or frame_fps <= 0:
        raise ValueError('invalid_video_fps')
    if type(config.get('max_video_frames', 48)) is not int or config.get('max_video_frames', 48) < 2:
        raise ValueError('invalid_max_video_frames')
    for client in config['clients'].values():
        if client.get('video_input', config.get('video_input', 'url')) not in ('url', 'inline', 'frames'):
            raise ValueError('invalid_client_video_input')
    return config


def preflight_services(config):
    # Resolve configuration only: do not send requests, probe credentials, or decode videos.
    roles = sorted({r for roles in ROLES.values() for r in roles})
    services = {r: {'endpoint': resolve_endpoint(config, r), 'model': resolve_model(config, r)} for r in roles}
    if len({services[r]['model'] for r in ROLES[5]}) != 3:
        raise ValueError('configure_three_distinct_verification_models')
    return services


def artifact(run_dir, stage):
    return run_dir / f'{stage}_{STAGES[stage]}.jsonl'


def latest_records(path, repair=False):
    result = {}
    if not path.exists():
        return result
    offset = 0
    with path.open('rb') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or not {'id', 'status'} <= item.keys():
                raise ValueError('invalid_stage_record')
        except (ValueError, UnicodeDecodeError):
            if not repair or i != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError('corrupt_stage_records') from None
            path.with_suffix('.incomplete_tail.partial').write_bytes(line)
            with path.open('r+b') as f:
                f.truncate(offset)
            break
        result[item['id']] = item
        offset += len(line)
        if repair and i == len(lines) - 1 and not line.endswith(b'\n'):
            with path.open('ab') as f:
                f.write(b'\n')
    return result


class Rejected(Exception):
    def __init__(self, reason, data):
        self.reason, self.data = reason, data


class NeedsReview(Rejected):
    pass


def text_message(text):
    return [{'role': 'user', 'content': text}]


class Context:
    def __init__(self, config, run_dir, client=None):
        self.config, self.run_dir, self.client = config, run_dir, client
        private = run_dir / '.private' / 'media_urls.json'
        self.urls = json.loads(private.read_text()) if private.exists() else {}
        self.frame_cache = {}

    def frame_content(self, row):
        if not row.get('media_path'):
            raise ValueError('frame_input_requires_local_video')
        path = Path(row['media_path'])
        fps = self.config.get('video_fps', 2)
        cap = self.config.get('max_video_frames', 48)
        key = digest([file_hash(path), fps, cap])
        if key in self.frame_cache:
            return self.frame_cache[key]
        info = json.loads(subprocess.run(
            ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
            capture_output=True, text=True, check=True).stdout)
        stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
        duration = float(info['format']['duration'])
        numerator, denominator = stream.get('avg_frame_rate', '24/1').split('/')
        native_fps = float(numerator) / float(denominator)
        if duration <= 0 or native_fps <= 0:
            raise ValueError('invalid_video_metadata')
        count = min(cap, max(2, math.ceil(duration * fps)))
        last = max(0, duration - 1 / native_fps)
        times = [round(i * last / (count - 1), 6) for i in range(count)]
        directory = self.run_dir / '.private' / 'video_frames' / key
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / 'manifest.json'
        cached = json.loads(manifest.read_text()) if manifest.exists() else None
        if cached and (cached.get('source_sha256') != file_hash(path)
                       or len(cached.get('frames', [])) != count
                       or [f.get('time_seconds') for f in cached['frames']] != times):
            raise ValueError('cached_video_manifest_changed')
        frames, content = [], [{'type': 'text', 'text': (
            f'Video supplied as {count} ordered visual frames spanning approximately 0 to {duration:.3f} seconds. '
            'Audio is excluded. Unsampled details may be unavailable; do not invent events between frames.')}]
        for index, timestamp in enumerate(times):
            frame = directory / f'{index:03d}.jpg'
            if cached:
                if file_hash(frame) != cached['frames'][index]['sha256']:
                    raise ValueError('cached_video_frame_changed')
            else:
                subprocess.run(['ffmpeg', '-y', '-nostdin', '-v', 'error', '-ss', str(timestamp), '-i', str(path),
                                '-frames:v', '1', '-q:v', '2', str(frame)], capture_output=True, check=True)
            if not frame.is_file() or frame.stat().st_size == 0:
                raise ValueError('video_frame_extraction_failed')
            frames.append({'time_seconds': timestamp, 'sha256': file_hash(frame), 'file': frame.name})
            content.extend([{'type': 'text', 'text': f'Frame near {timestamp:.3f} seconds:'},
                            {'type': 'image_url', 'image_url': {
                                'url': 'data:image/jpeg;base64,' + base64.b64encode(frame.read_bytes()).decode()}}])
        if not cached:
            write_json(manifest, {'source_sha256': file_hash(path), 'requested_fps': fps,
                                  'max_frames': cap, 'duration_seconds': duration, 'frames': frames})
        self.frame_cache[key] = content
        return content

    def media_messages(self, row, prompt, role=None):
        url = self.urls.get(row['id'])
        client = self.config['clients'].get(role, {})
        video_input = client.get('video_input', self.config.get('video_input', 'url'))
        if row['media_type'] == 'video' and video_input == 'frames':
            return [{'role': 'user', 'content': self.frame_content(row) + [{'type': 'text', 'text': prompt}]}]
        if row['media_type'] == 'video' and video_input == 'inline':
            if not row.get('media_path'):
                raise ValueError('inline_video_requires_local_media')
            path = Path(row['media_path'])
            if path.suffix.lower() != '.mp4':
                raise ValueError('inline_video_requires_mp4')
            encoded = base64.b64encode(path.read_bytes()).decode()
            content = {'type': 'video_url', 'video_url': {'url': f'data:video/mp4;base64,{encoded}'}}
        elif url:
            content = {'type': row['media_type'] + '_url', row['media_type'] + '_url': {'url': url}}
        elif row['media_type'] == 'image' and row.get('media_path'):
            path = Path(row['media_path'])
            mime = mimetypes.guess_type(path)[0]
            if mime not in ('image/png', 'image/jpeg', 'image/webp', 'image/gif'):
                raise ValueError('unsupported_image_type')
            encoded = base64.b64encode(path.read_bytes()).decode()
            content = {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}}
        else:
            raise ValueError('video_requires_http_url_mapping')
        return [{'role': 'user', 'content': [content, {'type': 'text', 'text': prompt}]}]


def prepare_media(input_path, url_map_path, limit=None, video_input='url'):
    urls = {}
    if url_map_path:
        for row in read_rows(url_map_path):
            key = str(row.get('id') or row.get('bench_id') or '')
            value = row.get('url') or row.get('media_url') or row.get('video_url')
            if not key or not isinstance(value, str):
                raise ValueError('invalid_url_map_row')
            if key in urls and urls[key] != value:
                raise ValueError('conflicting_url_map')
            urls[key] = value
    groups, private, source_ids, hashes = {}, {}, set(), {}
    for index, r in enumerate(read_rows(input_path)):
        if limit is not None and index >= limit:
            break
        source_id = str(r.get('id', ''))
        if not source_id or source_id in source_ids:
            raise ValueError('missing_or_duplicate_input_id')
        source_ids.add(source_id)
        media_type = r.get('media_type')
        if media_type not in ('image', 'video'):
            raise ValueError('media_type_must_be_image_or_video')
        path, sha = None, None
        if r.get('media_path'):
            path = Path(r['media_path'])
            if not path.is_absolute():
                path = input_path.parent / path
            path = path.resolve()
            if not path.is_file():
                raise ValueError('missing_media_for_' + source_id)
            if path not in hashes:
                hashes[path] = file_hash(path)
            sha = hashes[path]
            media_id = digest([media_type, sha])[:24]
        else:
            if not r.get('media_id'):
                raise ValueError('url_only_inputs_require_explicit_media_id')
            media_id = digest([media_type, r['media_id']])[:24]
        url = urls.get(source_id) or r.get('url')
        if url:
            parsed = urlsplit(url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('invalid_media_url_for_' + source_id)
            if media_id in private and private[media_id] != url:
                raise ValueError('conflicting_urls_for_duplicate_media')
            private[media_id] = url
        if not path and not url:
            raise ValueError('missing_media_input')
        if media_type == 'video' and video_input == 'url' and not url:
            raise ValueError('video_requires_http_url_for_' + source_id)
        if media_type == 'video' and video_input in ('inline', 'frames') and (not path or path.suffix.lower() != '.mp4'):
            raise ValueError('inline_or_frame_video_requires_local_mp4')
        group = groups.setdefault(media_id, {'id': media_id, 'media_type': media_type,
                                             'media_path': str(path) if path else None,
                                             'media_sha256': sha, 'sources': []})
        group['sources'].append({'id': source_id, 'source': r.get('source', '')})
    if not groups:
        raise ValueError('empty_media_manifest')
    return sorted(groups.values(), key=lambda r: r['id']), private


def run_stage(stage, argv=None):
    from .stages import PROCESSORS, write_export, write_review_queue
    parser = argparse.ArgumentParser(description=f'Offline QA step {stage}: {STAGES[stage]}')
    parser.add_argument('--config', type=Path, default=PACKAGE / 'config.example.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    if stage == 1:
        parser.add_argument('--input', type=Path, required=True)
        parser.add_argument('--url-map', type=Path)
        parser.add_argument('--limit', type=int, help='Use the first N source rows for a cost/quality pilot')
    if stage == 2:
        parser.add_argument('--supplement-from', type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = Client(config, run_dir) if stage in ROLES else None
        try:
            endpoints = {role: client.endpoint(role) for role in ROLES.get(stage, [])}
            models = {role: resolve_model(config, role) for role in ROLES.get(stage, [])}
            if stage == 5:
                if len(set(models.values())) != 3:
                    raise ValueError('configure_three_distinct_verification_models')
            input_path = args.input.resolve() if stage == 1 else artifact(run_dir, stage - 1)
            if stage == 1:
                if args.limit is not None and args.limit < 1:
                    raise ValueError('limit_must_be_positive')
                rows, private = prepare_media(input_path, args.url_map, args.limit, config.get('video_input', 'url'))
            else:
                prev = json.loads((run_dir / f'{stage-1}_manifest.json').read_text())
                if not prev['complete']:
                    raise ValueError('previous_stage_incomplete')
                if prev.get('output_sha256') != file_hash(input_path):
                    raise ValueError('previous_stage_artifact_changed')
                records = latest_records(input_path)
                rows = [r['data'] for r in records.values() if r['status'] == 'ok']
                first = json.loads((run_dir / '1_manifest.json').read_text())
                if not first['complete'] or first['output_sha256'] != file_hash(artifact(run_dir, 1)):
                    raise ValueError('initial_media_artifact_changed')
                initial_media = latest_records(artifact(run_dir, 1))
                for record in initial_media.values():
                    media = record['data']
                    if media.get('media_path') and file_hash(media['media_path']) != media['media_sha256']:
                        raise ValueError('source_media_changed_use_new_run')
                private_path = run_dir / '.private' / 'media_urls.json'
                if digest(json.loads(private_path.read_text())) != first['signature']['url_mapping_sha256']:
                    raise ValueError('media_url_mapping_changed_use_new_run')
            code_hash = digest({str(p.relative_to(PACKAGE)): file_hash(p)
                                for p in PACKAGE.rglob('*') if p.suffix in ('.py', '.txt')})
            if stage > 1 and (prev['signature']['config'] != config or prev['signature']['code_sha256'] != code_hash):
                raise ValueError('previous_stage_config_or_code_changed_use_new_run')
            signature = {'protocol': PROTOCOL, 'stage': stage, 'config': config,
                         'input_sha256': file_hash(input_path), 'code_sha256': code_hash,
                         'endpoints': endpoints, 'resolved_models': models}
            supplements = None
            if stage == 2 and args.supplement_from:
                supplements = latest_records(args.supplement_from)
                if set(supplements) - {r['id'] for r in rows}:
                    raise ValueError('supplement_source_has_unknown_media_ids')
                if any(r['status'] not in ('ok', 'rejected') for r in supplements.values()):
                    raise ValueError('resolve_prior_errors_before_supplementing')
                signature['supplement_sha256'] = file_hash(args.supplement_from)
            private_path = run_dir / '.private' / 'media_urls.json'
            if stage == 1:
                signature['url_mapping_sha256'] = digest(private)
                signature['resolved_media_sha256'] = digest(rows)
                signature['input_limit'] = args.limit
            elif stage in (2, 5, 6):
                signature['url_mapping_sha256'] = file_hash(private_path)
            meta_path = run_dir / f'{stage}_manifest.json'
            output = artifact(run_dir, stage)
            if meta_path.exists():
                if not args.resume:
                    raise ValueError('stage_exists_use_resume_or_new_run')
                old = json.loads(meta_path.read_text())
                if old['signature'] != signature:
                    raise ValueError('stage_inputs_config_or_code_changed_use_new_run')
                if old.get('complete') and old.get('output_sha256') != file_hash(output):
                    raise ValueError('completed_stage_artifact_changed')
            elif output.exists():
                raise ValueError('stage_output_without_manifest')
            if stage == 1:
                private_path.parent.mkdir(exist_ok=True, mode=0o700)
                write_json(private_path, private)
                private_path.chmod(0o600)
            meta = {'signature': signature, 'complete': False, 'input_media': len(rows)}
            write_json(meta_path, meta)
            previous = latest_records(output, repair=args.resume)
            expected = {r['id'] for r in rows}
            if set(previous) - expected:
                raise ValueError('unexpected_ids_in_stage_output')
            todo = [r for r in rows if previous.get(r['id'], {}).get('status') not in ('ok', 'rejected', 'needs_review')]
            context = Context(config, run_dir, client)
            context.supplements = supplements

            def process(row):
                try:
                    data = PROCESSORS[stage](context, row)
                    return {'id': row['id'], 'status': 'ok', 'data': data}
                except NeedsReview as exc:
                    return {'id': row['id'], 'status': 'needs_review', 'reason': exc.reason, 'data': exc.data}
                except Rejected as exc:
                    return {'id': row['id'], 'status': 'rejected', 'reason': exc.reason, 'data': exc.data}
                except Exception as exc:
                    # Exceptions may contain private URLs or API credentials; do not serialize their text.
                    return {'id': row['id'], 'status': 'error', 'error_type': type(exc).__name__}

            with output.open('a', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=config['workers']) as pool:
                for future in as_completed([pool.submit(process, r) for r in todo]):
                    record = future.result()
                    record['finished_at_unix'] = time.time()
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    previous[record['id']] = record
            counts = Counter(r['status'] for r in previous.values())
            meta.update(counts=dict(counts), complete=set(previous) == expected and not counts['error'],
                        output_sha256=file_hash(output))
            if stage == 5:
                meta['pending_review_questions'] = write_review_queue(run_dir, list(previous.values()))
            if stage == 8 and meta['complete']:
                write_export(run_dir, [r['data'] for r in previous.values() if r['status'] == 'ok'])
            write_json(meta_path, meta)
            print(json.dumps({'stage': stage, 'counts': dict(counts), 'complete': meta['complete'],
                              **({'pending_review_questions': meta['pending_review_questions']} if stage == 5 else {})}))
            if not meta['complete']:
                raise SystemExit(1)
        finally:
            if client:
                client.close()
