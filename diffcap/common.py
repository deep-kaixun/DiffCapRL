import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_resumable_output(path):
    """Recover an interrupted final write; never discard malformed complete lines."""
    path = Path(path)
    rows, offset, tail = [], 0, None
    with path.open('rb') as f:
        for raw in f:
            try:
                if raw.strip():
                    rows.append(json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if raw.endswith(b'\n') or f.read(1):
                    raise
                tail = raw
                break
            offset += len(raw)
    if tail is not None:
        path.with_suffix(path.suffix+'.interrupted_tail').write_bytes(tail)
        with path.open('r+b') as f:
            f.truncate(offset)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)
