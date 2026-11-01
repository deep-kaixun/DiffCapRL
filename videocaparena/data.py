"""Load VideoCapArena caption pairs and human preferences."""
import json
from pathlib import Path, PurePosixPath


def read(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f'Invalid JSON on line {line_number}') from None
        if not isinstance(row, dict):
            raise ValueError(f'Expected an object on line {line_number}')
        rows.append(row)
    return rows


def keyed(rows):
    result = {}
    for row in rows:
        value = row.get('id')
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Expected a nonempty string id')
        if value in result:
            raise ValueError(f'Duplicate id: {value}')
        result[value] = row
    return result


def validate_dataset(root, require_media=False):
    root = Path(root)
    rows = read(root / 'data/test.jsonl')
    if not rows:
        raise ValueError('Dataset contains no caption pairs')
    keyed(rows)
    for row in rows:
        rid = row['id']
        if type(row.get('best_idx')) is not int or row['best_idx'] not in (0, 1):
            raise ValueError(f'{rid}: expected human best_idx 0 or 1')
        captions = row.get('captions')
        if not isinstance(captions, list) or len(captions) != 2 or not all(
                isinstance(c, str) and c.strip() for c in captions):
            raise ValueError(f'{rid}: expected two nonempty captions')
        category = row.get('category')
        if category is not None and (not isinstance(category, str) or not category.strip()):
            raise ValueError(f'{rid}: category must be a nonempty string')
        if require_media:
            name = row.get('media_path')
            if not isinstance(name, str) or not name:
                raise ValueError(f'{rid}: missing media_path')
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError(f'{rid}: expected a relative media path')
            media = root / Path(*path.parts)
            media.resolve().relative_to(root.resolve())
            if not media.is_file() or not media.stat().st_size:
                raise ValueError(f'{rid}: missing or empty video')
    return rows
