"""Score caption choices against VideoCapArena human preferences."""
import argparse
import json
from pathlib import Path

if __package__:
    from .data import keyed, read, validate_dataset
else:
    from data import keyed, read, validate_dataset

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'datasets/videocaparena'


def score(rows, predictions):
    """Missing choices and abstentions count as incorrect on the full dataset."""
    if not rows:
        raise ValueError('Dataset contains no caption pairs')
    reference = keyed(rows)
    predicted = keyed(predictions)
    unknown = set(predicted) - set(reference)
    if unknown:
        raise ValueError('Prediction IDs are absent from the dataset: ' + ', '.join(sorted(unknown)))
    for rid, row in predicted.items():
        if 'best_idx' not in row:
            raise ValueError(f'{rid}: prediction needs best_idx (0, 1, or null)')
        choice = row['best_idx']
        if choice is not None and (type(choice) is not int or choice not in (0, 1)):
            raise ValueError(f'{rid}: best_idx must be 0, 1, or null')

    def metrics(group):
        correct = sum(predicted.get(row['id'], {}).get('best_idx') == row['best_idx'] for row in group)
        submitted = sum(row['id'] in predicted for row in group)
        abstained = sum(row['id'] in predicted and predicted[row['id']]['best_idx'] is None for row in group)
        return dict(total=len(group), correct=correct, accuracy=correct / len(group),
                    submitted=submitted, missing=len(group) - submitted, abstained=abstained)

    result = metrics(rows)
    groups = {}
    for row in rows:
        if row.get('category'):
            groups.setdefault(row['category'], []).append(row)
    result['by_category'] = {category: metrics(group) for category, group in sorted(groups.items())}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA,
                        help='Dataset directory containing data/test.jsonl')
    parser.add_argument('--predictions', type=Path, required=True,
                        help='JSONL records with id and predicted best_idx (0/1/null)')
    parser.add_argument('--output', type=Path, help='Optional summary JSON path')
    parser.add_argument('--check-media', action='store_true', help='Also verify local video files')
    args = parser.parse_args(argv)
    try:
        rows = validate_dataset(args.data_root, require_media=args.check_media)
        result = score(rows, read(args.predictions))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
