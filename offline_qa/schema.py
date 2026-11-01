"""Strict MCQ parsing, identity, deterministic option shuffling and sampling."""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict

LETTERS = 'ABCD'
VIDEO_CATEGORIES = ('Bkg', 'Evt', 'Ins', 'IAct', 'IApp', 'IPos', 'Mov', 'Per', 'Reg')
IMAGE_CATEGORIES = ('Glo', 'Bkg', 'Ins', 'IAtt', 'IAct', 'Spa', 'Txt', 'Reg')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def parse_json(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.I)
    return json.loads(text)


def generated_items(value):
    """Accept the paper's numbered object and the historical/new list schemas."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get('questions'), list):
        return value['questions']
    if isinstance(value, dict) and 'question' in value:
        return [value]
    if isinstance(value, dict):
        numbers = sorted(int(k.split('_')[1]) for k in value if re.fullmatch(r'Question_\d+', k))
        if numbers:
            return [{'question': value[f'Question_{i}'], 'ref_answer': value.get(f'Answer_{i}')}
                    for i in numbers]
    raise ValueError('unsupported_generation_schema')


def norm(text):
    return ' '.join(text.split()).casefold()


def split_options(text):
    """Do not guess if embedded option labels are ambiguous or out of order."""
    marker = re.search(r'(?:The\s+)?options\s+are\s+as\s+follows\s*:', text, re.I)
    body = text[marker.end():] if marker else text
    matches = list(re.finditer(r'(?<!\w)([A-D])[.、:：)）]\s*', body))
    if [m[1] for m in matches] != list(LETTERS):
        raise ValueError('ambiguous_or_missing_option_labels')
    stem = text[:marker.start()] if marker else body[:matches[0].start()]
    choices = [body[m.end():matches[i+1].start() if i+1 < 4 else len(body)].strip().rstrip(',;').strip()
               for i, m in enumerate(matches)]
    return stem.strip(), choices


def normalize_item(item, media_id):
    if not isinstance(item, dict) or not isinstance(item.get('question'), str):
        raise ValueError('missing_question')
    question = item['question'].strip()
    choices = item.get('choices', item.get('options'))
    if choices is None:
        question, choices = split_options(question)
    elif isinstance(choices, dict):
        if set(choices) != set(LETTERS):
            raise ValueError('invalid_option_keys')
        choices = [choices[k] for k in LETTERS]
    if not question or not isinstance(choices, list) or len(choices) != 4:
        raise ValueError('expected_four_choices')
    if any(not isinstance(c, str) or not c.strip() for c in choices):
        raise ValueError('empty_choice')
    choices = [c.strip() for c in choices]
    if len({norm(c) for c in choices}) != 4:
        raise ValueError('duplicate_choices')
    # Positional references can change meaning when options are shuffled.
    positional = r'\b(?:all|none|both) of (?:the )?(?:above|below)\b|\b(?:option|choice)s?\s+[A-D]\b|\bboth [A-D] and [A-D]\b|以上(?:都|皆)|上述(?:所有|全部)'
    if any(re.search(positional, c, re.I) for c in choices):
        raise ValueError('position_dependent_choice')
    answer = item.get('answer_index', item.get('ref_answer', item.get('answer')))
    if isinstance(answer, str) and answer.strip().upper() in LETTERS and len(answer.strip()) == 1:
        answer = LETTERS.index(answer.strip().upper())
    if type(answer) is not int or answer not in range(4):
        raise ValueError('invalid_answer')
    if item.get('modality', 'Visual') != 'Visual':
        raise ValueError('nonvisual_modality')
    identity = [media_id, norm(question), sorted(map(norm, choices))]
    return {'qa_id': digest(identity)[:24], 'question': question, 'choices': choices,
            'answer_index': answer, 'category': item.get('category', 'unclassified'),
            'focus': item.get('focus', ''), 'modality': 'Visual'}


def shuffled(qa, seed):
    correct = qa['choices'][qa['answer_index']]
    choices = sorted(qa['choices'], key=norm)
    permutation = list(range(4))
    random.Random(digest([seed, qa['qa_id']])).shuffle(permutation)
    output = [choices[i] for i in permutation]
    return {**qa, 'choices': output, 'answer_index': output.index(correct)}


def rotate(qa):
    return {**qa, 'choices': qa['choices'][1:] + qa['choices'][:1],
            'answer_index': (qa['answer_index'] - 1) % 4}


def normalize_pool(items, media_id, seed):
    groups = defaultdict(list)
    rejected = []
    for i, item in enumerate(items):
        try:
            q = normalize_item(item, media_id)
            groups[q['qa_id']].append((i, q))
        except ValueError as exc:
            rejected.append({'candidate_index': i, 'reason': str(exc)})
    kept = []
    for qid, group in groups.items():
        answers = {norm(q['choices'][q['answer_index']]) for _, q in group}
        if len(answers) > 1:
            rejected.extend({'candidate_index': i, 'qa_id': qid, 'reason': 'conflicting_reference_answers'}
                            for i, _ in group)
            continue
        kept.append(shuffled(group[0][1], seed))
        rejected.extend({'candidate_index': i, 'qa_id': qid, 'reason': 'duplicate_question'}
                        for i, _ in group[1:])
    return kept, rejected


def question_text(q):
    return q['question'] + ' The options are as follows: ' + ', '.join(
        f'{letter}. {choice}' for letter, choice in zip(LETTERS, q['choices']))


def public_question(q):
    # Never pass answer_index, verification results or difficulty to a solver.
    return {'id': q['qa_id'], 'question': question_text(q)}


def answer_index(value):
    if value is None or (isinstance(value, str) and value.strip().casefold() in ('null', 'none', 'unknown')):
        return None
    if isinstance(value, str) and value in LETTERS and len(value) == 1:
        return LETTERS.index(value)
    raise ValueError('invalid_solver_answer')


def answer_batch(value, qas):
    if not isinstance(value, dict) or not isinstance(value.get('answers'), list):
        raise ValueError('missing_answers_list')
    result = {}
    for item in value['answers']:
        if not isinstance(item, dict) or 'id' not in item or 'answer' not in item:
            raise ValueError('invalid_answer_record')
        qid = str(item['id'])
        if qid in result:
            raise ValueError('duplicate_answer_id')
        result[qid] = answer_index(item['answer'])
    if set(result) != {q['qa_id'] for q in qas}:
        raise ValueError('incomplete_or_extra_answer_ids')
    return result


def sample_pool(qas, target, weights, seed):
    """Capacity-aware weighted difficulty quotas; then balance categories in each bucket."""
    buckets = defaultdict(list)
    for q in qas:
        buckets[q['calibration']['bucket']].append(q)
    target = min(target, len(qas))
    quotas = dict.fromkeys(weights, 0)
    for _ in range(target):
        available = [k for k in weights if quotas[k] < len(buckets[k])]
        bucket = max(available, key=lambda k: (target * weights[k] - quotas[k], digest([seed, k])))
        quotas[bucket] += 1
    selected = []
    categories = Counter()
    for bucket in sorted(weights, key=lambda k: digest([seed, k])):
        candidates = list(buckets[bucket])
        for _ in range(quotas[bucket]):
            q = min(candidates, key=lambda q: (categories[q['category']], digest([seed, q['qa_id']])))
            candidates.remove(q)
            selected.append(q)
            categories[q['category']] += 1
    return sorted(selected, key=lambda q: q['qa_id']), quotas
