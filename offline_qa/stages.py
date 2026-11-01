"""All eight stages. Calibration and sampling are included in the default workflow."""
import json
import os
import re
from collections import Counter

from .common import PACKAGE, Rejected, NeedsReview, latest_records, text_message, write_json
from .schema import (IMAGE_CATEGORIES, LETTERS, VIDEO_CATEGORIES, answer_batch, answer_index,
                     digest, generated_items, normalize_pool, public_question, question_text,
                     rotate, sample_pool)


def nonempty_text(text):
    if not text.strip():
        raise ValueError('empty_caption')
    return text.strip()


def qa_or_reject(row, qas, stage, decisions):
    result = {**row, 'qa': qas, stage + '_decisions': decisions}
    if not qas:
        raise Rejected('no_questions_after_' + stage, result)
    return result


def prepare(ctx, row):
    return row


def generate(ctx, row):
    name = 'video_paper.txt' if row['media_type'] == 'video' else 'image_reference.txt'
    prompt = (PACKAGE / 'prompts' / name).read_text()
    target = ctx.config['target_questions'][row['media_type']]
    candidate_limit = target + ctx.config.get('candidate_reserve', 2)
    prompt = re.sub(r'\{(target_questions|candidate_limit)\}',
                    lambda m: str({'target_questions': target, 'candidate_limit': candidate_limit}[m[1]]), prompt)
    existing = []
    supplements = getattr(ctx, 'supplements', None)
    if supplements is not None and row['id'] in supplements:
        old = supplements[row['id']]
        existing = old['data'].get('qa', [])
        if old['status'] == 'ok':
            return {**row, 'candidates': existing, 'generation_prompt': 'reused_accepted_pool',
                    'supplemented': False}
        missing = max(1, ctx.config['target_questions'][row['media_type']] - len(existing))
        prompt += ('\nSupplement task: the existing questions below are retained. Generate only '
                   + str(missing) + ' NEW questions about other visible facts, without repeating them. '
                   'Use the same output schema. Existing questions:\n'
                   + json.dumps([public_question(q) for q in existing], ensure_ascii=False))

    def validate(value):
        items = generated_items(value)
        if not items:
            raise ValueError('empty_generated_pool')
        unique, _ = normalize_pool(existing + items, row['id'], ctx.config['seed'])
        if len(items) > candidate_limit or len(unique) > candidate_limit:
            raise ValueError('generated_pool_exceeds_candidate_limit')
        return items

    items = ctx.client.json('offline_qa_maker', ctx.media_messages(row, prompt, 'offline_qa_maker'), validate)
    return {**row, 'candidates': existing + items, 'generation_prompt': name,
            'generation_prompt_sha256': digest(prompt)}


def validate(ctx, row):
    qa, rejected = normalize_pool(row['candidates'], row['id'], ctx.config['seed'])
    base = {k: v for k, v in row.items() if k != 'candidates'}
    return qa_or_reject(base, qa, 'validation', rejected)


def blind_answer(ctx, q):
    prompt = ('Answer this single multiple-choice question without any image, video, caption, or reference answer. '
              'If you cannot determine the answer, return null. Output only JSON: {"answer":"A"} '
              '(A/B/C/D or null).\n' + question_text(q))

    def validate(value):
        if not isinstance(value, dict) or 'answer' not in value:
            raise ValueError('missing_blind_answer')
        return answer_index(value['answer'])

    return ctx.client.json('blind_solver', text_message(prompt), validate)


def audit_question_pool(ctx, row):
    """Require a text-only answerability check for every candidate, then remove sources."""
    questions = row['qa']
    payload = [{'id': i, 'stem': q['question'], 'choices': q['choices']}
               for i, q in enumerate(questions)]
    prompt = (
        '# Pool independence audit\n'
        'For EVERY target question, try to select one option using ONLY factual assertions in the supplied '
        'question STEMS, including its own stem. No image, video, caption, reference answers or world '
        'knowledge are available. This is a text-leakage test, not visual verification. '
        'Check all other stems before answering null. A source must refer to the same object or fact: '
        'sharing a scene or a word does not prove object identity. Options are alternatives, never assertions. '
        'A partial asserted attribute is sufficient if it rules out every distractor. For example, a source '
        'calling the same object red can reveal an option "small red metal box" when every other option '
        'has a different color; it need not establish size and material as well. '
        'If one option can be selected from a stem, give A/B/C/D, the SOURCE id and an exact revealing '
        'substring of that source STEM. For self-answering questions the source is the target itself. '
        'If no option is determined from stems, give answer_from_stems=null, source_id=null, quote="". '
        'Do not guess a visual detail. Give one concise explanation per target. '
        'Separately group genuine duplicate tests of the same subject and tested attribute. Price versus '
        'quantity, color versus material, and shared objects alone are not duplicates. Do not group a '
        'question twice. Do not impose any retention quota. '
        'Return JSON only {"checks":[{"id":0,"answer_from_stems":null,"source_id":null, '
        '"quote":"","explanation":"Requires media"}],"duplicate_groups":[]}. '
        'A duplicate group is {"ids":[0,1],"explanation":"Same fact"}. Include every target id exactly '
        'once in checks, even when there are no leaks. IDs are local integers, not answer positions. '
        'Questions:\n' + json.dumps(payload, ensure_ascii=False))

    def validate_audit(value):
        if (not isinstance(value, dict) or not isinstance(value.get('checks'), list)
                or not isinstance(value.get('duplicate_groups'), list)):
            raise ValueError('invalid_pool_audit')
        seen = set()
        checks = []
        issues = []
        for item in value['checks']:
            if not isinstance(item, dict):
                raise ValueError('invalid_pool_audit_check')
            target = item.get('id')
            if type(target) is not int or target not in range(len(questions)) or target in seen:
                raise ValueError('invalid_pool_audit_target')
            seen.add(target)
            if 'answer_from_stems' not in item:
                raise ValueError('missing_stem_answer')
            selected = answer_index(item['answer_from_stems'])
            source = item.get('source_id')
            quote = item.get('quote')
            explanation = item.get('explanation')
            if (not isinstance(quote, str) or not isinstance(explanation, str)
                    or not explanation.strip()):
                raise ValueError('invalid_stem_answer_evidence')
            if selected is None:
                if source is not None or quote.strip():
                    raise ValueError('null_stem_answer_has_source')
            else:
                if type(source) is not int or source not in range(len(questions)):
                    raise ValueError('invalid_pool_audit_source')
                if not quote.strip() or quote not in questions[source]['question']:
                    raise ValueError('missing_literal_leak_evidence')
                issues.append({'id': questions[source]['qa_id'],
                               'reason': 'self_answer_leak' if source == target else 'answer_leaking_stem',
                               'related_ids': [questions[target]['qa_id']], 'quote': quote,
                               'explanation': explanation, 'stem_answer_index': selected})
            checks.append({'qa_id': questions[target]['qa_id'], 'stem_answer_index': selected,
                           'source_id': questions[source]['qa_id'] if source is not None else None,
                           'quote': quote, 'explanation': explanation})
        if seen != set(range(len(questions))):
            raise ValueError('incomplete_pool_audit_coverage')
        grouped = set()
        removed = {issue['id'] for issue in issues}
        for group in value['duplicate_groups']:
            if not isinstance(group, dict):
                raise ValueError('invalid_duplicate_group')
            ids = group.get('ids')
            explanation = group.get('explanation')
            if (not isinstance(ids, list) or len(ids) < 2
                    or any(type(i) is not int or i not in range(len(questions)) for i in ids)
                    or len(set(ids)) != len(ids) or grouped.intersection(ids)
                    or not isinstance(explanation, str) or not explanation.strip()):
                raise ValueError('invalid_duplicate_group')
            grouped.update(ids)
            eligible = [i for i in sorted(ids) if questions[i]['qa_id'] not in removed]
            for i in eligible[1:]:
                qid = questions[i]['qa_id']
                issues.append({'id': qid, 'reason': 'redundant_fact',
                               'related_ids': [questions[eligible[0]]['qa_id']],
                               'explanation': explanation})
                removed.add(qid)
        return {'checks': checks, 'issues': issues}

    audit = ctx.client.json('dependency_checker', text_message(prompt), validate_audit)
    rejected = {issue['id'] for issue in audit['issues']}
    result = {**row, 'qa': [q for q in questions if q['qa_id'] not in rejected],
              'pool_audit_decisions': audit['issues'], 'pool_audit_checks': audit['checks'],
              'pool_audit_version': 'stem_answer_coverage_v3'}
    if not result['qa']:
        raise NeedsReview('no_independent_questions_after_pool_audit', result)
    return result


def blind_filter(ctx, row):
    if ctx.config.get('pool_audit', True):
        row = audit_question_pool(ctx, row)
    kept, decisions = [], []
    for q in row['qa']:
        first = blind_answer(ctx, q)
        decision = {'qa_id': q['qa_id'], 'first_answer_index': first, 'keep': True,
                    'reason': 'blind_miss_or_unknown'}
        if first == q['answer_index']:
            prompt = (
                'Determine whether this question can be reliably answered WITHOUT seeing its specific image/video. '
                'Look for world-knowledge-only answers, a revealed answer in the question, or decisive option clues. '
                'A guess is not evidence of independence from vision. Do not infer any particular visual scene. '
                'Output JSON {"blind_answerable":true/false,"reason":"short explanation"}.\n' + question_text(q))

            def validate(value):
                if not isinstance(value, dict) or type(value.get('blind_answerable')) is not bool or not isinstance(value.get('reason'), str):
                    raise ValueError('invalid_dependency_decision')
                return value

            dependency = ctx.client.json('dependency_checker', text_message(prompt), validate)
            decision.update(dependency=dependency, reason='blind_hit_but_requires_visual_evidence')
            if dependency['blind_answerable']:
                permuted = q
                checks = []
                for i in range(ctx.config['blind_permutation_checks']):
                    permuted = rotate(permuted)
                    answer = blind_answer(ctx, permuted)
                    checks.append({'rotation': i+1, 'answer_index': answer,
                                   'matches_reference': answer == permuted['answer_index']})
                    if answer != permuted['answer_index']:
                        break
                decision.update(permutation_checks=checks, reason='unstable_blind_answer')
                if all(c['matches_reference'] for c in checks):
                    decision.update(keep=False, reason='consistent_blind_answer_and_dependency')
        decisions.append(decision)
        if decision['keep']:
            kept.append(q)
    return qa_or_reject(row, kept, 'blind_solver', decisions)


def verify(ctx, row):
    categories = VIDEO_CATEGORIES if row['media_type'] == 'video' else IMAGE_CATEGORIES

    def check(role, qas):
        questions = [public_question(q) for q in qas]
        prompt = (
            'Independently answer these questions from VISIBLE evidence in the supplied media. Ignore audio. '
            'No reference answers or other reviewers answers are provided. '
            'Do not use other questions or world knowledge to fill missing visual facts. Abstain with answer=null if you '
            'cannot resolve the detail: inability to answer does not establish that the question is wrong. '
            'For each question, check visual answerability and whether exactly one option is supported. '
            'State the actual visible evidence and its location (image region; for video, approximate '
            'timestamp or interval and region). Do not invent an exact timestamp when unavailable. '
            'Set evidence_sufficient=false for uncertain or missing evidence. '
            'Category codes: ' + ', '.join(categories) + '. '
            'Video: Bkg=background, Evt=event, Ins=instances/count, IAct=action, IApp=appearance/OCR, '
            'IPos=position, Mov=camera motion, Per=viewpoint, Reg=region. '
            'Image: Glo=global scene, Bkg=background, Ins=count, IAtt=attribute, IAct=action/state, '
            'Spa=spatial relation, Txt=OCR, Reg=region. '
            'Output only JSON {"answers":[{"id":"question ID","answer":"A",'
            '"visually_answerable":true,"unique":true,"evidence_sufficient":true,'
            '"category":"Bkg","evidence":"visible fact","evidence_location":"region/time",'
            '"abstention_reason":""}]}. Use A/B/C/D or null; include every ID exactly once.\n'
            + json.dumps(questions, ensure_ascii=False))
        if row['media_type'] == 'video':
            prompt = prompt.replace(
                'Image: Glo=global scene, Bkg=background, Ins=count, IAtt=attribute, IAct=action/state, '
                'Spa=spatial relation, Txt=OCR, Reg=region. ',
                'Use ONLY the listed video category codes. Object motion is IAct; Mov is camera motion. '
                'Use IPos for spatial relations and IApp for appearance. Do not use image category codes. ')

        def validate(value):
            parsed = answer_batch(value, qas)
            results = {}
            for item in value['answers']:
                if any(type(item.get(k)) is not bool for k in ('visually_answerable', 'unique', 'evidence_sufficient')):
                    raise ValueError('missing_visual_checks')
                if item.get('category') not in categories:
                    raise ValueError('invalid_visual_category')
                item = dict(item)
                item.setdefault('abstention_reason', '')
                for field in ('evidence', 'evidence_location', 'abstention_reason'):
                    if field in item and item[field] is None:
                        item[field] = ''
                if any(not isinstance(item.get(k), str) for k in ('evidence', 'evidence_location', 'abstention_reason')):
                    raise ValueError('missing_evidence_fields')
                if item['evidence_sufficient'] and (not item['evidence'].strip() or not item['evidence_location'].strip()):
                    raise ValueError('claimed_evidence_has_no_description_or_location')
                results[str(item['id'])] = {**item, 'answer_index': parsed[str(item['id'])], 'role': role}
            return results

        return ctx.client.json(role, ctx.media_messages(row, prompt, role), validate, seed=ctx.config['seed'])

    def grounded(v):
        return (v['answer_index'] is not None and v['visually_answerable'] and v['unique']
                and v['evidence_sufficient'] and bool(v['evidence'].strip()) and bool(v['evidence_location'].strip()))

    def supports(v, q):
        return grounded(v) and v['answer_index'] == q['answer_index']

    weak = check('weak_verifier', row['qa'])
    full_teacher = ctx.config['verification_strategy'] == 'full_teacher'
    unresolved = [q for q in row['qa'] if not supports(weak[q['qa_id']], q)]
    first_pool = row['qa'] if full_teacher else unresolved
    teacher_1 = check('teacher_1', first_pool) if first_pool else {}
    second_pool = [q for q in first_pool if not supports(teacher_1[q['qa_id']], q)
                   or (full_teacher and not supports(weak[q['qa_id']], q))]
    teacher_2 = check('teacher_2', second_pool) if second_pool else {}
    kept, decisions, review = [], [], []
    for q in row['qa']:
        qid = q['qa_id']
        w, t1, t2 = weak[qid], teacher_1.get(qid), teacher_2.get(qid)
        evidence_check = None
        reason = 'insufficient_visual_evidence'
        if full_teacher:
            if supports(w, q) and supports(t1, q):
                evidence_check, reason = t1, 'weak_and_teacher_1_support_reference'
            elif t2 is not None and supports(t1, q) and supports(t2, q):
                evidence_check, reason = t1, 'both_teachers_support_reference'
        else:
            if supports(w, q):
                evidence_check, reason = w, 'weak_supports_reference'
            elif t1 is not None and supports(t1, q):
                evidence_check, reason = t1, 'teacher_1_supports_reference'
            elif t2 is not None and supports(t2, q) and not grounded(t1):
                evidence_check, reason = t2, 'teacher_2_supports_reference_after_teacher_1_abstention'
        accepted = evidence_check is not None
        if not accepted and t1 is not None and t2 is not None:
            if grounded(t1) and grounded(t2) and t1['answer_index'] != t2['answer_index']:
                reason = 'teachers_disagree'
            elif grounded(t1) and grounded(t2) and t1['answer_index'] == t2['answer_index'] != q['answer_index']:
                reason = 'teachers_agree_on_alternative_reference_needs_review'
        checks = [w] + ([t1] if t1 is not None else []) + ([t2] if t2 is not None else [])
        decision = {'qa_id': qid, 'action': 'keep' if accepted else 'review', 'reason': reason,
                    'original_answer': LETTERS[q['answer_index']], 'checks': checks,
                    'verification_strategy': ctx.config['verification_strategy'],
                    'teacher_checked': t1 is not None,
                    'reference_supporting_checks': sum(supports(v, q) for v in checks),
                    'accepted_by': evidence_check['role'] if accepted else None}
        decisions.append(decision)
        if accepted:
            kept.append({**q, 'category': evidence_check['category'], 'visual_evidence': evidence_check['evidence'],
                         'visual_evidence_location': evidence_check['evidence_location'], 'verification': decision})
        else:
            review.append({**q, 'verification': decision})
    result = {**row, 'qa': kept, 'verification_decisions': decisions, 'review_qa': review}
    if not kept:
        raise NeedsReview('no_auto_verified_questions', result)
    return result


def write_review_queue(run_dir, records):
    path = run_dir / '5_review_queue.jsonl'
    temp = path.with_suffix('.jsonl.partial')
    count = 0
    with temp.open('w', encoding='utf-8') as f:
        for record in sorted(records, key=lambda r: r['id']):
            data = record.get('data', {})
            for q in data.get('review_qa', []):
                f.write(json.dumps({'media_id': record['id'], 'media_type': data['media_type'],
                                    'media_path': data.get('media_path'), 'url_key': record['id'],
                                    'sources': data['sources'], 'qa': q}, ensure_ascii=False) + '\n')
                count += 1
    os.replace(temp, path)
    return count


def calibrate(ctx, row):
    captions, predictions = [], []
    solver_prompt = (PACKAGE / 'prompts' / 'solver_paper.txt').read_text()
    if row['media_type'] == 'image':
        solver_prompt = solver_prompt.replace('Video Caption', 'Image Caption')
    # Use positional integer IDs as in the published solver prompt; translate back to stable QA IDs.
    indexed = [{**q, 'qa_id': str(i)} for i, q in enumerate(row['qa'])]
    questions = [{'id': i, 'question': question_text(q)} for i, q in enumerate(row['qa'])]
    for index in range(ctx.config['captions_per_media']):
        seed = int(digest([ctx.config['seed'], row['id'], index])[:8], 16) % (2 ** 31)
        caption = ctx.client.call('captioner', ctx.media_messages(row, f'Describe this {row["media_type"]} in detail.', 'captioner'),
                                  nonempty_text, seed=seed)
        # Only the paper's two placeholders are replaced; JSON example braces remain intact.
        replacements = {'caption': caption, 'questions_json': json.dumps(questions, ensure_ascii=False)}
        prompt = re.sub(r'\{(caption|questions_json)\}', lambda m: replacements[m[1]], solver_prompt)
        answers = ctx.client.json('offline_qa_solver', text_message(prompt), lambda value: answer_batch(value, indexed))
        captions.append({'rollout_index': index, 'requested_seed': seed, 'caption': caption})
        predictions.append(answers)
    output = []
    for i, q in enumerate(row['qa']):
        answers = [p[str(i)] for p in predictions]
        correct = sum(a == q['answer_index'] for a in answers)
        n = len(answers)
        output.append({**q, 'calibration': {'answers': answers, 'correct': correct, 'trials': n,
                                           'accuracy': correct / n, 'unknown_rate': answers.count(None) / n,
                                           'discriminative': 0 < correct < n,
                                           'bucket': 'hard' if correct == 0 else 'easy' if correct == n else 'medium'}})
    return {**row, 'qa': output, 'captioner_outputs': captions,
            'calibration_diagnostics': {'caption_rollouts': len(captions),
                                        'distinct_caption_texts': len({c['caption'] for c in captions}),
                                        'difficulty_measures_captioner_and_solver': True}}


def sample(ctx, row):
    selected, quotas = sample_pool(row['qa'], ctx.config['target_questions'][row['media_type']],
                                   ctx.config['difficulty_weights'], ctx.config['seed'])
    if len(selected) < ctx.config['min_questions']:
        raise Rejected('insufficient_verified_questions', {**row, 'sampling_quotas': quotas,
                                                         'available_questions': len(selected)})
    ids = {q['qa_id'] for q in selected}
    return {**row, 'qa': selected, 'sampling_quotas': quotas,
            'sampling_decisions': [{'qa_id': q['qa_id'], 'keep': q['qa_id'] in ids,
                                    'reason': 'selected' if q['qa_id'] in ids else 'difficulty_category_quota'}
                                   for q in row['qa']]}


def export(ctx, row):
    qas = [{'qa_id': q['qa_id'], 'question': question_text(q), 'ref_answer': LETTERS[q['answer_index']],
            'modality': 'Visual', 'category': q['category'], 'focus': q.get('focus', ''),
            'type': 'step_2_gen_video_qa' if row['media_type'] == 'video' else 'step2_image_gen_qa',
            'claim_source': row['media_type'], 'verification_strategy': q['verification']['verification_strategy'],
            'verified_by': q['verification']['accepted_by'], 'teacher_checked': q['verification']['teacher_checked']} for q in row['qa']]
    source_rows = row['sources'] if ctx.config['expand_sources'] else [{'id': row['id'], 'source': 'merged_media'}]
    items = []
    for source in source_rows:
        text = f'Describe this {row["media_type"]} in detail.'
        item = {'id': source['id'], 'media_id': row['id'], 'data_type': row['media_type'],
                'data_source': source['source'], 'problem_type': 'caption',
                'problem': f'<{row["media_type"]}> ' + text, 'problem_reserved_text': text,
                'options': [], 'answer': [], 'qa_offline': qas, 'url_key': row['id']}
        if row.get('media_path'):
            item['images' if row['media_type'] == 'image' else 'videos'] = [row['media_path']]
        else:
            # A generated dataset needs local media before ordinary EasyR1 training can read it.
            item['media_download_required'] = True
        items.append(item)
    return {'media_id': row['id'], 'items': items}


def write_export(run_dir, rows):
    items = sorted([item for row in rows for item in row['items']], key=lambda r: r['id'])
    if len({r['id'] for r in items}) != len(items):
        raise ValueError('duplicate_export_ids')
    path = run_dir / 'train.jsonl'
    temp = path.with_suffix('.jsonl.partial')
    with temp.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(temp, path)
    write_json(run_dir / 'train.json', items)
    verified = list(latest_records(run_dir / '5_verified.jsonl').values())
    reviews = [q for r in verified for q in r.get('data', {}).get('review_qa', [])]
    review_media = sum(bool(r.get('data', {}).get('review_qa')) for r in verified)
    decisions = [d for r in verified for d in r.get('data', {}).get('verification_decisions', [])]
    write_json(run_dir / 'summary.json', {'protocol': 'offline-qa-visual-v2', 'samples': len(items),
                                         'unique_media': len(rows), 'qa_count_with_sample_weights': sum(len(r['qa_offline']) for r in items),
                                         'media_counts': dict(Counter(r['data_type'] for r in items)),
                                         'answer_counts': dict(Counter(q['ref_answer'] for r in items for q in r['qa_offline'])),
                                         'pending_review_questions': len(reviews), 'media_with_pending_review': review_media,
                                         'verification_reasons': dict(Counter(d['reason'] for d in decisions)),
                                         'teacher_checked_questions': sum(d['teacher_checked'] for d in decisions),
                                         'weak_only_accepted_questions': sum(d['action'] == 'keep' and not d['teacher_checked'] for d in decisions),
                                         'review_queue': '5_review_queue.jsonl',
                                         'review_complete': not reviews,
                                         'notes': ['Train output contains only automatically verified and sampled QA.',
                                                   'Review QA remain unresolved, not proven incorrect; no answers were relabeled.']})


PROCESSORS = {1: prepare, 2: generate, 3: validate, 4: blind_filter, 5: verify,
              6: calibrate, 7: sample, 8: export}
