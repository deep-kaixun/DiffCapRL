"""OpenAI-compatible requests with bounded retries and validated-response caching."""
import json
import os
import sqlite3
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schema import digest, parse_json
from .request_budget import RequestBudget


class APIError(RuntimeError):
    pass


def resolve_model(config, role):
    c = config['clients'][role]
    model = c.get('model') or os.environ.get(c.get('model_env', ''), '')
    if not model or not model.strip():
        raise ValueError('missing_model_for_' + role)
    return model


def resolve_endpoint(config, role):
    c = config['clients'][role]
    base = c.get('base_url') or os.environ.get(c.get('base_url_env', ''), '')
    if not base:
        raise ValueError('missing_endpoint_for_' + role)
    if not base.startswith(('http://', 'https://')):
        raise ValueError('invalid_endpoint_for_' + role)
    return base.rstrip('/') + '/chat/completions'


class Client:
    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir
        self.lock = threading.Lock()
        budget = config.get('request_budget')
        self.budget = RequestBudget(budget['ledger'], budget['limit']) if budget else None
        self.db = sqlite3.connect(run_dir / 'responses.sqlite3', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, raw TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS usage_events '
                        '(role TEXT, event TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, '
                        'cached_prompt_tokens INTEGER, usage_reported INTEGER)')
        self.db.commit()

    def close(self):
        stats = {}
        for role, event, count, prompt, completion, cached, reported in self.db.execute(
                'SELECT role,event,COUNT(*),SUM(prompt_tokens),SUM(completion_tokens),'
                'SUM(cached_prompt_tokens),SUM(usage_reported) FROM usage_events GROUP BY role,event'):
            entry = stats.setdefault(role, {'network_attempts': 0, 'successful_requests': 0,
                                           'cache_hits': 0, 'failed_attempts': 0,
                                           'prompt_tokens_reported': 0, 'completion_tokens_reported': 0,
                                           'cached_prompt_tokens_reported': 0, 'attempts_without_usage': 0})
            if event == 'cache_hit':
                entry['cache_hits'] += count
            else:
                entry['network_attempts'] += count
                entry['successful_requests' if event == 'success' else 'failed_attempts'] += count
                entry['prompt_tokens_reported'] += prompt
                entry['completion_tokens_reported'] += completion
                entry['cached_prompt_tokens_reported'] += cached
                entry['attempts_without_usage'] += count - reported
        path = self.run_dir / 'usage.json'
        tmp = path.with_suffix('.json.partial')
        tmp.write_text(json.dumps({'roles': stats, 'notes': [
            'Counts include retries; cache hits make no new model request.',
            'Token usage is provider-reported, not a price quote or invoice.',
            'Missing usage is unknown; it does not mean the request was free.']}, indent=2) + '\n')
        os.replace(tmp, path)
        self.db.close()

    def record_usage(self, role, event, raw=None):
        usage = raw.get('usage') if isinstance(raw, dict) else None
        reported = isinstance(usage, dict) and 'prompt_tokens' in usage and 'completion_tokens' in usage
        usage = usage if isinstance(usage, dict) else {}
        prompt = usage.get('prompt_tokens', 0) or 0
        completion = usage.get('completion_tokens', 0) or 0
        details = usage.get('prompt_tokens_details') or {}
        cached = details.get('cached_tokens', 0) or 0
        with self.lock:
            self.db.execute('INSERT INTO usage_events VALUES (?,?,?,?,?,?)',
                            (role, event, prompt, completion, cached, int(reported)))
            self.db.commit()

    def endpoint(self, role):
        return resolve_endpoint(self.config, role)

    def call(self, role, messages, validator, seed=None):
        c = self.config['clients'][role]
        endpoint = self.endpoint(role)
        payload = {'model': resolve_model(self.config, role), 'messages': messages,
                   'temperature': c['temperature'], 'max_tokens': c['max_tokens']}
        if seed is not None:
            payload['seed'] = seed
        extra = c.get('extra_body', {})
        if set(extra) & {'model', 'messages', 'temperature', 'max_tokens', 'seed'}:
            raise ValueError('extra_body_overrides_core_settings')
        payload.update(extra)
        key = digest([endpoint, c.get('revision', ''), payload])
        with self.lock:
            saved = self.db.execute('SELECT raw FROM responses WHERE key=?', (key,)).fetchone()
        if saved:
            self.record_usage(role, 'cache_hit')
            return validator(json.loads(saved[0])['choices'][0]['message']['content'])
        token = os.environ.get(c.get('api_key_env', 'OPENAI_API_KEY'), '')
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        error = 'unknown'
        for attempt in range(self.config['request_attempts']):
            raw = None
            try:
                req = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method='POST')
                if self.budget is not None:
                    self.budget.reserve(role, key)
                with urlopen(req, timeout=self.config['timeout_seconds']) as response:
                    raw = json.load(response)
                choice = raw['choices'][0]
                if choice.get('finish_reason') == 'length':
                    raise ValueError('truncated_response')
                content = choice['message']['content']
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('empty_response')
                result = validator(content)
                with self.lock:
                    self.db.execute('INSERT OR REPLACE INTO responses VALUES (?,?)',
                                    (key, json.dumps(raw, ensure_ascii=False)))
                    self.db.commit()
                self.record_usage(role, 'success', raw)
                return result
            except HTTPError as exc:
                error = 'http_' + str(exc.code)
                if exc.code not in (408, 429) and exc.code < 500:
                    self.record_usage(role, 'failure', raw)
                    raise APIError(role + ':' + error) from None
            except (URLError, TimeoutError, OSError) as exc:
                error = type(exc).__name__
            except (ValueError, KeyError, TypeError, IndexError):
                error = 'invalid_response'
            self.record_usage(role, 'failure', raw)
            # Raw invalid model responses stay in the private run directory, never become successes.
            with self.lock:
                with (self.run_dir / 'request_failures.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'request_hash': key, 'role': role, 'attempt': attempt + 1,
                                        'error': error, 'response': raw}, ensure_ascii=False) + '\n')
            if attempt + 1 < self.config['request_attempts']:
                time.sleep(self.config['retry_delay_seconds'] * (2 ** attempt))
        raise APIError(role + ':' + error)

    def json(self, role, messages, validator, seed=None):
        return self.call(role, messages, lambda text: validator(parse_json(text)), seed)
