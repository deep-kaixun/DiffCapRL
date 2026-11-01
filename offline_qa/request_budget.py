"""Persistent, process-safe ceiling on network attempts (including retries)."""
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class RequestBudgetExceeded(RuntimeError):
    pass


class RequestBudget:
    def __init__(self, path, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError('invalid_request_budget')
        self.path = Path(path)
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), ceiling INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (role TEXT, request_hash TEXT, reserved_at REAL)')
            db.execute('INSERT OR IGNORE INTO budget VALUES (1,?)', (limit,))
            if db.execute('SELECT ceiling FROM budget WHERE id=1').fetchone()[0] != limit:
                raise ValueError('request_budget_limit_changed')

    def reserve(self, role, request_hash):
        # A crash after reservation conservatively consumes an attempt. Never refund
        # transport failures: providers can bill requests whose responses are lost.
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            count = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
            if count >= self.limit:
                raise RequestBudgetExceeded('network_request_budget_exhausted')
            db.execute('INSERT INTO attempts VALUES (?,?,?)', (role, request_hash, time.time()))
            return count + 1
