import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from .config import runtime


@contextmanager
def db():
    con = sqlite3.connect(runtime() / 'detective.sqlite', timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute('PRAGMA journal_mode=WAL')
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize():
    with db() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, name TEXT UNIQUE, password TEXT);
        CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id TEXT, csrf TEXT, expires REAL);
        CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, user_id TEXT, title TEXT, data TEXT, updated REAL);
        CREATE TABLE IF NOT EXISTS usage(id TEXT PRIMARY KEY, purpose TEXT, model TEXT, reserved REAL,
          cost REAL, input_tokens INTEGER, output_tokens INTEGER, state TEXT, created REAL);
        ''')


def save_case(user_id, data):
    with db() as con:
        con.execute('INSERT INTO cases VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data,title=excluded.title,updated=excluded.updated',
                    (data['id'], user_id, data['title'], json.dumps(data, ensure_ascii=False), time.time()))


def load_case(user_id, case_id):
    with db() as con:
        row = con.execute('SELECT data FROM cases WHERE id=? AND user_id=?', (case_id, user_id)).fetchone()
    return json.loads(row['data']) if row else None


def budget_status():
    with db() as con:
        row = con.execute('SELECT COALESCE(SUM(cost),0) cost, COALESCE(SUM(reserved),0) reserved, COUNT(*) calls FROM usage').fetchone()
    return dict(row)


def reserve(purpose, model, amount, limit):
    call_id = uuid.uuid4().hex
    with db() as con:
        con.execute('BEGIN IMMEDIATE')
        total = con.execute('SELECT COALESCE(SUM(cost+reserved),0) FROM usage').fetchone()[0]
        if total + amount > limit:
            raise ValueError('已达到本地预算上限，本次调用未发出。')
        con.execute('INSERT INTO usage VALUES(?,?,?,?,?,?,?,?,?)',
                    (call_id, purpose, model, amount, 0, 0, 0, 'pending', time.time()))
    return call_id


def settle(call_id, cost, inputs, outputs):
    with db() as con:
        con.execute("UPDATE usage SET reserved=0,cost=?,input_tokens=?,output_tokens=?,state='completed' WHERE id=?",
                    (cost, inputs, outputs, call_id))


def uncertain(call_id):
    # Preserve the reservation on timeouts and errors: the provider may have billed.
    with db() as con:
        con.execute("UPDATE usage SET state='uncertain' WHERE id=?", (call_id,))
