import hashlib
import hmac
import secrets
import time
from fastapi import HTTPException, Request
from .store import db


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 310000).hex()
    return salt + ':' + digest


def verify_password(password, stored):
    return hmac.compare_digest(hash_password(password, stored.split(':')[0]), stored)


def session(request: Request):
    token = request.cookies.get('detective_session', '')
    with db() as con:
        row = con.execute('SELECT * FROM sessions WHERE token=? AND expires>?',
                          (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
    if not row:
        raise HTTPException(401, '请先登录。')
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        if not hmac.compare_digest(request.headers.get('x-csrf-token', ''), row['csrf']):
            raise HTTPException(403, '会话校验失败，请刷新后重试。')
    return dict(row)
