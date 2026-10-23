"""Read-only loopback preview of the latest private training calibration page.

Run with: .venv/bin/python -m uvicorn scripts.calibration_server:app --host 127.0.0.1 --port 8766
"""
import base64
import hashlib
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from scripts.prepare_calibration_review import ROOT, RUBRIC

app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)


@app.get('/')
def index():
    folder=ROOT/'artifacts/calibration'/RUBRIC
    pointer=folder/'manifest.private.json'
    if not pointer.exists():raise HTTPException(404,'请先生成训练集校准包。')
    binding=json.loads(pointer.read_text())['bundle_binding']
    if not re.fullmatch('[0-9a-f]{64}',binding):raise HTTPException(409,'校准包绑定无效。')
    page=(folder/binding[:16]/'index.html').read_text()
    script=re.search(r'<script>\n(.*?)</script>',page,re.S).group(1)
    # The leading newline is part of the actual script text.
    digest=base64.b64encode(hashlib.sha256(('\n'+script).encode()).digest()).decode()
    return HTMLResponse(page,headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
        'X-Frame-Options':'DENY','Referrer-Policy':'no-referrer',
        'Content-Security-Policy':f"default-src 'none'; script-src 'sha256-{digest}'; style-src 'unsafe-inline'; connect-src 'none'; frame-ancestors 'none'; form-action 'none'"})
