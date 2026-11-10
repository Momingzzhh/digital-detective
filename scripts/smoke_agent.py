#!/usr/bin/env python3
"""Run the real local Agent through HTTP handlers, using synthetic material only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(args):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    private = ROOT / 'artifacts/agent_smoke' / stamp
    private.mkdir(parents=True)
    os.environ.update(RUNTIME_DIR=str(private), MODEL_BACKEND='local',
                     HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.adapter:
        from app.model_release import file_sha256
        directory = (ROOT / args.adapter).resolve()
        os.environ.update(LOCAL_ADAPTER_PATH=str(directory),
                          LOCAL_ADAPTER_SHA256=file_sha256(directory / 'adapters.safetensors'),
                          LOCAL_ADAPTER_CONFIG_SHA256=file_sha256(directory / 'adapter_config.json'))
    from fastapi.testclient import TestClient
    from app.main import app
    from app.agent import citations_valid
    from app.local_model import adapter_path
    from app.models import Conclusion
    from app.store import budget_status
    attempts = []
    def blocked(*a, **kw):
        attempts.append('network_attempt')
        raise RuntimeError('Network disabled for local Agent smoke')
    socket.socket.connect = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    report = {'generated_at': datetime.now(timezone.utc).isoformat(),
              'synthetic_mechanism_test': True, 'accuracy_claim': False,
              'included_in_dataset_count': False, 'real_model_inference': True,
              'rounds': [], 'checks': {}, 'passed': False}
    report['code_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        'app/agent.py', 'app/main.py', 'app/llm.py', 'app/local_model.py', 'app/local_context.py',
        'app/model_release.py', 'app/models.py', 'scripts/smoke_agent.py')}
    def check(name, value):
        report['checks'][name] = bool(value)
        if not value:
            raise AssertionError(name)
    try:
        with TestClient(app) as client:
            before = budget_status()
            credentials = {'name': '本机交付流程检查', 'password': secrets.token_urlsafe(24)}
            check('registration', client.post('/api/register', json=credentials).status_code == 200)
            login = client.post('/api/login', json=credentials)
            check('login', login.status_code == 200)
            headers = {'X-CSRF-Token': login.json()['csrf']}
            status = client.get('/api/status').json()
            check('local_model_configured', status['configured'] and status['backend'] == 'local')
            material = ('合成软件测试材料，不是真实案件。失主林某报告：11月1日18时发现办公室的一台电脑不见了。'
                        '证人赵某称：当天17时看到维修员陈某进入办公室，但没有看到陈某携带电脑离开。'
                        '维修工单记载：陈某于当天17时到办公室维修打印机。监控设备已损坏，没有录像。'
                        '目前不清楚电脑是否由其他工作人员移走，没有提供指纹或其他物证。')
            upload = client.post('/api/extract-file', headers=headers,
                                 files={'file': ('合成材料.txt', material.encode(), 'text/plain')})
            check('upload_text', upload.status_code == 200 and upload.json()['text'] == material)
            response = client.post('/api/cases', headers=headers,
                                   json={'title': '合成流程检查', 'text': upload.json()['text']})
            check('create_case', response.status_code == 200)
            cid = response.json()['id']
            for turn in (1, 2):
                if turn == 2:
                    supplement = '补充核实：工作人员周某说，自己按部门安排将电脑移到了会议室；失主林某确认已在会议室找到原电脑。'
                    added = client.post(f'/api/cases/{cid}/messages', headers=headers, json={'text': supplement})
                    check('add_evidence', added.status_code == 200)
                started = time.monotonic()
                response = client.post(f'/api/cases/{cid}/analyze', headers=headers)
                check(f'round_{turn}_started', response.status_code == 200)
                stages = []
                while time.monotonic() - started < args.timeout:
                    case = client.get(f'/api/cases/{cid}').json()
                    job = case['job']
                    if job.get('stage') not in stages:
                        stages.append(job.get('stage'))
                        print(json.dumps({'round': turn, 'stage': job.get('stage'), 'state': job['state']}, ensure_ascii=False), flush=True)
                    if job['state'] != 'running':
                        break
                    time.sleep(1)
                (private / f'round-{turn}.private.json').write_text(json.dumps(case, ensure_ascii=False, indent=2))
                check(f'round_{turn}_completed', job['state'] == 'completed')
                conclusion = Conclusion.model_validate(case['analysis'])
                check(f'round_{turn}_citations', not citations_valid(conclusion, case['evidence']))
                sources = {s['id']: s['text'] for s in case['sources']}
                check(f'round_{turn}_original_quotes', bool(case['evidence']) and all(
                    sources[e['source_id']][e['start']:e['end']] == e['quote'] for e in case['evidence']))
                check(f'round_{turn}_no_unsupported_accusation', not conclusion.charges and not conclusion.perpetrators)
                check(f'round_{turn}_review_and_graph', bool(case['review']) and bool(case['graph']['nodes']))
                if turn == 2:
                    check('supplement_extracted', any(e['source_id'] == 'S-2' for e in case['evidence']))
                report['rounds'].append({'round': turn, 'seconds': round(time.monotonic()-started, 2),
                                         'evidence_count': len(case['evidence']), 'stages': stages,
                                         'status': conclusion.status})
            exported = client.get(f'/api/cases/{cid}/export')
            check('export', exported.status_code == 200 and len(exported.json()['confidence_history']) == 2)
            history = exported.json().get('model_history', [])
            check('model_provenance', len(history) == 2 and all(
                item.get('adapter_sha256') == status['deployment'].get('adapter_sha256') for item in history))
            check('delete', client.delete(f'/api/cases/{cid}', headers=headers).status_code == 200
                  and client.get(f'/api/cases/{cid}').status_code == 404)
            check('cloud_ledger_unchanged', budget_status() == before)
            check('logout', client.post('/api/logout', headers=headers).status_code == 200)
            check('network_attempts_zero', not attempts)
            report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)[:300]
        raise
    finally:
        path = adapter_path()
        if path:
            report['adapter_sha256'] = hashlib.sha256((Path(path) / 'adapters.safetensors').read_bytes()).hexdigest()
        report['network_attempts'] = len(attempts)
        content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
        (private / 'report.json').write_text(content)
        (ROOT / args.output).write_text(content)
        print(content, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter')
    parser.add_argument('--output', default='data/local_agent_smoke_report.json')
    parser.add_argument('--timeout', type=int, default=900)
    main(parser.parse_args())
