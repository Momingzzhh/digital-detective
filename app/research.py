"""Local human review and source-bound interactive acceptance sessions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from .auth import session
from .config import ROOT, settings
from .five_dimensions import fingerprint, record_binding, THRESHOLD
from .local_model import model_path, adapter_path
from .store import db, save_case, load_case

router = APIRouter(prefix='/api/research')
CHECKS = ('source_authentic', 'source_offsets_correct', 'sufficient_clues', 'no_outcome_leakage',
          'all_actors_covered', 'gold_correct', 'five_dimensions_correct', 'logic_depth_necessary',
          'conflicts_or_multiple_suspects_real', 'grouping_correct')
BASE = ROOT / 'data/processed/five_dimensions_v1'


def enabled(user=Depends(session)):
    if settings().get('RESEARCH_REVIEW_ENABLED', 'false') != 'true':
        raise HTTPException(403, '人工复核界面未开启，请由本机研究负责人设置RESEARCH_REVIEW_ENABLED=true。')
    return user


def records():
    path = BASE / 'current.private.jsonl'
    return [json.loads(l) for l in path.open()] if path.exists() else []


def get_record(cid):
    row = next((r for r in records() if r['case_id'] == cid), None)
    if not row: raise HTTPException(404, '五维标注案例尚未导出。')
    if row.get('record_binding') != record_binding(row): raise HTTPException(409, '数据绑定已改变。')
    from scripts.export_review_bundle import verify
    try: verify(row, {'raw_record': row['raw_record']})
    except (ValueError, KeyError): raise HTTPException(409, '原文或证据偏移校验失败。')
    return row


def identity(user):
    with db() as con:
        return con.execute('SELECT name FROM users WHERE id=?', (user['user_id'],)).fetchone()[0]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temporary.replace(path)


@router.get('/items')
def items(split: str = 'dev', user=Depends(enabled)):
    return [{'case_id': r['case_id'], 'split': r['split'], 'evidence_count': len(r['evidence']),
             'status': r['status']} for r in records() if r['split'] == split]


@router.get('/items/{cid}')
def item(cid: str, user=Depends(enabled)):
    row = get_record(cid)
    # Explicit reviewer endpoint, never used by the acting agent.
    return {'case_id': cid, 'record_binding': row['record_binding'], 'split': row['split'],
            'source_fulltext': row['raw_record']['qw'], 'source': row['source'],
            'evidence': row['evidence'], 'hidden_gold': row['hidden_gold'],
            'five_dimensions': row['five_dimensions'], 'required_checks': list(CHECKS)}


class HumanReview(BaseModel):
    record_binding: str
    decision: str
    checks: dict[str, bool]
    notes: str = Field(min_length=10, max_length=5000)


@router.post('/items/{cid}/review')
def review(cid: str, value: HumanReview, user=Depends(enabled)):
    row = get_record(cid)
    if value.record_binding != row['record_binding']: raise HTTPException(409, '请刷新后复核最新版本。')
    if value.decision not in ('approve', 'reject'): raise HTTPException(422, '必须明确通过或拒绝。')
    if set(value.checks) != set(CHECKS): raise HTTPException(422, '复核项不完整。')
    if value.decision == 'approve' and not all(value.checks.values()):
        raise HTTPException(422, '存在未核实项目，不可标记通过。')
    record = value.model_dump() | {'case_id': cid, 'reviewer_id': identity(user),
              'reviewed_at': datetime.now(timezone.utc).isoformat(), 'human_attestation': True}
    save(ROOT / 'data/review/five_dimensions_human' / (cid + '.json'), record)
    return {'saved': True, 'decision': value.decision}


def approved(row):
    path = ROOT / 'data/review/five_dimensions_human' / (row['case_id'] + '.json')
    if not path.exists(): return False
    r = json.loads(path.read_text())
    return (r.get('human_attestation') is True and r.get('reviewer_id') and r.get('reviewed_at') and
            r.get('record_binding') == row['record_binding'] and r.get('decision') == 'approve' and
            set(r.get('checks', {})) == set(CHECKS) and all(r['checks'].values()))


def model_binding():
    import hashlib
    path = adapter_path()
    adapter = Path(path) / 'adapters.safetensors' if path else None
    if adapter and not adapter.is_file(): raise HTTPException(409, '适配器文件不存在。')
    manifest = model_path() / 'local_manifest.json'
    return fingerprint({'base': json.loads(manifest.read_text()) if manifest.exists() else None,
        'adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest() if adapter else None})


def acceptance_model_ready():
    report = ROOT / 'data/latest_training_report.json'
    if not report.exists(): return False
    value = json.loads(report.read_text())
    state_path = ROOT / 'data/five_dimensions_report.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not value.get('dataset_binding') or value['dataset_binding'] != state.get('dataset_binding'): return False
    manifest = model_path()/'local_manifest.json'
    if not manifest.exists() or value.get('base_manifest_binding') != fingerprint(json.loads(manifest.read_text())):
        return False
    path = adapter_path()
    if not path or value.get('pilot') or not value.get('actual_weight_update_verified'): return False
    import hashlib
    adapter = Path(path) / 'adapters.safetensors'
    return adapter.is_file() and hashlib.sha256(adapter.read_bytes()).hexdigest() == value.get('adapter_sha256')


@router.post('/items/{cid}/session')
def start(cid: str, user=Depends(enabled)):
    row = get_record(cid)
    if not approved(row): raise HTTPException(409, '请先由研究人员完成此案的原文与五维标注复核。')
    if row['split'] == 'train': raise HTTPException(409, '训练案例不可作为正式验收案件。')
    report = ROOT / 'data/five_dimensions_report.json'
    if not report.exists() or not json.loads(report.read_text()).get('dataset_target_assembled') or not acceptance_model_ready():
        raise HTTPException(409, '600案数据及正式权重训练尚未完成；试运行不能进入正式验收。')
    model = model_binding()
    if row['split'] == 'test':
        score_path = ROOT / 'data/validation_acceptance_report.json'
        validation = json.loads(score_path.read_text()) if score_path.exists() else {}
        current_data = fingerprint([r['record_binding'] for r in records()])
        if (not validation.get('passed') or validation.get('model_binding') != model or
            validation.get('dataset_binding') != current_data):
            raise HTTPException(409, '完整验证集尚未通过，测试集保持关闭。')
        # Opening tests freezes the whole cycle; training cannot silently resume.
        freeze = ROOT / 'runtime/five_dimensions_test_freeze.json'
        if not freeze.exists():
            save(freeze, {'opened_at': datetime.now(timezone.utc).isoformat(),
                          'model_binding': model,
                          'data_binding': fingerprint([r['record_binding'] for r in records()])})
        elif (json.loads(freeze.read_text())['model_binding'] != model or
              json.loads(freeze.read_text())['data_binding'] != current_data):
            raise HTTPException(409, '测试集已绑定另一模型，不可重复调参后再测同一集合。')
        session_file = ROOT / 'runtime/five_dimensions_test_sessions' / (cid + '.json')
        if session_file.exists():
            existing = json.loads(session_file.read_text())
            case = load_case(user['user_id'], existing['session_id'])
            if not case: raise HTTPException(403, '此测试案已由另一账户启动，请使用原账户继续唯一的验收会话。')
            return {'session_case_id': existing['session_id'], 'resumed': True}
    uid = uuid.uuid4().hex
    from scripts.evaluate import visible_case
    case = visible_case(row, row['evidence'][:2])
    case.update(id=uid, title='真人验收 ' + cid,
        graph={'nodes': [], 'edges': []}, trace=[],
        research={'case_id': cid, 'record_binding': row['record_binding'], 'split': row['split'],
                  'mode': 'human_source_bound', 'source_only': True,
                  'model_binding': model, 'analysis_attempts': 0,
                  'released_ids': [e['id'] for e in row['evidence'][:2]]})
    save_case(user['user_id'], case)
    if row['split'] == 'test': save(session_file, {'session_id': uid, 'model_binding': model})
    return {'session_case_id': uid}


class Release(BaseModel):
    evidence_ids: list[str] = Field(min_length=1, max_length=80)


@router.post('/sessions/{sid}/release')
def release(sid: str, value: Release, user=Depends(enabled)):
    case = load_case(user['user_id'], sid)
    if not case or not case.get('research'): raise HTTPException(404, '真人验收会话不存在。')
    # Reuse the same mutation lock as ordinary case analysis.
    from .main import assert_idle
    assert_idle(sid)
    binding = case['research']; row = get_record(binding['case_id'])
    if binding['record_binding'] != row['record_binding']: raise HTTPException(409, '数据版本已改变。')
    ev = {e['id']: e for e in row['evidence']}
    if not set(value.evidence_ids) <= ev.keys(): raise HTTPException(422, '只能释放原始材料中的证据。')
    new = [ev[e] for e in value.evidence_ids if e not in binding['released_ids']]
    if not new: raise HTTPException(409, '所选证据已提供。')
    from scripts.evaluate import visible_case
    case['evidence'].extend(visible_case(row, new)['evidence'])
    case['messages'].append({'role': 'user', 'text': '补充原始材料：\n' + '\n'.join(e['text'] for e in new)})
    binding['released_ids'].extend(e['id'] for e in new)
    save_case(user['user_id'], case)
    return {'released': [e['id'] for e in new]}


class HumanScore(BaseModel):
    chronology_all_correct: bool
    no_unsupported_events: bool
    notes: str = Field(min_length=10, max_length=5000)


@router.post('/sessions/{sid}/score')
def score(sid: str, value: HumanScore, user=Depends(enabled)):
    case = load_case(user['user_id'], sid)
    if not case or not case.get('research') or not case['research'].get('analysis_attempts'):
        raise HTTPException(409, '请先完成真人交互与分析。')
    from .main import assert_idle
    from scripts.evaluate import normalized
    assert_idle(sid)
    binding = case['research']; row = get_record(binding['case_id'])
    if binding.get('record_binding') != row['record_binding']: raise HTTPException(409, '数据版本已改变，请勿对旧会话评分。')
    if not binding.get('source_only') or not approved(row): raise HTTPException(409, '此会话不符合来源约束或人工复核条件。')
    if binding.get('model_binding') != model_binding(): raise HTTPException(409, '分析期间模型版本已改变。')
    score_path = ROOT / 'data/review/five_dimensions_scores' / row['split'] / (row['case_id'] + '.json')
    if row['split'] == 'test' and score_path.exists(): raise HTTPException(409, '最终测试评分已保存，不可重复改分。')
    prediction = case.get('analysis') or {'charges': [], 'perpetrators': [], 'status': '证据不足'}
    gold = row['hidden_gold']
    charge = normalized(prediction['charges'], True) == normalized(gold['charges'], True)
    actor = normalized(prediction['perpetrators']) == normalized(gold['perpetrators'])
    correct = (prediction['status'] != '证据不足' and charge and actor and
               value.chronology_all_correct and value.no_unsupported_events)
    result = {'case_id': row['case_id'], 'record_binding': row['record_binding'], 'split': row['split'],
        'session_id': sid, 'reviewer_id': identity(user), 'human_scored_at': datetime.now(timezone.utc).isoformat(),
        'model_binding': binding['model_binding'], 'analysis_attempts': binding['analysis_attempts'],
        'mode': 'human_source_bound', 'human_analysis_rounds': len(case.get('confidence_history', [])),
        'charge_correct': charge, 'perpetrator_correct': actor, 'all_correct': bool(correct),
        'chronology_correct': bool(value.chronology_all_correct and value.no_unsupported_events),
        'refused_or_failed': prediction['status'] == '证据不足',
        'scoring': value.model_dump(), 'session_sha256': fingerprint(case)}
    save(score_path, result)
    return result
