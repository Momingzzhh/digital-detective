"""Authenticated train-only human review. Never opens a formal benchmark gate."""
from datetime import datetime, timezone
import json
import re
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field, StrictBool
from .config import ROOT
from .models import StrictModel
from .research import enabled
from .environment_models import PrivateCase
from .experiment_manifest import digest
from .interactive_curation import reachability

router = APIRouter(prefix='/api/research/v2')
BASE = ROOT/'data/processed/research_v2'
RECEIPTS = ROOT/'data/review/research_v2_human'


def bundle():
    try:
        pointer = json.loads((BASE/'current.json').read_text())
        if not re.fullmatch('[0-9a-f]{16}', pointer['directory']): raise ValueError('Invalid directory')
        folder = BASE/pointer['directory']
        manifest = json.loads((folder/'manifest.private.json').read_text())
        raw = json.loads((folder/'cases.private.json').read_text())
        binding = digest({'source_file': manifest['source_sha256'], 'cases': raw})
        if binding != pointer['binding'] or binding != manifest['binding']:
            raise ValueError('Bundle changed')
        cases = [PrivateCase.model_validate(c) for c in raw]
        if len({c.case_id for c in cases}) != len(cases): raise ValueError('Duplicate case')
        return folder, cases
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, '训练复核材料不存在或绑定失效，请重新构建。') from exc


def required_checks(case):
    checks = {'case:grouping': '关联案件与历史训练身份核对',
        'case:coverage': '关键事件、人物和未知项覆盖充分',
        'case:query': '查询条件可由可见信息提出，错人错时不返回材料',
        'case:alternatives': '替代支持集、反证与未决条件已逐项核对',
        'case:atomicity': '事件为原子事实，明确区分陈述与已证实事实',
        'case:legal_unknown': '未审查的法律要件保留未知，不照抄裁判标签'}
    for e in case.evidence:
        for key, label in [('source', '来源与原文偏移'), ('boundary', '无裁判结论泄漏'),
                           ('meaning', '否定、时间、金额和行为主体未被改写')]:
            checks[f'evidence:{e.id}:{key}'] = e.id+'：'+label
    for c in case.claims:
        checks[f'claim:{c.id}:support'] = c.id+'：引用支持整个结论或明确保持未知'
        checks[f'claim:{c.id}:state'] = c.id+'：移除决定性证据后不再认定，反证与替代路径处理正确'
    return checks


def material(cid):
    folder, cases = bundle()
    case = next((c for c in cases if c.case_id == cid), None)
    if case is None: raise HTTPException(404, '未列入当前训练复核批次。')
    try:
        sources = json.loads((folder/'sources.private.json').read_text())
        row = sources[cid]
        if digest(row) != case.source_row_sha256: raise ValueError('Source changed')
        from scripts.export_review_bundle import verify
        verify(row, {'raw_record': row['raw_record']})
        audits = [json.loads(p.read_text()) for p in sorted((folder/'local_audits'/cid).glob('*.json'))]
        disposition_path = folder/'assistant_audit_disposition.private.json'
        dispositions = json.loads(disposition_path.read_text()) if disposition_path.exists() else {}
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, '来源绑定失效，请重建材料。') from exc
    content = {'case': case.model_dump(), 'source': row['raw_record'], 'source_metadata': row['source'],
               'legacy_annotation': row['five_dimensions'], 'sentence_spans': row['sentence_spans'],
               'local_audits': audits, 'assistant_audit_disposition': dispositions.get(cid),
               'solvability': reachability(case),
               'required_checks': required_checks(case)}
    return content | {'review_binding': digest(content)}


class HumanReceipt(StrictModel):
    review_binding: str
    decision: Literal['approve_curation', 'reject']
    checks: dict[str, StrictBool]
    notes: str = Field(min_length=10, max_length=10000)
    human_minutes: float = Field(gt=0, le=10080, allow_inf_nan=False)
    reviewer_level: Literal['single_nonexpert'] = 'single_nonexpert'
    personally_reviewed: Literal[True]


def current_receipt(cid, content):
    paths = sorted((RECEIPTS/digest(cid)).glob('*.json'), reverse=True)
    if not paths: return {'status': 'not_reviewed', 'formal_benchmark_eligible': False}
    record = json.loads(paths[0].read_text())
    valid = record.get('review_binding') == content['review_binding']
    return {'status': record.get('decision') if valid else 'stale',
            'reviewed_at': record.get('reviewed_at'), 'formal_benchmark_eligible': False}


@router.get('/items')
def items(user=Depends(enabled)):
    _, cases = bundle()
    return [{'case_id': c.case_id, 'split': 'train', 'evidence_count': len(c.evidence)} for c in cases]


@router.get('/items/{cid}')
def item(cid: str, user=Depends(enabled)):
    content = material(cid)
    return content | {'receipt': current_receipt(cid, content)}


@router.post('/items/{cid}/review')
def review(cid: str, value: HumanReceipt, user=Depends(enabled)):
    content = material(cid)
    if value.review_binding != content['review_binding']:
        raise HTTPException(409, '来源、标注或机器审查已变化，请刷新后重新核对。')
    if set(value.checks) != set(content['required_checks']):
        raise HTTPException(422, '逐项核对清单不完整。')
    if value.decision == 'approve_curation' and not all(value.checks.values()):
        raise HTTPException(422, '尚有未核对项目，不能通过。')
    now = datetime.now(timezone.utc)
    record = value.model_dump() | {'case_id': cid, 'reviewer_user_id': user['user_id'],
        'reviewed_at': now.isoformat(), 'version': 'human-curation-receipt-v1',
        'formal_benchmark_eligible': False}
    folder = RECEIPTS/digest(cid); folder.mkdir(parents=True, exist_ok=True)
    path = folder/(now.strftime('%Y%m%dT%H%M%S%f')+'-'+uuid.uuid4().hex+'.json')
    with path.open('x') as stream: json.dump(record, stream, ensure_ascii=False, indent=2)
    return {'saved': True, 'decision': value.decision, 'formal_benchmark_eligible': False}
