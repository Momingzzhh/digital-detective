#!/usr/bin/env python3
"""Bounded local 14B proposals/audits, train-only; never grants human approval."""
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Literal
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pydantic import Field
from app.models import StrictModel
from app.environment_models import PrivateCase
from app.experiment_manifest import digest, file_digest
from app.curation_identity import identity_errors
from scripts.build_interactive_cases import save, OUT


class AtomicProposal(StrictModel):
    claim: str
    source_attribution: str = Field(min_length=2)
    epistemic_status: Literal['source_statement_unverified']
    identity_sentence_ids: list[str] = Field(min_length=1)
    sentence_ids: list[str] = Field(min_length=1)
    counter_sentence_ids: list[str]
    unknowns: list[str]


class AuditIssue(StrictModel):
    category: Literal['source_boundary', 'actor', 'event', 'support', 'state', 'query']
    description: str
    sentence_ids: list[str]


class CurationAudit(StrictModel):
    decision: Literal['needs_revision', 'no_issue_found']
    issues: list[AuditIssue]
    atomic_event_proposals: list[AtomicProposal] = Field(max_length=1)
    unresolved: list[str]


PROMPT = '''你是案件训练材料审查员。只根据给定证据和句子核对，不使用裁判结果。
核对来源边界、否定词、时间、金额、行为主体、来源陈述与事实的区别，检查引用是否支持整个结论。
将复合事件改为少量关键的原子事件候选，每项仅一个动作或事实，明确“某人称”而非当作确定事实。
不得虚构角色、矛盾、替代支持集或法律要件，不要求三层推理。反证必须真实，未知可留空或写明未决。
查询目前仅按通用材料类型；不许假装已有具体人物/时间索引。无问题不代表人工通过。
必须返回符合schema的JSON；引用仅可用给定sentence_ids。'''
PROMPT += '''\n每个候选须填写source_attribution（如“陈某某供述”），epistemic_status固定为source_statement_unverified。
identity_sentence_ids必须引用说明陈述者、或将“我/男子/老板”对应具体名字的原文句子；不能只引用含代词的动作句。
claim本身也须保留来源限定。身份对应不明确则保留“该陈述中的男子/老板”，并在unknowns注明，不凭全案答案填名字。
只生成最多一个最重要事件以便人工核查，其余未覆盖内容写入unresolved，不宣称完整覆盖。
identity_reference_candidates仅列文本直接含该名字的句子，必须检查这些引用能否支持身份对应，不能只引含代词的动作句。'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 7 or not 512 <= args.max_tokens <= 4096 or not 1 <= args.attempt <= 2:
        parser.error('Bounded pilot only, at most two explicitly numbered attempts')
    pointer = json.loads((OUT/'current.json').read_text()); folder = OUT/pointer['directory']
    raw = json.loads((folder/'cases.private.json').read_text())
    sources = json.loads((folder/'sources.private.json').read_text())
    cases = [PrivateCase.model_validate(c) for c in raw]
    for case in cases:
        if digest(sources[case.case_id]) != case.source_row_sha256: raise ValueError('Source changed')
    model = ROOT/'models/qwen3-14b-mlx-4bit'
    os.environ.update(LOCAL_MODEL_PATH=str(model), LOCAL_ADAPTER_PATH='', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    from scripts.annotation_lock import acquire_annotation_lock
    acquire_annotation_lock()
    from transformers import AutoTokenizer
    from scripts.annotate_five_dimensions_batch import bounded_generation
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    schema = CurationAudit.model_json_schema()
    for case in cases[:args.limit]:
        if (ROOT/'runtime/research_v2.stop').exists(): break
        row = sources[case.case_id]
        visible_ids = {e.id for e in case.evidence}
        sentences = [s for s in row['sentence_spans'] if s['evidence_id'] in visible_ids]
        material = {'evidence': [{'id': e.id, 'text': e.text} for e in case.evidence],
            'identity_reference_candidates': {name: [s['id'] for s in sentences if name in s['text']]
                for name in row['five_dimensions']['candidate_names']},
            'sentence_spans': sentences,
            'claims': [c.model_dump(exclude={'support_citations'}) for c in case.claims]}
        request = [{'role': 'system', 'content': PROMPT},
                   {'role': 'user', 'content': json.dumps(material, ensure_ascii=False)}]
        identity = {'case_binding': digest(case.model_dump()), 'request': digest(request),
            'schema': digest(schema), 'model_manifest': file_digest(model/'local_manifest.json'),
            'code': {p: file_digest(ROOT/p) for p in ('scripts/audit_interactive_cases.py',
                      'scripts/annotate_five_dimensions_batch.py', 'app/local_model.py', 'app/local_context.py',
                      'app/curation_identity.py')},
            'max_tokens': args.max_tokens, 'context_limit': 16384, 'adapter': None, 'attempt': args.attempt}
        key = digest(identity)
        target = folder/'local_audits'/case.case_id/(key+'.json')
        if target.exists():
            previous = json.loads(target.read_text())
            if previous['status'] != 'started': continue
            raise ValueError('Interrupted attempt retained; explicit new attempt required')
        result = {'identity': identity, 'status': 'started', 'human_approval': False,
                  'started_at': datetime.now(timezone.utc).isoformat()}
        save(target, result)
        print(json.dumps({'case': case.case_id, 'status': 'started'}), flush=True)
        try:
            generated, stats, reserved = bounded_generation(tokenizer, [(request, CurationAudit)],
                args.max_tokens, 16384)
            result['stats'] = stats
            if reserved: result.update(status='context_reserved', error=str(reserved[0]))
            else:
                text = generated[0][1]; result['raw_output'] = text
                parsed = CurationAudit.model_validate_json(text)
                ids = {s['id'] for s in sentences}
                for item in parsed.atomic_event_proposals:
                    if set(item.sentence_ids+item.counter_sentence_ids+item.identity_sentence_ids)-ids:
                        raise ValueError('Unknown sentence')
                    if set(item.sentence_ids)&set(item.counter_sentence_ids): raise ValueError('Same support and counter')
                for issue in parsed.issues:
                    if set(issue.sentence_ids)-ids: raise ValueError('Unknown issue reference')
                if parsed.decision == 'no_issue_found' and parsed.issues: raise ValueError('Contradictory audit')
                problems = identity_errors([p.model_dump() for p in parsed.atomic_event_proposals],
                    sentences, row['five_dimensions']['candidate_names'])
                result.update(status='identity_support_held' if problems else 'machine_audited_unverified',
                              audit=parsed.model_dump(), identity_screen_errors=problems,
                              identity_screen_limitations='Explicit known actor names only; aliases, attribution and entailment still require source review.')
        except (Exception, KeyboardInterrupt) as exc:
            result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                          error=type(exc).__name__+': '+str(exc))
            save(target, result)
            if isinstance(exc, KeyboardInterrupt): break
        save(target, result)
        print(json.dumps({'case': case.case_id, 'status': result['status']}), flush=True)


if __name__ == '__main__': main()
