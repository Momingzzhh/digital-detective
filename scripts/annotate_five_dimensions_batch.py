#!/usr/bin/env python3
"""Local constrained minibatches, without model-based case selection."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.five_dimensions import FiveDimensions, VERSION, fingerprint, validate_annotation
from app.local_model import complete_batch
from app.local_model import model_path
from app.config import settings
from scripts.annotate_five_dimensions import PROMPT, normalize_references
from scripts.compact_annotation import INSTRUCTIONS, schema_for, payload, canonical
from pydantic import ValidationError
from scripts.annotation_lock import acquire_annotation_lock
from app.annotation_quality import (SemanticAudit, AUDIT_PROMPT, AUDIT_VERSION, audit_binding,
    quality_passed, obvious_quality_errors)

QUOTAS = {'train': 420, 'dev': 90, 'test': 90}
PROTOCOL = 'compact-json-v6-local-curator'
RESULT = ROOT/'runtime/five_dimensions_annotation_result.json'
PROGRESS = ROOT/'runtime/five_dimensions_annotation_progress.json'
FAILURE = ROOT/'runtime/five_dimensions_annotation_failure.json'
REPAIR_POLICY = 'one-source-bound-semantic-repair-v1'
REPAIR_PROMPT = '''上次候选未通过核验。请对照本案原文审查反馈并修正标注；反馈也可能误读原文，不能为通过检查而盲从。
纯证人、纯被害人、正常购货者、仅在场者不自动成为犯罪协助者，不能用“无法确认”收纳所有无关人物。删除原本不存在的矛盾；不知情和互补分工不是矛盾。没有别名可留空，真实但未解决的矛盾应如实保留。
每个角色和事件须由引用句子支持。依赖链须从较小的原始观察出发，经其他来源关联再核查行为；不能先给全案结论后用时间顺序罗列细节。无法形成必要的三层关联或覆盖不足时如实保留不足，不得捏造依赖或强行将完整性改为true。只允许这一次来源约束的修订，仍须重新经过全部校验。'''


def binding(row):
    return fingerprint({'version': VERSION, 'prompt': PROMPT, 'compact_prompt':INSTRUCTIONS,'protocol':PROTOCOL,
        'audit_version':AUDIT_VERSION,'audit_prompt':AUDIT_PROMPT,
        'annotator_manifest':json.loads((model_path()/'local_manifest.json').read_text()),
        'schema': FiveDimensions.model_json_schema(), 'source': row['source_binding']})


def material(row):
    offset = min(e['start'] for e in row['evidence'])
    return {'required_evidence_ids': [e['id'] for e in row['evidence']],
        'evidence': [{k: e[k] for k in ('id','kind','text')} for e in row['evidence']],
        'sentences': [{k: s[k] for k in ('id','evidence_id','start','end')} for s in row['sentence_spans']],
        'hidden_source_reference': {'adjudicated_fact_before_evidence': row['source_fact_excerpt'][:offset],
                                     'recorded_outcome': row['hidden_gold']}}


def save(path, value):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def retry_checkpoint(old, args):
    if old['status'] == 'machine_annotated_unverified': return False
    older = old.get('generation_protocol') != PROTOCOL
    if older and (args.retry_rejected or old.get('generation_protocol') == 'compact-json-v1-eos-stop'):
        return True
    if (old['status']=='semantic_quality_rejected' and getattr(args,'repair_rejected',False)
        and old.get('semantic_repair_attempts',0)<1
        and (old.get('raw_compact_annotation') or old.get('raw_annotation'))):return True
    errors = old['status'] == 'local_annotation_error' or (
        not old.get('raw_annotation') and 'validation error' in old.get('error',''))
    attempts = old.get('protocol_attempts', 1) if not older else 0
    return bool(args.retry_errors and errors and attempts < args.max_case_attempts)


def main(args):
    acquire_annotation_lock()
    agent_path=str(model_path())
    if args.model_profile=='curator':
        os.environ['LOCAL_MODEL_PATH']=str(ROOT/'models/qwen3-14b-mlx-4bit')
    os.environ['LOCAL_ADAPTER_PATH'] = ''  # Fixed base annotator, never the agent under evaluation.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path()), local_files_only=True, trust_remote_code=False)
    context_limit = int(settings().get('LOCAL_CONTEXT_TOKENS', '16384'))
    output = ROOT / 'data/review/five_dimensions_v2'; output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (ROOT / 'data/processed/five_dimensions_v2/candidates.private.jsonl').open()]
    if args.split != 'all': rows = [r for r in rows if r['split'] == args.split]
    counts = Counter(); pending = []; existing = Counter()
    attempts = {}; repair_context = {}
    # Source metadata priority is independent of agent predictions and scores.
    rows.sort(key=lambda r: len(r['hidden_gold'].get('adjudicated_perpetrators',[])) < 2)
    for row in rows:
        path = output / (row['case_id'] + '.json'); expected = binding(row)
        if path.exists():
            old = json.loads(path.read_text())
            attempts[row['case_id']] = old.get('protocol_attempts',1) if old.get('generation_protocol') == PROTOCOL else 0
            if old.get('annotation_binding') == expected:
                existing[old['status']] += 1
                if not retry_checkpoint(old,args): continue
                if old['status']=='semantic_quality_rejected' and (old.get('raw_compact_annotation') or old.get('raw_annotation')):
                    repair_context[row['case_id']]={'previous_annotation':old.get('raw_compact_annotation') or old['raw_annotation'],
                        'feedback':old.get('semantic_audit',{'error':old.get('error','')}),
                        'previous_result_binding':fingerprint(old)}
                history = output / 'history'; history.mkdir(exist_ok=True)
                archived = history / (row['case_id'] + '.' + fingerprint(old)[:16] + '.json')
                if not archived.exists(): archived.write_bytes(path.read_bytes())
            else:
                history = output / 'history'; history.mkdir(exist_ok=True)
                archived = history / (row['case_id'] + '.' + old['annotation_binding'][:12] + '.json')
                if not archived.exists(): archived.write_bytes(path.read_bytes())
        pending.append(row)
    def refresh_counts():
        subprocess.run([sys.executable,str(ROOT/'scripts/assemble_five_dimensions.py')],check=True,
                       stdout=subprocess.DEVNULL,env=os.environ|{'LOCAL_MODEL_PATH':agent_path})
        report = json.loads((ROOT/'data/five_dimensions_report.json').read_text())
        return Counter(report['current_split_counts'])
    # Export, rather than raw annotation counts, also accounts for overlength
    # training cases and explicit source-bound human rejections.
    counts = refresh_counts()
    (ROOT / 'runtime').mkdir(exist_ok=True)
    (ROOT / 'runtime/five_dimensions_annotation.pid').write_text(str(os.getpid()))
    processed = 0
    RESULT.unlink(missing_ok=True)
    while pending and processed < args.limit and any(counts[s] < n for s,n in QUOTAS.items()):
        if (ROOT / 'runtime/stop_five_dimensions_annotation').exists(): break
        candidates = [r for r in pending if counts[r['split']] < QUOTAS[r['split']]]
        batch = candidates[:min(args.batch_size, args.limit - processed)]
        if not batch: break
        ids = {r['case_id'] for r in batch}; pending = [r for r in pending if r['case_id'] not in ids]
        progress_base = {'pid':os.getpid(),'case_ids':sorted(ids),'processed_before_batch':processed,
                         'batch_size':len(batch),'started_at':datetime.now(timezone.utc).isoformat()}
        last_progress = 0.0
        def progress(event, force=False):
            nonlocal last_progress
            now = time.monotonic()
            if force or now-last_progress >= 5:
                save(PROGRESS,progress_base | event | {'updated_at':datetime.now(timezone.utc).isoformat()})
                last_progress = now
        progress({'stage':'preparing'},True)
        schemas = [schema_for(row) for row in batch]; requests = []; valid_rows = []; valid_schemas = []
        reserved = 0
        for row, schema in zip(batch, schemas):
            system = INSTRUCTIONS + '\n只输出JSON数据实例。Schema：' + json.dumps(schema.model_json_schema(), ensure_ascii=False)
            content=payload(row)
            if row['case_id'] in repair_context:
                system+='\n'+REPAIR_PROMPT
                content['source_bound_revision']=repair_context[row['case_id']]
            messages = [{'role':'system','content':system}, {'role':'user','content':json.dumps(content,ensure_ascii=False)}]
            prompt_tokens = len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False))
            if prompt_tokens + args.max_tokens > context_limit:
                save(output/(row['case_id']+'.json'), {'case_id':row['case_id'],'group_id':row['group_id'],
                    'split':row['split'],'annotation_binding':binding(row),'source_binding':row['source_binding'],
                    'generation_protocol':PROTOCOL,'actual_prompt_sha256':fingerprint(messages),
                    'human_reviewed':False,'status':'local_context_reserved_for_later',
                    'prompt_tokens':prompt_tokens,'generation_budget':args.max_tokens,'context_limit':context_limit})
                reserved += 1
                continue
            valid_rows.append(row); valid_schemas.append(schema); requests.append((messages,schema))
        begin = time.monotonic()
        try:
            texts, stats = complete_batch(requests, args.max_tokens, progress=progress) if requests else ([],{})
        except Exception as exc:
            # A driver failure is not a case annotation. Preserve all prior
            # checkpoints and let the supervisor restart at a smaller batch.
            save(FAILURE,progress_base | {'failed_at':datetime.now(timezone.utc).isoformat(),'error':str(exc)[:500]})
            raise
        results = []
        for index,(row, schema, text) in enumerate(zip(valid_rows, valid_schemas, texts)):
            result = {'case_id':row['case_id'],'group_id':row['group_id'],'split':row['split'],
                'annotation_binding':binding(row),'source_binding':row['source_binding'],
                'annotator':'local-'+args.model_profile+'-constrained-minibatch','human_reviewed':False,
                'annotator_manifest_binding':fingerprint(json.loads((model_path()/'local_manifest.json').read_text())),
                'generation_protocol':PROTOCOL,
                'protocol_attempts':attempts.get(row['case_id'],0)+1,
                'semantic_repair_attempts':int(row['case_id'] in repair_context),
                'actual_prompt_sha256':fingerprint(requests[index][0]),
                'generated_at':datetime.now(timezone.utc).isoformat()}
            if row['case_id'] in repair_context:
                result.update(repair_policy=REPAIR_POLICY,repair_prompt_sha256=fingerprint(REPAIR_PROMPT),
                    previous_result_binding=repair_context[row['case_id']]['previous_result_binding'])
            try:
                raw = schema.model_validate_json(text).model_dump(); result['raw_compact_annotation'] = raw
                data = canonical(raw, row); result['raw_annotation'] = data
                normalized, repairs = normalize_references(data, row['sentence_spans'])
                result['identifier_repairs'] = repairs
                ann = validate_annotation(row, normalized, row['sentence_spans'])
                result.update(status='machine_annotated_unverified',five_dimensions=ann)
                errors=obvious_quality_errors(ann)
                if errors:result.update(status='semantic_quality_rejected',error='; '.join(errors))
            except ValidationError as exc:
                result.update(status='local_annotation_error',error=str(exc)[:300],raw_output_text=text)
            except ValueError as exc: result.update(status='complexity_or_reference_rejected',error=str(exc)[:300])
            except Exception as exc: result.update(status='local_annotation_error',error=str(exc)[:300])
            results.append((row,result))
        audit_failure=None
        audit_pairs=[(row,r) for row,r in results if r['status']=='machine_annotated_unverified']
        if audit_pairs:
            audit_requests=[([{'role':'system','content':AUDIT_PROMPT+'\nSchema：'+json.dumps(SemanticAudit.model_json_schema(),ensure_ascii=False)},
                {'role':'user','content':json.dumps({'source':payload(row),'candidate':r['five_dimensions']},ensure_ascii=False)}],SemanticAudit)
                for row,r in audit_pairs]
            def audit_progress(event):progress(event | {'phase':'semantic_audit'})
            try:
                audit_texts,audit_stats=complete_batch(audit_requests,1400,progress=audit_progress)
                for (row,r),raw_audit in zip(audit_pairs,audit_texts):
                    r['semantic_audit_raw']=raw_audit
                    try:
                        r['semantic_audit']=SemanticAudit.model_validate_json(raw_audit).model_dump()
                        r['semantic_audit_binding']=audit_binding(row,r['five_dimensions'])
                        if not quality_passed(r,row):r.update(status='semantic_quality_rejected',error='Independent local semantic screen failed')
                    except ValueError as exc:r.update(status='local_annotation_error',error=str(exc)[:300])
                stats['semantic_audit']=audit_stats
            except Exception as exc:
                for row,r in audit_pairs:r.update(status='local_annotation_error',error='Semantic screen incomplete: '+str(exc)[:200])
                if 'insufficient memory' in str(exc).lower():audit_failure=exc
        for row,result in results:
            save(output / (row['case_id'] + '.json'), result)
            existing[result['status']] += 1
        if audit_failure:
            save(FAILURE,progress_base | {'failed_at':datetime.now(timezone.utc).isoformat(),'error':str(audit_failure)[:500]})
            raise audit_failure
        processed += len(batch)
        counts = refresh_counts()
        if args.model_profile=='curator' and stats.get('peak_memory'):
            resource_path=ROOT/'data/local_resource_probe.json'
            resource=json.loads(resource_path.read_text()) if resource_path.exists() else {}
            resource.update(curator_14b_resource_probe_completed=True,
                curator_note='At least one actual generation batch completed; memory evidence does not establish semantic quality or accuracy.',
                curator_14b_last_batch={'recorded_at':datetime.now(timezone.utc).isoformat(),
                    'batch_size':len(batch),'protocol':PROTOCOL,'generation_stats':stats,
                    'status_counts':dict(Counter(r['status'] for _,r in results))})
            save(resource_path,resource)
            current=json.loads((ROOT/'data/five_dimensions_report.json').read_text())
            current.update(annotator_preparation='local_weights_ready',
                annotator_manifest_binding=fingerprint(json.loads((model_path()/'local_manifest.json').read_text())))
            save(ROOT/'data/five_dimensions_report.json',current)
        print(json.dumps({'processed':processed,'seconds':round(time.monotonic()-begin,2),
            'accepted_by_split':dict(counts),'attempt_status_counts':dict(existing),
            'reserved_over_context_this_batch':reserved,'generation_stats':stats}), flush=True)
        with (ROOT/'runtime/five_dimensions_annotation_batches.jsonl').open('a') as stream:
            stream.write(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),
                'protocol':PROTOCOL,'batch_size':len(batch),'seconds':round(time.monotonic()-begin,2),
                'generation_stats':stats,'attempt_status_counts':dict(Counter(r['status'] for _,r in results))})+'\n')
        progress({'stage':'saved','accepted_by_split':dict(counts)},True)
    result = {'finished':True,'pid':os.getpid(),'processed':processed,'accepted_by_split':dict(counts),
        'eligible_pending_after':sum(counts[r['split']] < QUOTAS[r['split']] for r in pending),
        'finished_at':datetime.now(timezone.utc).isoformat()}
    save(RESULT,result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,choices=[1,2,4,8],default=1)
    parser.add_argument('--model-profile',choices=['agent','curator'],default='curator')
    parser.add_argument('--max-tokens',type=int,default=4000)
    parser.add_argument('--retry-errors',action='store_true')
    parser.add_argument('--max-case-attempts',type=int,default=2)
    parser.add_argument('--split',choices=['all','train','dev','test'],default='all')
    parser.add_argument('--repair-rejected',action='store_true',help='One source-bound revision of a semantic failure; original output is archived, all gates rerun')
    parser.add_argument('--retry-rejected',action='store_true',help='Retry failures from an older protocol once; never changes splits or validity checks')
    main(parser.parse_args())
