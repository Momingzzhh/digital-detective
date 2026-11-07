#!/usr/bin/env python3
"""Freeze training-only cohorts and audit new labels for small-batch admission."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.annotation_quality import (AUDIT_PROMPT, SemanticAudit, audit_binding,
                                    obvious_quality_errors, quality_passed,
                                    semantic_schema, audit_structure_errors)
from app.complexity_review import complexity_exclusion, source_review_hold
from app.auxiliary_training import bound_file
from app.five_dimensions import fingerprint, validate_small_batch_annotation
from app.small_batch_training import POOL, require_revision_identity
from app.source_policy import require_clean_evidence
from scripts.annotation_lock import acquire_annotation_lock
from scripts.compact_annotation import INSTRUCTIONS, schema_for, payload, canonical

PROTOCOL = 'small-batch-source-annotation-v1'
PROMPT = INSTRUCTIONS.replace('不足三层会被隔离', '至少两层才可进入本轮小批量训练，优先真实三层') + '''
本轮优先真实三层，确有必要关联但仅两层时保留实际两层。不足两层或材料不完整应如实保留，不许为了启动训练捏造前提。
原文直接支持的结论不需要绕远路；不得故意少引直接支持句子以制造推理依赖。自动通过也仍为待人工复核。
claim和role简洁表述；争议事实注明陈述来源和未决状态。'''
CODE = ('scripts/annotate_small_batch.py', 'scripts/compact_annotation.py',
        'scripts/annotate_five_dimensions_batch.py', 'app/five_dimensions.py',
        'app/source_policy.py', 'app/annotation_quality.py', 'app/local_model.py',
        'app/local_context.py', 'scripts/prepare_training_source_revision.py',
        'scripts/build_v3.py', 'app/complexity_review.py', 'app/small_batch_training.py',
        'app/field_semantic.py')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def create_cohort(limit):
    if not 1 <= limit <= 10:
        raise ValueError('Freeze one to ten cases per cohort')
    seen = set()
    for folder in ('five_dimensions_v2', 'training_source_screen', 'small_batch_annotations'):
        for path in (ROOT / 'data/review' / folder).rglob('*.json'):
            # Names suffice to exclude prior work; no heldout semantics read.
            seen.add(path.stem)
    rows = []; counts = Counter()
    for row in map(json.loads, (ROOT / POOL).open()):
        if row['split'] != 'train':
            continue
        if row['case_id'] in seen:
            counts['previously_processed'] += 1; continue
        if complexity_exclusion(row) or source_review_hold(row):
            counts['signed_feedback_hold'] += 1; continue
        try:
            require_clean_evidence(row['evidence'])
        except ValueError:
            counts['source_boundary_rejected'] += 1; continue
        chars = sum(len(e['text']) for e in row['evidence'])
        if not 1800 <= chars <= 6000:
            counts['outside_initial_source_length_band'] += 1; continue
        rows.append(row)
    rows.sort(key=lambda row: fingerprint(row['group_id']))
    rows = rows[:limit]
    if not rows:
        raise ValueError('No unprocessed source-clean training candidates in this band')
    cases = [{k: r[k] for k in ('case_id', 'group_id', 'split', 'source_binding')} for r in rows]
    value = {'protocol': PROTOCOL, 'selection': 'fixed group hash; 1800..6000 visible characters; unprocessed train only',
             'cases': cases, 'source_pool_sha256': hashlib.sha256((ROOT / POOL).read_bytes()).hexdigest(),
             'excluded_counts': dict(counts)}
    value['cohort_binding'] = fingerprint(value)
    path = ROOT / 'data/review/small_batch_cohorts' / value['cohort_binding'][:16] / 'cohort.private.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Frozen cohort differs')
    save(path, value)
    return path, value


def load_cohort(path):
    value = json.loads(path.read_text())
    if (value.get('protocol') != PROTOCOL or value.get('cohort_binding') != fingerprint(
        {k: v for k, v in value.items() if k != 'cohort_binding'})):
        raise ValueError('Frozen small-batch cohort binding changed')
    if type(value.get('allow_single_actor_without_conflict', False)) is not bool:
        raise ValueError('Single-actor cohort scope must be an explicit boolean')
    if hashlib.sha256((ROOT / POOL).read_bytes()).hexdigest() != value['source_pool_sha256']:
        raise ValueError('Frozen source pool changed')
    for entry in value.get('previous_protocol_results', []):
        previous = json.loads(bound_file(ROOT, entry))
        if not any(previous.get('case_id') == case['case_id'] and
                   previous.get('source_binding') == case['source_binding'] for case in value['cases']):
            raise ValueError('Previous protocol result belongs to another cohort source')
    pool = {r['case_id']: r for r in map(json.loads, (ROOT / POOL).open()) if r['split'] == 'train'}
    if value.get('source_revision_manifest'):
        from scripts.prepare_training_source_revision import load_revision
        for row in load_revision(ROOT, value['source_revision_manifest']):
            require_revision_identity(pool.get(row['case_id']), row)
            pool[row['case_id']] = row
    rows = []; groups = set()
    for entry in value['cases']:
        row = pool.get(entry['case_id'])
        if not row or any(entry[k] != row[k] for k in ('split', 'group_id', 'source_binding')):
            raise ValueError('Small-batch cohort must match original training identities')
        if row['group_id'] in groups:
            raise ValueError('Repeated group in small-batch cohort')
        if complexity_exclusion(row) or source_review_hold(row):
            raise ValueError('Signed feedback blocks this unreviewed source')
        require_clean_evidence(row['evidence'])
        groups.add(row['group_id']); rows.append(row)
    return value, rows


def load_draft(row, path, *, allow_single_actor_without_conflict=False):
    path = (ROOT / path).resolve()
    if not path.is_relative_to((ROOT / 'data/review').resolve()):
        raise ValueError('Assistant drafts must remain in the private review directory')
    data = path.read_bytes(); value = json.loads(data)
    if (value.get('author') != 'assistant_source_reader' or value.get('human_reviewed') is not False or
        value.get('formal_approval') is not False or
        any(value.get(k) != row[k] for k in ('case_id', 'group_id', 'split', 'source_binding'))):
        raise ValueError('Assistant draft must match its exact training source without claiming human approval')
    annotation = validate_small_batch_annotation(row, value['raw_annotation'], row['sentence_spans'],
        allow_single_actor_without_conflict=allow_single_actor_without_conflict)
    return annotation, {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest()}


def process(args):
    cohort, rows = load_cohort((ROOT / args.cohort).resolve())
    from app import field_semantic
    audit_protocol = cohort.get('semantic_audit_protocol')
    if audit_protocol not in (None, field_semantic.PROTOCOL):
        raise ValueError('Unknown semantic audit protocol')
    allow_single = cohort.get('allow_single_actor_without_conflict', False)
    if type(allow_single) is not bool:
        raise ValueError('Single-actor cohort scope must be an explicit boolean')
    if args.case_id:
        rows = [r for r in rows if r['case_id'] == args.case_id]
        if not rows:
            raise ValueError('Case is outside the frozen training cohort')
    draft = None; draft_entry = None
    if getattr(args, 'audit_draft', None):
        if len(rows) != 1:
            raise ValueError('An assistant draft must target exactly one frozen training source')
        draft, draft_entry = load_draft(rows[0], args.audit_draft,
            allow_single_actor_without_conflict=allow_single)
    out = ROOT / 'data/review/small_batch_annotations' / cohort['cohort_binding'][:16]
    os.environ.update(LOCAL_MODEL_PATH=str(ROOT / 'models/qwen3-14b-mlx-4bit'), LOCAL_ADAPTER_PATH='',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    acquire_annotation_lock()
    from app.local_model import model_path
    from app.config import settings
    from transformers import AutoTokenizer
    from scripts.annotate_five_dimensions_batch import bounded_generation
    tokenizer = AutoTokenizer.from_pretrained(str(model_path()), local_files_only=True, trust_remote_code=False)
    model_binding = fingerprint(json.loads((model_path() / 'local_manifest.json').read_text()))
    code_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in CODE}
    code_dir = out / 'provenance' / fingerprint(code_hashes)[:16]
    for name in CODE:
        target = code_dir / name
        content = (ROOT / name).read_bytes()
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Immutable annotation code snapshot changed')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
    context_limit = int(settings().get('LOCAL_CONTEXT_TOKENS', '16384'))
    processed = 0
    for row in rows:
        target = out / (row['case_id'] + '.json')
        request_binding = fingerprint({'source': row['source_binding'], 'prompt': PROMPT,
                                       'audit_prompt': AUDIT_PROMPT, 'code': code_hashes,
                                       'model': model_binding, 'assistant_draft': draft_entry,
                                       'semantic_audit_protocol': audit_protocol})
        previous_runtime = None
        previous_bad_audit = None
        if target.exists():
            previous = json.loads(target.read_text())
            runtime_only = (not previous.get('raw_output') and not previous.get('generation_stats') and
                            (previous['status'] in ('local_runtime_error', 'local_annotation_error') or
                             previous.get('error', '').startswith('MLX环境或Metal GPU不可用')))
            if args.retry_runtime_errors and runtime_only and not previous.get('previous_runtime_failure'):
                archived = out / 'history' / (row['case_id'] + '.' + fingerprint(previous)[:16] + '.json')
                if not archived.exists():
                    save(archived, previous)
                previous_runtime = {'path': str(archived.relative_to(ROOT)),
                                    'sha256': hashlib.sha256(archived.read_bytes()).hexdigest()}
            elif (getattr(args, 'repair_audit_format', False) and previous.get('semantic_audit') and
                  previous.get('five_dimensions') and not previous.get('audit_format_repair_attempts') and
                  audit_structure_errors(previous['semantic_audit'], previous['five_dimensions'])):
                archived = out / 'history' / (row['case_id'] + '.' + fingerprint(previous)[:16] + '.json')
                if not archived.exists():
                    save(archived, previous)
                previous_bad_audit = previous
                previous_bad_audit['archived_result'] = {'path': str(archived.relative_to(ROOT)),
                    'sha256': hashlib.sha256(archived.read_bytes()).hexdigest()}
            elif previous.get('source_binding') != row['source_binding']:
                raise ValueError('Existing annotation belongs to a different source version')
            elif previous['request_binding'] != request_binding and not previous.get('finished_at'):
                raise ValueError('Existing annotation uses different inputs; freeze a new cohort')
            else:
                continue
        if processed >= args.max_cases:
            break
        result = {k: row[k] for k in ('case_id', 'group_id', 'split', 'source_binding')}
        result.update(protocol=PROTOCOL, cohort_binding=cohort['cohort_binding'],
                      request_binding=request_binding, model_binding=model_binding,
                      code_file_sha256=code_hashes, code_snapshot_dir=str(code_dir.relative_to(ROOT)),
                      started_at=datetime.now(timezone.utc).isoformat(),
                      human_reviewed=False, formal_approval=False,
                      status='annotation_pending', annotator='local-14b-base-machine-only')
        if cohort.get('source_revision_manifest') and row.get('training_source_revision'):
            result['source_revision_manifest'] = cohort['source_revision_manifest']
        if previous_runtime:
            result['previous_runtime_failure'] = previous_runtime
            for key in ('audit_format_repair_attempts', 'previous_invalid_audit_result'):
                if key in previous:
                    result[key] = previous[key]
        if draft_entry:
            result.update(annotator='assistant_source_reader', assistant_draft_file=draft_entry,
                          generation_kind='assistant_prepared_then_separate_local_semantic_audit')
        if audit_protocol:
            result['semantic_audit_protocol'] = audit_protocol
            result['previous_protocol_results'] = cohort.get('previous_protocol_results', [])
        if previous_bad_audit:
            result.update(audit_format_repair_attempts=1,
                          previous_invalid_audit_result=previous_bad_audit['archived_result'])
        last = 0
        def progress(event, phase='annotation'):
            nonlocal last
            now = time.monotonic()
            if now - last > 5:
                save(ROOT / 'runtime/small_batch_annotation_progress.json',
                     {'case_id': row['case_id'], 'phase': phase, **event,
                      'updated_at': datetime.now(timezone.utc).isoformat()})
                last = now
        try:
            annotation = previous_bad_audit['five_dimensions'] if previous_bad_audit else draft
            reserved = []
            if annotation is None:
                schema = schema_for(row)
                messages = [{'role': 'system', 'content': PROMPT + '\nSchema：' + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
                            {'role': 'user', 'content': json.dumps(payload(row), ensure_ascii=False)}]
                result['annotation_prompt_binding'] = fingerprint(messages)
                generated, stats, reserved = bounded_generation(tokenizer, [(messages, schema)], args.max_tokens, context_limit, progress)
                result['generation_stats'] = stats
                if not reserved:
                    _, raw, count = generated[0]
                    result.update(raw_output=raw, prompt_tokens=count)
                    compact = schema.model_validate_json(raw).model_dump()
                    result['raw_compact_annotation'] = compact
                    annotation = canonical(compact, row)
            if reserved:
                result['status'] = 'annotation_context_reserved'
            else:
                result['raw_annotation'] = annotation
                from app.five_dimensions import FiveDimensions
                annotation = {k: annotation[k] for k in FiveDimensions.model_fields}
                annotation = validate_small_batch_annotation(row, annotation, row['sentence_spans'],
                    allow_single_actor_without_conflict=allow_single)
                result['five_dimensions'] = annotation
                if obvious_quality_errors(annotation):
                    raise ValueError('Obvious role or contradiction error')
                # Retain the full candidate even if the later GPU audit is
                # interrupted. An unfinished checkpoint cannot pass admission.
                result['status'] = 'semantic_audit_pending'
                save(target, result)
                audit_schema = (field_semantic.schema_for(row, annotation) if audit_protocol
                                else semantic_schema(annotation))
                audit_payload = {'source': payload(row), 'candidate': annotation}
                if previous_bad_audit:
                    audit_payload['one_format_correction'] = {
                        'previous_audit': previous_bad_audit['semantic_audit'],
                        'format_errors': audit_structure_errors(previous_bad_audit['semantic_audit'], annotation),
                        'candidate_unchanged': fingerprint(annotation) == fingerprint(previous_bad_audit['five_dimensions']),
                        'instruction': '只重新核对质检输出的格式和实际来源。仅检查已声明逻辑父边；必要性、角色、经过或矛盾若确有错误仍填false，不强制通过，不改写候选。'}
                audit_prompt = field_semantic.PROMPT if audit_protocol else AUDIT_PROMPT
                audit_messages = [{'role': 'system', 'content': audit_prompt + '\nSchema：' + json.dumps(audit_schema.model_json_schema(), ensure_ascii=False)},
                    {'role': 'user', 'content': json.dumps(audit_payload, ensure_ascii=False)}]
                result['semantic_audit_prompt_binding'] = fingerprint(audit_messages)
                audited, audit_stats, audit_reserved = bounded_generation(tokenizer, [(audit_messages, audit_schema)], 1600,
                    context_limit, lambda e: progress(e, 'semantic_audit'))
                result['semantic_generation_stats'] = audit_stats
                if audit_reserved:
                    result['status'] = 'semantic_context_reserved'
                else:
                    _, audit_raw, audit_count = audited[0]
                    result.update(semantic_audit_raw=audit_raw, semantic_audit_prompt_tokens=audit_count)
                    parsed_audit = audit_schema.model_validate_json(audit_raw).model_dump()
                    if audit_protocol:
                        result['semantic_audit_wire'] = parsed_audit
                        result['semantic_audit'] = field_semantic.normalize(parsed_audit, row, annotation)
                        result['semantic_audit_binding'] = field_semantic.binding(row, annotation)
                    else:
                        result['semantic_audit'] = parsed_audit
                        result['semantic_audit_binding'] = audit_binding(row, annotation)
                    result['status'] = ('automatically_audited_pending_human_review'
                                        if quality_passed(result, row) else 'semantic_quality_rejected')
        except (ValueError, KeyError) as exc:
            runtime = isinstance(exc.__cause__, (ImportError, RuntimeError)) and not result.get('raw_output')
            result.update(status='local_runtime_error' if runtime else 'annotation_or_support_rejected', error=str(exc)[:400])
        except Exception as exc:
            result.update(status='local_annotation_error', error=str(exc)[:400])
        result['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(target, result); processed += 1
        print(json.dumps({'case_id': row['case_id'], 'status': result['status'],
                          'actual_depth': result.get('five_dimensions', {}).get('reasoning_depth')}, ensure_ascii=False), flush=True)
    results = [json.loads(p.read_text()) for p in out.glob('*.json')]
    report = {'protocol': PROTOCOL, 'cohort_binding': cohort['cohort_binding'],
              'generated_at': datetime.now(timezone.utc).isoformat(), 'cohort_cases': len(cohort['cases']),
              'completed': len(results), 'status_counts': dict(Counter(r['status'] for r in results)),
              'archived_runtime_only_failures': sum(bool(r.get('previous_runtime_failure')) for r in results),
              'accepted_actual_depth_counts': dict(Counter(str(r['five_dimensions']['reasoning_depth'])
                  for r in results if r['status'] == 'automatically_audited_pending_human_review')),
              'heldout_semantics_read': False, 'human_approvals_created': 0,
              'independent_accuracy_verified': False,
              'same_model_family_annotation_and_audit': any(r.get('annotator') == 'local-14b-base-machine-only' for r in results),
              'assistant_prepared_drafts': sum(r.get('annotator') == 'assistant_source_reader' for r in results)}
    report['audit_format_repairs'] = sum(r.get('audit_format_repair_attempts', 0) for r in results)
    report['semantic_audit_protocol'] = audit_protocol or 'legacy-global-flags'
    save(ROOT / 'data/small_batch_annotation_report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create-cohort', type=int)
    parser.add_argument('--cohort')
    parser.add_argument('--case-id')
    parser.add_argument('--audit-draft',help='Exact assistant-authored private draft; run only the separate semantic audit')
    parser.add_argument('--repair-audit-format',action='store_true',help='One archived retry for an internally malformed audit; candidate unchanged')
    parser.add_argument('--retry-runtime-errors', action='store_true',
                        help='One archived retry only when no model output was generated')
    parser.add_argument('--max-cases', type=int, default=1, choices=[1, 2])
    parser.add_argument('--max-tokens', type=int, default=3500)
    args = parser.parse_args()
    if bool(args.create_cohort) == bool(args.cohort):
        parser.error('Choose a cohort size to freeze or an existing cohort to process')
    if not 1000 <= args.max_tokens <= 4000:
        parser.error('Use 1000..4000 output tokens')
    if args.create_cohort:
        path, cohort = create_cohort(args.create_cohort)
        print(json.dumps({'cohort': str(path.relative_to(ROOT)), 'cases': cohort['cases']}, ensure_ascii=False))
    else:
        process(args)
