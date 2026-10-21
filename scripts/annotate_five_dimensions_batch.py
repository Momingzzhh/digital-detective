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

QUOTAS = {'train': 420, 'dev': 90, 'test': 90}
PROTOCOL = 'compact-json-v3-narrow-source-associations'


def binding(row):
    return fingerprint({'version': VERSION, 'prompt': PROMPT,
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


def main(args):
    acquire_annotation_lock()
    os.environ['LOCAL_ADAPTER_PATH'] = ''  # Fixed base annotator, never the agent under evaluation.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path()), local_files_only=True, trust_remote_code=False)
    context_limit = int(settings().get('LOCAL_CONTEXT_TOKENS', '16384'))
    output = ROOT / 'data/review/five_dimensions_v1'; output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (ROOT / 'data/processed/five_dimensions_v1/candidates.private.jsonl').open()]
    counts = Counter(); pending = []; existing = Counter()
    # Source metadata priority is independent of agent predictions and scores.
    rows.sort(key=lambda r: len(r['hidden_gold']['perpetrators']) < 2)
    for row in rows:
        path = output / (row['case_id'] + '.json'); expected = binding(row)
        if path.exists():
            old = json.loads(path.read_text())
            if old.get('annotation_binding') == expected:
                existing[old['status']] += 1
                output_error = old['status'] == 'local_annotation_error' or (
                    not old.get('raw_annotation') and 'validation error' in old.get('error',''))
                old_compact = (old.get('generation_protocol') == 'compact-json-v1-eos-stop' and
                               old['status'] != 'machine_annotated_unverified')
                if (args.retry_rejected and old['status'] != 'machine_annotated_unverified' and
                    old.get('generation_protocol') != PROTOCOL): old_compact = True
                if not old_compact and (not args.retry_errors or not output_error): continue
                history = output / 'history'; history.mkdir(exist_ok=True)
                archived = history / (row['case_id'] + '.compact-v1.json')
                if not archived.exists(): archived.write_bytes(path.read_bytes())
            else:
                history = output / 'history'; history.mkdir(exist_ok=True)
                archived = history / (row['case_id'] + '.' + old['annotation_binding'][:12] + '.json')
                if not archived.exists(): archived.write_bytes(path.read_bytes())
        pending.append(row)
    def refresh_counts():
        subprocess.run([sys.executable,str(ROOT/'scripts/assemble_five_dimensions.py')],check=True,
                       stdout=subprocess.DEVNULL)
        report = json.loads((ROOT/'data/five_dimensions_report.json').read_text())
        return Counter(report['current_split_counts'])
    # Export, rather than raw annotation counts, also accounts for overlength
    # training cases and explicit source-bound human rejections.
    counts = refresh_counts()
    (ROOT / 'runtime').mkdir(exist_ok=True)
    (ROOT / 'runtime/five_dimensions_annotation.pid').write_text(str(os.getpid()))
    processed = 0
    while pending and processed < args.limit and any(counts[s] < n for s,n in QUOTAS.items()):
        if (ROOT / 'runtime/stop_five_dimensions_annotation').exists(): break
        candidates = [r for r in pending if counts[r['split']] < QUOTAS[r['split']]]
        batch = candidates[:min(args.batch_size, args.limit - processed)]
        if not batch: break
        ids = {r['case_id'] for r in batch}; pending = [r for r in pending if r['case_id'] not in ids]
        schemas = [schema_for(row) for row in batch]; requests = []; valid_rows = []; valid_schemas = []
        reserved = 0
        for row, schema in zip(batch, schemas):
            system = INSTRUCTIONS + '\n只输出JSON数据实例。Schema：' + json.dumps(schema.model_json_schema(), ensure_ascii=False)
            messages = [{'role':'system','content':system}, {'role':'user','content':json.dumps(payload(row),ensure_ascii=False)}]
            prompt_tokens = len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
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
            texts, stats = complete_batch(requests, args.max_tokens) if requests else ([],{})
            batch_error = None
        except Exception as exc:
            texts = [''] * len(valid_rows); stats = {}; batch_error = str(exc)[:300]
        for index,(row, schema, text) in enumerate(zip(valid_rows, valid_schemas, texts)):
            result = {'case_id':row['case_id'],'group_id':row['group_id'],'split':row['split'],
                'annotation_binding':binding(row),'source_binding':row['source_binding'],
                'annotator':'local-base-constrained-minibatch','human_reviewed':False,
                'generation_protocol':PROTOCOL,
                'actual_prompt_sha256':fingerprint(requests[index][0]),
                'generated_at':datetime.now(timezone.utc).isoformat()}
            try:
                if batch_error: raise RuntimeError(batch_error)
                raw = schema.model_validate_json(text).model_dump(); result['raw_compact_annotation'] = raw
                data = canonical(raw, row); result['raw_annotation'] = data
                normalized, repairs = normalize_references(data, row['sentence_spans'])
                result['identifier_repairs'] = repairs
                ann = validate_annotation(row, normalized, row['sentence_spans'])
                result.update(status='machine_annotated_unverified',five_dimensions=ann)
            except ValidationError as exc: result.update(status='local_annotation_error',error=str(exc)[:300])
            except ValueError as exc: result.update(status='complexity_or_reference_rejected',error=str(exc)[:300])
            except Exception as exc: result.update(status='local_annotation_error',error=str(exc)[:300])
            save(output / (row['case_id'] + '.json'), result)
            existing[result['status']] += 1
        processed += len(batch)
        counts = refresh_counts()
        print(json.dumps({'processed':processed,'seconds':round(time.monotonic()-begin,2),
            'accepted_by_split':dict(counts),'attempt_status_counts':dict(existing),
            'reserved_over_context_this_batch':reserved,'generation_stats':stats}), flush=True)
        if batch_error: raise RuntimeError(batch_error)
    print(json.dumps({'finished':True,'processed':processed,'accepted_by_split':dict(counts)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=1980)
    parser.add_argument('--batch-size',type=int,choices=[1,2,4,8],default=4)
    parser.add_argument('--max-tokens',type=int,default=2600)
    parser.add_argument('--retry-errors',action='store_true')
    parser.add_argument('--retry-rejected',action='store_true',help='Retry failures from an older protocol once; never changes splits or validity checks')
    main(parser.parse_args())
