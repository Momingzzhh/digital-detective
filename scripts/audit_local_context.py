#!/usr/bin/env python3
"""CPU-only token length audit; never opens model weights or scores case answers."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.local_context import chat_token_ids,TOKEN_POLICY
from app.five_dimensions import fingerprint
from scripts.compact_annotation import schema_for,payload,INSTRUCTIONS


def main():
    from transformers import AutoTokenizer,__version__
    curator_path=ROOT/'models/qwen3-14b-mlx-4bit'
    curator=AutoTokenizer.from_pretrained(str(curator_path),local_files_only=True,trust_remote_code=False)
    rows=list(map(json.loads,(ROOT/'data/processed/five_dimensions_v2/candidates.private.jsonl').open()))
    pool_binding=fingerprint([r['source_binding'] for r in rows])
    if pool_binding != json.loads((ROOT/'data/five_dimensions_report.json').read_text())['pool_binding']:
        raise ValueError('Candidate pool binding changed')
    counts=Counter();samples=[]
    for row in rows:
        schema=schema_for(row)
        messages=[{'role':'system','content':INSTRUCTIONS+'\n只输出JSON数据实例。Schema：'+json.dumps(schema.model_json_schema(),ensure_ascii=False)},
                  {'role':'user','content':json.dumps(payload(row),ensure_ascii=False)}]
        size=len(chat_token_ids(curator,messages,thinking=False));fits=size+4000<=16384
        counts[(row['split'],'fits' if fits else 'reserved')]+=1
        samples.append({'case_id':row['case_id'],'split':row['split'],'source_binding':row['source_binding'],
                        'prompt_tokens':size,'fits':fits})
    (ROOT/'runtime').mkdir(exist_ok=True)
    (ROOT/'runtime/context_preflight.private.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'token_length_policy':TOKEN_POLICY,
        'transformers_version':__version__,'pool_binding':pool_binding,
        'curator_manifest_binding':fingerprint(json.loads((curator_path/'local_manifest.json').read_text())),
        'annotation_context_limit':16384,'annotation_generation_budget':4000,'fresh_annotation_only':True,
        'repairs_and_semantic_audits_checked_separately_at_runtime':True,'pool_cases':len(rows),
        'by_split':{s:{k:counts[(s,k)] for k in ('fits','reserved')} for s in ('train','dev','test')},
        'source_truncation':False,'qualified_cases_from_this_check':0,
        'min_prompt_tokens':min(r['prompt_tokens'] for r in samples),'max_prompt_tokens':max(r['prompt_tokens'] for r in samples)}
    pilot=ROOT/'artifacts/training/20261021T151617Z/data/train.jsonl'
    if pilot.exists():
        agent=AutoTokenizer.from_pretrained(str(ROOT/'models/qwen3-4b-instruct-2507-4bit'),local_files_only=True,trust_remote_code=False)
        lengths=[len(chat_token_ids(agent,e['messages'],generation=False)) for e in map(json.loads,pilot.open())]
        report['superseded_v1_pilot_length_reaudit']={'examples':len(lengths),'max_tokens':max(lengths),
                                                   'over_6144':sum(n>6144 for n in lengths),'acceptance_result':False}
    (ROOT/'data/local_context_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
