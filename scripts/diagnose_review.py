#!/usr/bin/env python3
"""Retry one saved audit, printing schema diagnostics without response inputs."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_v3 import Audit, Annotation, REVIEW, ANNOTATE, sentences
from app.llm import structured
from app.store import initialize, budget_status


async def main(case_id,version,mode):
    initialize()
    row = next(json.loads(s) for s in (ROOT/'data/processed/v3_preflight.jsonl').open()
               if json.loads(s)['case_id'] == case_id)
    saved = json.loads((ROOT/'data/review'/version/f'{case_id}.json').read_text())
    payload = {'original': {'fact': row['source_fact_excerpt'],
                            'judgment': row['source_judgment_excerpt']},
               'published_evidence': [{k:e[k] for k in ('id','text','source_context_before')}
                                      for e in saved['evidence']],
               'hidden_gold': saved['hidden_gold']}
    try:
        if mode=='annotation':
            payload={'evidence':[{k:e[k] for k in ('id','text','kind')} for e in saved['evidence']],
                     'sentences':sentences(saved['evidence']),
                     'hidden_adjudicated_fact':row['source_fact_excerpt']}
            result=await structured(ANNOTATE,payload,Annotation,'v3_annotation_diagnostic',1800,
                                    model_override='qwen3.8-flash')
        else:
            result=await structured(REVIEW,payload,Audit,'v3_audit_diagnostic',4000,
                                    model_override='qwen3.8-max')
        print(json.dumps({mode:result.model_dump()},ensure_ascii=False,indent=2))
        out=ROOT/'data/review/diagnostics';out.mkdir(exist_ok=True)
        (out/f'{case_id}.{mode}.json').write_text(json.dumps(result.model_dump(),ensure_ascii=False,indent=2)+'\n')
    except Exception as exc:
        cause = exc.__cause__
        safe = {'error':str(exc), 'cause_type':type(cause).__name__}
        if hasattr(cause, 'errors'):
            safe['schema_errors'] = [{k:v for k,v in e.items() if k in ('type','loc','msg')}
                                     for e in cause.errors(include_input=False, include_url=False)]
        print(json.dumps(safe, ensure_ascii=False, indent=2))
    print(json.dumps({'budget':budget_status()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_id')
    parser.add_argument('--source-version',default='v3_7')
    parser.add_argument('--mode',choices=('audit','annotation'),default='audit')
    args=parser.parse_args()
    if not args.source_version.replace('_','').isalnum():parser.error('Invalid version')
    asyncio.run(main(args.case_id,args.source_version,args.mode))
