#!/usr/bin/env python3
"""Create explicit train-only source revisions; never replace the frozen pool."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.five_dimensions import fingerprint
from app.source_policy import require_clean_evidence
from scripts.build_v3 import START,ITEM,sentences
from scripts.export_review_bundle import verify,write_rows

VERSION='calibration-source-supplement-v1'


def numbered_items(row):
    fact=row['raw_record']['ajjbqk'];marker=START.search(fact)
    if not marker:raise ValueError('No explicit source evidence section')
    matches=list(ITEM.finditer(fact[marker.end():]));result=[]
    for index,match in enumerate(matches):
        start=marker.end()+match.start()
        end=marker.end()+matches[index+1].start() if index+1<len(matches) else len(fact)
        result.append((start,end))
    return result


def revise(row,plan,feedbacks):
    if row['split']!='train' or plan['source_binding_before']!=row['source_binding']:
        raise ValueError('Only source-bound training cases can be revised')
    if not feedbacks or any(f.get('human_attestation') is not True or f['source_binding']!=row['source_binding']
        or f['annotation_result_binding']!=plan['annotation_result_binding'] or f['case_id']!=row['case_id'] for f in feedbacks):
        raise ValueError('Confirmed feedback must match the prior source and annotation')
    out=copy.deepcopy(row);fact=row['raw_record']['ajjbqk'];items=numbered_items(row);changes=[]
    used=set()
    for action in plan['changes']:
        number=action['source_item'];start,end=items[number-1];raw=fact[start:end]
        if action.get('start_at'):
            needle=action['start_at']
            if raw.count(needle)!=1:raise ValueError('Ambiguous excerpt start')
            start+=raw.index(needle)
        if action.get('end_before'):
            needle=action['end_before']
            if raw.count(needle)!=1:raise ValueError('Ambiguous excerpt end')
            end=items[number-1][0]+raw.index(needle)
        if start>=end:raise ValueError('Empty source revision')
        quote=fact[start:end];text=quote.replace('被告人','当事人').replace('犯罪嫌疑人','当事人')
        existing=action.get('replace_id')
        if existing:
            positions=[i for i,e in enumerate(out['evidence']) if e['id']==existing]
            if len(positions)!=1 or existing in used:raise ValueError('Unknown or repeated replacement')
            position=positions[0];eid=existing;before=fingerprint(out['evidence'][position])
        else:
            position=len(out['evidence']);eid=f'E{position+1}';before=None
        used.add(eid)
        item={'id':eid,'text':text,'quote':quote,'kind':action['kind'],'start':start,'end':end,
            'source_field':'ajjbqk','source_context_before':fact[max(0,start-180):start],
            'transforms':[] if quote==text else [{'type':'neutral_role_label','from':['被告人','犯罪嫌疑人'],'to':'当事人'}]}
        offset=row['source'].get('fact_offset_in_fulltext')
        if offset is not None:
            item.update(fulltext_start=offset+start,fulltext_end=offset+end)
        require_clean_evidence([item])
        if any(e['quote']==quote for e in out['evidence'] if e['id']!=eid):raise ValueError('Duplicate source fragment')
        if existing:out['evidence'][position]=item
        else:out['evidence'].append(item)
        changes.append({'evidence_id':eid,'source_item':number,'start':start,'end':end,
            'action':'replace' if existing else 'append','before_binding':before,
            'after_binding':fingerprint(item),'reason':action['reason']})
    require_clean_evidence(out['evidence'])
    out['sentence_spans']=sentences(out['evidence'])
    out['source_binding']=fingerprint({k:out[k] for k in
        ('case_id','group_id','split','source','evidence','sentence_spans','source_fact_sha256')})
    out['source_revision']={'version':VERSION,'parent_source_binding':row['source_binding'],
        'plan_binding':fingerprint(plan),'feedback_bindings':[fingerprint(f) for f in feedbacks],
        'changes':changes,'deferred':plan.get('deferred',[])}
    out['pipeline_version']=VERSION;out['status']='awaiting_source_revision_annotation'
    out['human_reviewed']=False
    out.pop('five_dimensions',None)
    verify(out,{'raw_record':out['raw_record']})
    # Existing evidence IDs stay stable where quotes are unchanged. Changed IDs
    # carry new offsets and a new source binding; old approvals never transfer.
    return out


def main(args):
    plans=json.loads(args.plan.read_text())
    pool_path=ROOT/'data/processed/five_dimensions_v2/candidates.private.jsonl'
    pool_hash=fingerprint(pool_path.read_text())
    pool={r['case_id']:r for r in map(json.loads,pool_path.open())}
    revised=[]
    for plan in plans['cases']:
        row=pool[plan['case_id']]
        folder=ROOT/'data/review/five_dimensions_v2_calibration'/row['case_id']
        feedbacks=[json.loads(p.read_text()) for p in folder.glob('*.json')]
        feedbacks=[f for f in feedbacks if f.get('feedback_protocol')=='word-calibration-feedback-v1'
                   and f.get('source_binding')==row['source_binding']
                   and f.get('annotation_result_binding')==plan['annotation_result_binding']]
        revised.append(revise(row,plan,feedbacks))
    binding=fingerprint([r['source_binding'] for r in revised]);out=ROOT/'data/processed/five_dimensions_v2_calibration_revision'/binding[:16]
    out.mkdir(parents=True,exist_ok=True);target=out/'candidates.private.jsonl'
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in revised)
    if target.exists() and target.read_text()!=data:raise ValueError('Immutable revision already exists with different bytes')
    if not target.exists():target.write_text(data)
    if fingerprint(pool_path.read_text())!=pool_hash:raise ValueError('Frozen pool changed unexpectedly')
    manifest={'version':VERSION,'generated_at':datetime.now(timezone.utc).isoformat(),
        'source_revision_binding':binding,'plan_binding':fingerprint(plans),'cases':len(revised),
        'source_dataset':str(target.relative_to(ROOT)),'train_only':True,'frozen_pool_unchanged':True,
        'source_offsets_verified':True,'qualified_cases_added':0,
        'changes_by_case':[{k:r[k] for k in ('case_id','source_binding','source_revision')} for r in revised]}
    (out/'manifest.private.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('source_dataset','cases','train_only','frozen_pool_unchanged','qualified_cases_added')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    main(parser.parse_args())
