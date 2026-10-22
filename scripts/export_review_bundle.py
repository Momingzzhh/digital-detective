#!/usr/bin/env python3
"""Export private machine-screened material, with answers separate from agent input.

This is a curation bundle, never a certified benchmark. All exact quotations
are checked again against the pinned local source before export.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def visible(row):
    return {'case_id':row['case_id'], 'group_id':row['group_id'], 'split':row['split'],
            'evidence':[{'id':e['id'],'text':e['text'],
                         'kind':e['kind'].replace('被告人','当事人').replace('犯罪嫌疑人','当事人')}
                        for e in row['evidence']]}


def verify(row, source):
    full=source['raw_record']['qw'];fact=source['raw_record']['ajjbqk']
    if sha(full)!=row['source']['sha256_fulltext'] or sha(fact)!=row['source_fact_sha256']:
        raise ValueError('Source hash changed: '+row['case_id'])
    index={e['id']:e for e in row['evidence']}
    if len(index)!=len(row['evidence']):raise ValueError('Duplicate evidence IDs')
    for e in row['evidence']:
        if fact[e['start']:e['end']]!=e['quote']:
            raise ValueError('Evidence source offset mismatch')
        if e['text']!=e['quote'].replace('被告人','当事人').replace('犯罪嫌疑人','当事人'):
            raise ValueError('Unexpected evidence modification')
        if 'fulltext_start' in e and full[e['fulltext_start']:e['fulltext_end']]!=e['quote']:
            raise ValueError('Fulltext source offset mismatch')
    for event in row['hidden_gold'].get('chronology',[]):
        if 'supporting_sentences' in event:
            verify_support(event,index)
            continue
        if len(event['evidence_ids'])!=1:raise ValueError('Expected one exact supporting sentence')
        e=index[event['evidence_ids'][0]]
        if e['text'][event['sentence_start']:event['sentence_end']]!=event['summary']:
            raise ValueError('Gold sentence not exact')
    for actor in row['hidden_gold'].get('actor_roles',[]): verify_support(actor,index)


def verify_support(item,index):
    refs=item['supporting_sentences']
    if not refs or set(item['evidence_ids'])!={r['evidence_id'] for r in refs}:
        raise ValueError('Ground-truth support IDs differ from referenced sentences')
    for ref in refs:
        text=index[ref['evidence_id']]['text']
        if text[ref['sentence_start']:ref['sentence_end']]!=ref['text']:
            raise ValueError('Ground-truth support sentence differs from source')


def write_rows(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-version',default='v3_7')
    args=parser.parse_args()
    if not args.source_version.replace('_','').isalnum():parser.error('Invalid version identifier')
    sources={}
    for name in ('candidates.jsonl','lecardv2_candidates.jsonl','lecardv2_mteb_candidates.jsonl','mnbvc_candidates.jsonl'):
        file=ROOT/'data/processed'/name
        if file.exists():
            for line in file.open():
                row=json.loads(line);sources[row['case_id']]=row
    group_audit=json.loads((ROOT/'data/processed/case_group_audit.json').read_text())
    for name,expected in group_audit['source_files_sha256'].items():
        if hashlib.sha256((ROOT/'data/processed'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Group audit is stale; rerun scripts/audit_case_groups.py')
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'data/review'/args.source_version).glob('*.json'))]
    quarantine_path=ROOT/'data/review/agent_audit'/args.source_version/'quarantine.json'
    quarantine={r['case_id']:r for r in json.loads(quarantine_path.read_text())['entries']} if quarantine_path.exists() else {}
    selected=[];groups=set();excluded=Counter()
    for row in records:
        if row.get('status')!='machine_pass_unverified':continue
        verify(row,sources[row['case_id']])
        audited=quarantine.get(row['case_id'])
        if audited and hashlib.sha256(json.dumps(row['evidence'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==audited['evidence_sha256']:
            excluded['independent_agent_source_rejection']+=1;continue
        if any(charge in e['text'] for charge in row['hidden_gold']['charges'] for e in row['evidence']):
            excluded['exact_hidden_charge_in_visible_evidence']+=1;continue
        linked=group_audit['cases'][row['case_id']]
        if linked['cross_split_conflict']:
            excluded['cross_split_linkage_conflict']+=1;continue
        group=linked['linked_group_id']
        if group in groups:
            excluded['same_linked_group']+=1;continue
        row['source_group_id']=row['group_id'];row['group_id']=group
        groups.add(group);selected.append(row)
    out=ROOT/'data/processed/review_bundle'/args.source_version
    write_rows(out/'machine_curated.unverified.jsonl',selected)
    for part in ('train','dev','test'):
        subset=[r for r in selected if r['split']==part]
        write_rows(out/'visible'/(part+'.jsonl'),[visible(r) for r in subset])
        write_rows(out/'hidden'/(part+'.jsonl'),[{'case_id':r['case_id'],'hidden_gold':r['hidden_gold'],
                    'source':r['source'],'audit':r['audit']} for r in subset])
    stats={'generated_at':datetime.now(timezone.utc).isoformat(),'source_version':args.source_version,
           'reviewed':len(records),'status_counts':dict(Counter(r['status'] for r in records)),
           'machine_curated_cases':len(selected),'machine_curated_groups':len(groups),
           'excluded_after_export_audit':dict(excluded),
           'split_counts':dict(Counter(r['split'] for r in selected)),
           'source_offsets_verified':True,'qualified':0,'human_reviewed':0,
           'target_1000_complete':False,'accuracy_90_verified':False,
           'artifacts':{str(p.relative_to(out)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
                        for p in sorted(out.rglob('*.jsonl'))},
           'limitations':['Machine passes are provisional, not sufficient-evidence certification.',
                          'Related proceeding grouping and final outcomes still require source review.',
                          'Visible views contain no hidden fields, but semantic leakage needs review.',
                          'Raw judgments and all derived text must remain private local files.']}
    (out/'manifest.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'data/curation_report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:stats[k] for k in ('source_version','reviewed','machine_curated_cases','split_counts','qualified')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
