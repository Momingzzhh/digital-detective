#!/usr/bin/env python3
"""Freeze a private benchmark only after two independent human source reviews.

The attestation files are local and contain reviewer identities. The script
cannot prove a person actually reviewed the material; it enforces a concrete,
reproducible checklist and refuses to substitute model judgments for it.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
CHECKS=('source_authentic','all_evidence_is_original_source','no_outcome_leakage',
        'all_perpetrators_accounted_for','charge_supported','chronology_supported',
        'alternatives_resolved','identity_mapping_visible','same_case_group_checked')


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def group_binding(record):
    return digest({'group_id':record['group_id'],'split':record['split']})


def verify_reviews(record, paths):
    approved=[]; reviewers=set()
    for path in paths:
        review=json.loads(path.read_text())
        if review.get('case_id')!=record['case_id'] or review.get('source_sha256')!=record['source']['sha256_fulltext']:
            raise ValueError('Review source/case mismatch: '+str(path))
        if review.get('evidence_sha256')!=digest(record['evidence']) or review.get('gold_sha256')!=digest(record['hidden_gold']):
            raise ValueError('Review version changed after attestation: '+str(path))
        if review.get('group_split_sha256')!=group_binding(record):
            raise ValueError('Reviewed group/split changed: '+str(path))
        name=review.get('reviewer_id','').strip()
        if not name or name in reviewers or not review.get('reviewed_at'):
            raise ValueError('Reviewer identity/date missing or duplicated: '+str(path))
        reviewers.add(name)
        if review.get('decision')=='approve' and all(review.get('checks',{}).get(k) is True for k in CHECKS):
            approved.append(review)
    return len(approved)>=2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-version',default='v3_7')
    parser.add_argument('--minimum',type=int,default=1000)
    parser.add_argument('--template',help='Create a local review checklist for one case ID')
    parser.add_argument('--reviewer-id',help='Required with --template')
    args=parser.parse_args()
    if args.minimum<1000:parser.error('A complete benchmark needs at least 1000 independently reviewed cases')
    files=sorted((ROOT/'data/review'/args.source_version).glob('*.json'))
    if args.template:
        if not args.reviewer_id or not args.reviewer_id.replace('_','').replace('-','').isalnum():
            parser.error('--reviewer-id must be a simple nonempty identifier')
        candidate=ROOT/'data/review'/args.source_version/(args.template+'.json')
        if candidate not in files:parser.error('case checkpoint not found')
        record=json.loads(candidate.read_text())
        target=ROOT/'data/review/human'/args.template/(args.reviewer_id+'.json')
        target.parent.mkdir(parents=True,exist_ok=True)
        template={'case_id':args.template,'reviewer_id':args.reviewer_id,'reviewed_at':'',
                  'decision':'reject','source_sha256':record['source']['sha256_fulltext'],
                  'evidence_sha256':digest(record['evidence']),
                  'gold_sha256':digest(record['hidden_gold']),
                  'group_split_sha256':group_binding(record),
                  'checks':{key:False for key in CHECKS},'notes':''}
        with target.open('x') as out:out.write(json.dumps(template,ensure_ascii=False,indent=2)+'\n')
        print(target)
        return
    passed=[];groups=set();counts=Counter()
    for file in files:
        record=json.loads(file.read_text())
        if record.get('status')!='machine_pass_unverified':
            counts['machine_not_passed']+=1;continue
        checks=sorted((ROOT/'data/review/human'/record['case_id']).glob('*.json'))
        if not checks or not verify_reviews(record,checks):
            counts['human_review_incomplete']+=1;continue
        if record['group_id'] in groups:
            raise ValueError('Duplicate source case group: '+record['group_id'])
        groups.add(record['group_id'])
        record['status']='independently_verified'
        record['audit']={'human_reviewed':True,'reviewer_count':len(checks),
                         'attestation_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in checks]}
        passed.append(record)
        counts['independently_verified']+=1
    if len(passed)<args.minimum or len({r['group_id'] for r in passed})<args.minimum:
        print(json.dumps({'certified':len(passed),'required':args.minimum,
                          'status_counts':dict(counts),'benchmark_written':False},ensure_ascii=False,indent=2))
        raise SystemExit(2)
    for part in ('train','dev','test'):
        if not any(row['split']==part for row in passed):
            raise ValueError('Missing frozen split: '+part)
    # Revalidate the source and current cross-source linkage before writing any
    # formal benchmark; attestations alone cannot override corrupted material.
    from audit_case_groups import INPUTS
    from export_review_bundle import verify
    linkage=json.loads((ROOT/'data/processed/case_group_audit.json').read_text())
    for name in INPUTS:
        path=ROOT/'data/processed'/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=linkage['source_files_sha256'].get(name):
            raise ValueError('Group audit missing or stale')
    wanted={r['case_id']:r for r in passed};linked_groups=set()
    for name in INPUTS:
        path=ROOT/'data/processed'/name
        if not path.exists():continue
        for line in path.open():
            source=json.loads(line);key=source['case_id']
            if key not in wanted:continue
            record=wanted.pop(key);verify(record,source)
            group=linkage['cases'][key]
            if group['cross_split_conflict'] or group['linked_group_id'] in linked_groups:
                raise ValueError('Unresolved related-case split conflict or duplicate')
            linked_groups.add(group['linked_group_id'])
            record['source_group_id']=record['group_id'];record['group_id']=group['linked_group_id']
    if wanted:raise ValueError('Source document missing')
    destination=ROOT/'data/processed/benchmark.jsonl'
    temporary=destination.with_suffix('.tmp')
    with temporary.open('w') as out:
        for row in sorted(passed,key=lambda r:r['case_id']):out.write(json.dumps(row,ensure_ascii=False)+'\n')
    temporary.replace(destination)
    print(json.dumps({'certified':len(passed),'split_counts':dict(Counter(x['split'] for x in passed)),
                      'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                      'benchmark_written':True},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
