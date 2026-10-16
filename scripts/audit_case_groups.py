#!/usr/bin/env python3
"""Conservative cross-source linkage; quarantine conflicting frozen splits.

Case-number references can include earlier unrelated convictions. False merges
reduce the pool. Missing references and paraphrased duplicates can still escape.
This does not certify that an unflagged document is an independent case.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
from prepare_candidates import CASE,normalized

ROOT=Path(__file__).resolve().parents[1]
INPUTS=('candidates.jsonl','lecardv2_candidates.jsonl',
        'lecardv2_mteb_candidates.jsonl','mnbvc_candidates.jsonl')


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


COURT=re.compile(r'[\u4e00-\u9fff]{2,35}人民法院')
MODERN=re.compile(r'^\(\d{4}\)[\u4e00-\u9fff]{1,3}\d{2,4}刑')


def court_name(text):
    match=COURT.search(text)
    if not match:return None
    return re.split(r'判决书|裁定书|决定书',match.group())[-1]


def reference_keys(text):
    issuer=court_name(text[:500])
    for match in CASE.finditer(text):
        number=normalized(match.group())
        if MODERN.match(number):
            yield 'case:modern:'+number
        else:
            # Old short numbers recur in unrelated courts. Prefer an explicit
            # court immediately before the reference, otherwise issuer scope.
            court=court_name(text[max(0,match.start()-80):match.start()]) or issuer
            if court:yield 'case:old:'+court+':'+number


def link(rows):
    parents={r['case_id']:r['case_id'] for r in rows}
    def find(key):
        while parents[key]!=key:
            parents[key]=parents[parents[key]];key=parents[key]
        return key
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    seen={}
    for row in rows:
        key=row['case_id'];record=row['raw_record']
        keys=['old_group:'+row['group_id'],'full:'+sha(normalized(record['qw']))]
        keys+=list(reference_keys(record['qw']))
        fact=normalized(record.get('ajjbqk',''))
        if len(fact)>100:keys.append('fact:'+sha(fact))
        for marker in keys:
            if marker in seen:union(key,seen[marker])
            else:seen[marker]=key
    clusters=defaultdict(list)
    for row in rows:clusters[find(row['case_id'])].append(row)
    mapping={};quarantined=0;conflicts=0
    for key,members in clusters.items():
        parts=sorted({r['split'] for r in members});conflict=len(parts)>1
        conflicts+=int(conflict);quarantined+=len(members) if conflict else 0
        for row in members:
            mapping[row['case_id']]={'linked_group_id':'linked-'+sha(key)[:24],
                 'cross_split_conflict':conflict,'linked_documents':len(members)}
    return mapping,{'source_documents':len(rows),'linked_groups':len(clusters),
                    'cross_split_conflict_groups':conflicts,
                    'quarantined_documents':quarantined,
                    'cluster_size_distribution':dict(Counter(len(v) for v in clusters.values()))}


def main():
    rows=[];fingerprints={}
    for name in INPUTS:
        p=ROOT/'data/processed'/name
        if p.exists():
            fingerprints[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            rows.extend(json.loads(line) for line in p.open())
    mapping,stats=link(rows)
    private={'source_files_sha256':fingerprints,'cases':mapping,'counts':stats}
    p=ROOT/'data/processed/case_group_audit.json'
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(private,ensure_ascii=False));temp.replace(p)
    report={'method':'Transitive source group, normalized full/fact exact hash, modern court-coded case references and court-scoped old references',
            'counts':stats,'source_files_sha256':fingerprints,
            'limitations':['Old references use a nearby explicit court or issuing-court scope; missing/cross-court references can escape.',
                           'Prior convictions and shared references can falsely merge independent events.',
                           'Unmentioned related proceedings and paraphrased duplicates can still escape.',
                           'Conflicting frozen splits are quarantined, never silently reassigned.',
                           'No unflagged case is thereby certified independent.']}
    (ROOT/'data/case_group_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
