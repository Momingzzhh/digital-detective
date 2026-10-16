#!/usr/bin/env python3
"""Import locally downloaded LeCaRDv2 full-text triplets as unreviewed cases.

Triplets are a retrieval format; anchor/positive/negative are not separate
labels or answer keys. The same judgment can appear in many triplets. This
script deduplicates source documents and never certifies evidence sufficiency.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from prepare_candidates import CASE, normalized
from fetch_lecardv2 import FILES as SOURCE_FILES, digest as file_digest

ROOT=Path(__file__).resolve().parents[1]
REV='84ebcf30a342b044e4c99c3510e34bb8830356da'
COURT=re.compile(r'[\u4e00-\u9fff]{2,35}人民法院')
FINDING=re.compile(r'经审理查明[：:]?|本院经审理查明[：:]?')
REASON=re.compile(r'本院认为[：:]?')
VERDICT=re.compile(r'判决如下[：:]?|裁定如下[：:]?')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def case_key(text):
    head=text[:500]
    number=CASE.search(head)
    if not number:
        return None
    # A case number is treated as a conservative global deduplication key.
    # Older short numbers can collide across courts; false merges reduce the
    # candidate pool but cannot leak a related case across frozen splits.
    return normalized(number.group())


def split_document(text):
    f=FINDING.search(text)
    if not f:
        return None
    reason=REASON.search(text,f.end())
    verdicts=list(VERDICT.finditer(text))
    if not reason or not verdicts or verdicts[-1].start()<reason.end():
        return None
    fact=text[f.start():reason.start()]
    judgment=text[verdicts[-1].start():]
    if not fact or not judgment:
        return None
    return fact,judgment,f.start()


def main():
    for _,(name,expected) in SOURCE_FILES.items():
        source=ROOT/'data/raw'/name
        if not source.exists() or file_digest(source)!=expected:
            raise ValueError('Pinned source missing or checksum mismatch: '+name)
    old={}
    for line in (ROOT/'data/processed/candidates.jsonl').open():
        row=json.loads(line)
        key=case_key(row['raw_record'].get('qw',''))
        if key:
            old.setdefault(key,(row['group_id'],row['split']))
    seen_text=set();seen_case=set();counts=Counter();output=[]
    for part in sorted((ROOT/'data/raw').glob('lecardv2-triplet-all-*.parquet')):
        table=pq.ParquetFile(part)
        for batch_index,batch in enumerate(table.iter_batches(batch_size=128)):
            for row_index,record in enumerate(batch.to_pylist()):
                for role in ('anchor','positive','negative'):
                    text=record.get(role) or ''
                    full_hash=digest(text)
                    if full_hash in seen_text:
                        counts['repeated_triplet_document']+=1
                        continue
                    seen_text.add(full_hash)
                    key=case_key(text)
                    if not key or key in seen_case:
                        counts['missing_or_repeated_primary_case_number']+=1
                        continue
                    seen_case.add(key)
                    if key in old:
                        counts['also_in_lecard_v1']+=1
                        continue
                    parsed=split_document(text)
                    if not parsed:
                        counts['missing_fact_reason_or_verdict_section']+=1
                        continue
                    fact,judgment,fact_offset=parsed
                    group='lecardv2-group-'+digest(key)[:24]
                    ratio=int(digest(group)[:16],16)/2**64
                    split='train' if ratio<.7 else ('dev' if ratio<.85 else 'test')
                    case_id='lecardv2-'+digest(key)[:32]
                    output.append({'case_id':case_id,'group_id':group,'split':split,
                       'status':'candidate_unreviewed',
                       'source':{'repository':'https://github.com/THUIR/LeCaRDv2',
                                 'derivative':'https://huggingface.co/datasets/sentence-transformers/lecard-v2',
                                 'derivative_commit':REV,'parquet':part.name,
                                 'triplet_row':batch_index*128+row_index,'triplet_role':role,
                                 'sha256_fulltext':full_hash,'primary_case_key_sha256':digest(key),
                                 'fact_offset_in_fulltext':fact_offset},
                       'raw_record':{'ajjbqk':fact,'pjjg':judgment,'qw':text}})
                    counts['new_source_candidates']+=1
    output.sort(key=lambda row:row['case_id'])
    path=ROOT/'data/processed/lecardv2_candidates.jsonl'
    with path.open('w') as file:
        for row in output:
            file.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'source':'LeCaRDv2 triplet-all derivative','source_revision':REV,
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in sorted((ROOT/'data/raw').glob('lecardv2-triplet-all-*.parquet'))},
            'counts':dict(counts),'output_records':len(output),'qualified':0,
            'limitations':['Triplets repeat documents; no relevance labels are used as case truth.',
                           'Legal retrieval source is not an investigation benchmark.',
                           'Fact sections can contain court findings; only exact source evidence may be released.',
                           'Primary case-number and court grouping can still miss related proceedings.',
                           'Underlying judgments remain local-only pending rights/privacy review.']}
    (ROOT/'data/lecardv2_source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('counts','output_records','qualified')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
