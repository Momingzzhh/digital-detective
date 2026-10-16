#!/usr/bin/env python3
"""Add checksum-pinned MTEB judgments; retrieval relevance is never case truth."""
import json
from collections import Counter
from pathlib import Path
from prepare_lecardv2 import case_key, split_document, digest
from fetch_lecardv2 import digest as file_digest

ROOT=Path(__file__).resolve().parents[1]
REV='566865fc50a566db1635307496e46c5a1a118cb2'
SHA='f112eb82ff3b7cbc1e97e44d96ae9e520746df3858b77a0028115f1436454c0c'


def main():
    path=ROOT/'data/raw/lecardv2-mteb-corpus.jsonl'
    if file_digest(path)!=SHA:
        raise ValueError('MTEB source checksum mismatch')
    keys=set(); full_hashes=set()
    for name in ('candidates.jsonl','lecardv2_candidates.jsonl'):
        for line in (ROOT/'data/processed'/name).open():
            row=json.loads(line);text=row['raw_record']['qw']
            keys.add(case_key(text));full_hashes.add(digest(text))
    counts=Counter();output=[]
    for line_number,line in enumerate(path.open(),1):
        source=json.loads(line);text=source['text'];key=case_key(text)
        counts['documents']+=1
        if not key:
            counts['missing_case_number']+=1;continue
        if key in keys or digest(text) in full_hashes:
            counts['already_imported_or_duplicate']+=1;continue
        keys.add(key);full_hashes.add(digest(text))
        parsed=split_document(text)
        if not parsed:
            counts['missing_sections']+=1;continue
        fact,judgment,offset=parsed
        group='lecardv2-group-'+digest(key)[:24]
        ratio=int(digest(group)[:16],16)/2**64
        output.append({'case_id':'lecardv2-'+digest(key)[:32], 'group_id':group,
             'split':'train' if ratio<.7 else ('dev' if ratio<.85 else 'test'),
             'status':'candidate_unreviewed',
             'source':{'repository':'https://github.com/THUIR/LeCaRDv2',
                       'derivative':'https://huggingface.co/datasets/mteb/LeCaRDv2',
                       'derivative_commit':REV,'file':path.name,'source_line':line_number,
                       'source_document_id':source['_id'],'sha256_fulltext':digest(text),
                       'primary_case_key_sha256':digest(key),'fact_offset_in_fulltext':offset},
             'raw_record':{'ajjbqk':fact,'pjjg':judgment,'qw':text}})
        counts['new_source_candidates']+=1
    output.sort(key=lambda r:r['case_id'])
    target=ROOT/'data/processed/lecardv2_mteb_candidates.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    report={'source':'MTEB LeCaRDv2 corpus','revision':REV,'file_sha256':SHA,
            'download_url':f'https://hf-mirror.com/datasets/mteb/LeCaRDv2/resolve/{REV}/corpus.jsonl',
            'counts':dict(counts),'qualified':0,'limitations':[
                'Source evidence, all actors and outcomes still require review.',
                'Related proceedings can remain undetected despite primary number deduplication.',
                'Fulltext and all derived case content are private local files.']}
    (ROOT/'data/mteb_source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
