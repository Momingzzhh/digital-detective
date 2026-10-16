#!/usr/bin/env python3
"""Stream selected public judgment shards into private, unverified candidates."""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
from prepare_lecardv2 import case_key, split_document, digest
from fetch_lecardv2 import digest as file_digest

ROOT=Path(__file__).resolve().parents[1]
SUSPECT_CHARS=re.compile(r'[\u0f00-\u1fff\ue000-\uf8ff\ufffd]')


def main():
    manifest=json.loads((ROOT/'data/mnbvc_download_manifest.json').read_text())
    known=set();hashes=set()
    for name in ('candidates.jsonl','lecardv2_candidates.jsonl','lecardv2_mteb_candidates.jsonl'):
        path=ROOT/'data/processed'/name
        if path.exists():
            for line in path.open():
                row=json.loads(line);text=row['raw_record']['qw']
                known.add(case_key(text));hashes.add(digest(text))
    counts=Counter();per_shard={};output=[];selected={}
    destination=ROOT/'data/processed/mnbvc_candidates.jsonl'
    # Appending source shards must not replace a previously reviewed source
    # version or silently move a case between frozen partitions.
    if destination.exists():
        for line in destination.open():
            row=json.loads(line);text=row['raw_record']['qw']
            if digest(text)!=row['source']['sha256_fulltext']:
                raise ValueError('Preserved candidate source changed')
            if row['source']['derivative_commit']!=manifest['revision']:
                raise ValueError('Cannot mix source revisions')
            known.add(case_key(text));hashes.add(digest(text));output.append(row)
        counts['preserved_existing_candidates']=len(output)
    for file in sorted((ROOT/'data/raw').glob('mnbvc-*.jsonl.gz')):
        remote='data/'+file.name.removeprefix('mnbvc-')
        expected=manifest['files'][remote]
        if file.stat().st_size!=expected['bytes'] or file_digest(file)!=expected['sha256']:
            raise ValueError('Source shard checksum mismatch: '+file.name)
        selected[file.name]=expected
        local=Counter()
        for line_number,line in enumerate(gzip.open(file,'rt'),1):
            row=json.loads(line);local['documents']+=1
            text=''.join(x.get('内容','') for x in row.get('段落',[]))
            if '被告人' not in text or '判决如下' not in text:
                local['not_criminal_verdict']+=1;continue
            local['criminal_verdict_marker']+=1
            if row.get('是否待查文件') or row.get('低质量段落数',0) or SUSPECT_CHARS.search(text):
                local['flagged_or_suspected_corruption']+=1;continue
            key=case_key(text);full_hash=digest(text)
            if not key:
                local['missing_primary_number']+=1;continue
            if key in known or full_hash in hashes:
                local['already_imported_or_duplicate']+=1;continue
            known.add(key);hashes.add(full_hash)
            parsed=split_document(text)
            if not parsed:
                local['missing_sections']+=1;continue
            fact,judgment,offset=parsed
            group='mnbvc-group-'+digest(key)[:24]
            ratio=int(digest(group)[:16],16)/2**64
            output.append({'case_id':'mnbvc-'+digest(key)[:32],'group_id':group,
                  'split':'train' if ratio<.7 else ('dev' if ratio<.85 else 'test'),
                  'status':'candidate_unreviewed',
                  'source':{'repository':'https://github.com/wormtooth/MNBVC-judgment',
                            'derivative':manifest['dataset'],'derivative_commit':manifest['revision'],
                            'archive':file.name,'source_line':line_number,
                            'source_filename':row.get('文件名',''),
                            'sha256_fulltext':full_hash,'primary_case_key_sha256':digest(key),
                            'fact_offset_in_fulltext':offset,
                            'source_transform':'Concatenate 段落[].内容 without altering characters',
                            'provider_reports_corruption_and_restoration':True},
                  'raw_record':{'ajjbqk':fact,'pjjg':judgment,'qw':text}})
            local['new_source_candidates']+=1
        counts.update(local);per_shard[file.name]=dict(local)
        print(json.dumps({'shard':file.name,'counts':dict(local)},ensure_ascii=False),flush=True)
    output.sort(key=lambda row:row['case_id'])
    temporary=destination.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in output));temporary.replace(destination)
    report={'source':manifest['dataset'],'revision':manifest['revision'],'files':selected,
            'counts':dict(counts),'output_records':len(output),'per_shard':per_shard,'qualified':0,
            'limitations':['Source publisher reports corruption and restoration; regex filtering cannot prove every character is correct.',
                           'Evidence sufficiency, actor scope, effective verdict and related proceedings are unverified.',
                           'Files are a fixed corpus shard sample, not a random sample of criminal cases.',
                           'All original and derived case text remains private local-only.']}
    (ROOT/'data/mnbvc_source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'qualified':0},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
