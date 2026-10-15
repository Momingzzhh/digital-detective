#!/usr/bin/env python3
"""Prepare source-backed *unreviewed* candidates, not a certified benchmark."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REV = 'b1a4b72b2050a4aa643e6360d5ff3a6b1fd221e3'
CASE = re.compile(r'[（(]\s*(?:19|20)\d{2}\s*[）)]\s*[\u4e00-\u9fffA-Za-z0-9０-９]{1,25}号')
DETAIL = re.compile(r'(?:证言|供述|鉴定意见|辨认笔录|勘验笔录|检验报告|监控录像)[，,:：。]')


def normalized(s):
    return re.sub(r'\s+', '', s).replace('（', '(').replace('）', ')')


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def fingerprint(s):
    # Fixed-position sampled character shingles: cheap near-duplicate heuristic.
    if not s:
        return 0
    positions = range(0, max(1, len(s)-7), max(1, len(s)//128))
    vals = [int.from_bytes(hashlib.blake2b(s[i:i+8].encode(), digest_size=8).digest(), 'big') for i in positions]
    return sum((sum(1 if v & (1 << bit) else -1 for v in vals) >= 0) << bit for bit in range(64))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=3000)
    ap.add_argument('--preserve-splits', action='store_true', help='Append candidates while freezing existing case/group split assignments')
    args = ap.parse_args()
    existing_path=ROOT/'data/processed/candidates.jsonl'
    existing={}
    if args.preserve_splits and existing_path.exists():
        existing={d['case_id']:d for d in map(json.loads,existing_path.open())}
        if args.limit < len(existing):
            ap.error('--preserve-splits cannot shrink the existing pool')
    if not args.preserve_splits and any((ROOT/'data/review').glob('*.json')):
        ap.error('Review checkpoints exist: use --preserve-splits to prevent reassignment')
    frozen=dict(existing)
    if args.preserve_splits:
        for path in (ROOT/'data/review').glob('*.json'):
            d=json.loads(path.read_text())
            if all(k in d for k in ('case_id','group_id','split')):
                prev=frozen.get(d['case_id'])
                if prev and (prev['group_id'],prev['split'])!=(d['group_id'],d['split']):
                    raise ValueError('Existing candidate/checkpoint assignment conflict: '+d['case_id'])
                frozen[d['case_id']]=d
    rows = {}
    for archive in sorted((ROOT/'data/raw/lecard').glob('candidates[12].zip')):
        with zipfile.ZipFile(archive) as z:
            for member in sorted(z.namelist()):
                if not member.endswith('.json'):
                    continue
                d = json.loads(z.read(member))
                key = d.get('writId') or digest(d['qw'])
                rows.setdefault(key, (d, archive.name, member))
    ids = sorted(rows)
    parent = {x:x for x in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    seen, buckets, fps, lengths = {}, defaultdict(list), {}, {}
    numbers = {}
    for key in ids:
        d = rows[key][0]
        facts = normalized(d.get('ajjbqk',''))
        numbers[key] = sorted(set(normalized(x) for x in CASE.findall(d['qw'])))
        # Old case numbers are only unique within a court; do not merge
        # every '(2014)刑初字第1号' nationwide. Scope all references by
        # the issuing court; cross-court appeals remain a review limitation.
        court_match = re.search(r'[\u4e00-\u9fff]{2,35}人民法院', d['qw'][:350])
        court = court_match.group() if court_match else 'unknown-court-'+key
        grouping_keys = ['case:'+court+':'+x for x in numbers[key]]
        grouping_keys += ['full:'+digest(normalized(d['qw']))]
        if len(facts) > 100:
            grouping_keys.append('facts:'+digest(facts))
        if d.get('ajId'):
            grouping_keys.append('ajId:'+d['ajId'])
        for g in grouping_keys:
            if g in seen:
                union(key,seen[g])
            else:
                seen[g]=key
        fp = fingerprint(facts)
        fps[key],lengths[key] = fp,len(facts)
        if len(facts) > 200:
            possible = set()
            for j in range(4):
                possible.update(buckets[(j,(fp >> (16*j)) & 65535)])
            for previous in possible:
                if min(lengths[previous],len(facts))/max(lengths[previous],len(facts)) > .9 and bin(fps[previous]^fp).count('1') <= 3:
                    union(key,previous)
            for j in range(4):
                buckets[(j,(fp >> (16*j)) & 65535)].append(key)
    scored = []
    for key in ids:
        d = rows[key][0]
        full,facts = d['qw'],d.get('ajjbqk','')
        detail = len(DETAIL.findall(full))
        charges = sorted(set(re.findall(r'犯([^，。；：\s]{1,25}罪)',d.get('pjjg',''))))
        # Provisional only: compound charges and multiple defendants require review.
        actors = sorted(set(re.findall(r'(?:被告人|上诉人(?:（原审被告人）)?)\s*([\u4e00-\u9fff·某×]{2,12}?)(?=犯)',d.get('pjjg',''))))
        if detail < 2 or len(facts) < 300 or not charges:
            continue
        preferred_length = 3000 <= len(facts) <= 15000
        score = (10000 if preferred_length else 0) + min(detail,30)*100 + min(len(facts),10000)//100 - max(0,len(facts)-15000)//10
        scored.append((score,key,charges,actors,detail))
    ordered=sorted(scored,key=lambda r:(-r[0],r[1]))
    if args.preserve_splits:
        scored_by_id={'lecard-'+r[1]:r for r in ordered}
        missing=set(frozen)-set(scored_by_id)
        if missing:
            raise ValueError('Cannot preserve source cases missing from screening: '+','.join(sorted(missing)[:5]))
        # Keep old order as well as assignments: --limit/resume remains meaningful.
        chosen=[scored_by_id[k] for k in existing]
        chosen_ids=set(existing)
        for k in frozen:
            if k not in chosen_ids:
                chosen.append(scored_by_id[k]); chosen_ids.add(k)
        if len(chosen)>args.limit:
            raise ValueError('Limit smaller than frozen candidates/checkpoints')
        chosen += [r for r in ordered if 'lecard-'+r[1] not in chosen_ids][:args.limit-len(chosen)]
    else:
        chosen = ordered[:args.limit]
    groups = defaultdict(list)
    for row in chosen:
        groups[find(row[1])].append(row)
    # Preserve old group identity/split; newly encountered groups get stable hashes.
    target = {'train':len(chosen)*.70,'dev':len(chosen)*.15,'test':len(chosen)*.15}
    counts=Counter(); assignment={}; group_names={}
    frozen_groups=defaultdict(set)
    for case_id,d in frozen.items():
        raw_id=case_id.removeprefix('lecard-')
        if raw_id not in parent:
            raise ValueError('Frozen case not found in raw corpus: '+case_id)
        frozen_groups[find(raw_id)].add((d['group_id'],d['split']))
    for group in sorted(groups,key=lambda g:(-len(groups[g]),digest(g))):
        old=frozen_groups.get(group,set())
        if len(old)>1:
            raise ValueError('New grouping merges incompatible frozen groups; manual adjudication required: '+group)
        if old:
            name,split=next(iter(old))
        elif args.preserve_splits:
            name='lecard-group-'+group
            fraction=int(digest(name)[:16],16)/2**64
            split='train' if fraction<.70 else ('dev' if fraction<.85 else 'test')
        else:
            name='lecard-group-'+group
            split=max(target,key=lambda part:target[part]-counts[part])
        group_names[group]=name
        assignment[group]=split
        counts[split]+=len(groups[group])
    out=ROOT/'data/processed'; out.mkdir(parents=True,exist_ok=True)
    with (out/'candidates.jsonl.tmp').open('w') as f:
        for score,key,charges,actors,detail in chosen:
            d,archive,member=rows[key]
            row={
                'case_id':'lecard-'+key, 'group_id':group_names[find(key)],
                'split':assignment[find(key)],'status':'candidate_unreviewed',
                'source':{'repository':'https://github.com/myx666/LeCaRD','commit':REV,
                          'archive':archive,'member':member,'writ_id':key,
                          'sha256_fulltext':digest(d['qw']),'case_numbers':numbers[key]},
                'raw_record':d,
                'provisional_gold':{'charges':charges,'actors':actors,
                                    'extraction_method':'regex_from_pjjg','verified':False},
                'screening':{'detailed_evidence_markers':detail,'score':score,
                             'evidence_sufficient':None,'leakage_reviewed':False,
                             'human_reviewed':False}
            }
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'candidates.jsonl.tmp').replace(out/'candidates.jsonl')
    group_sizes=Counter(find(k) for k in ids)
    report={
        'status':'candidate_pool_only_not_qualified_benchmark',
        'candidates':len(chosen),'qualified':0,'human_reviewed':0,
        'source_unique_documents':len(ids),'screening_passed_documents':len(scored),
        'split':dict(counts),'target_split':{'train':.70,'dev':.15,'test':.15},
        'groups_selected':len(groups),'source_groups':len(group_sizes),
        'largest_source_group':max(group_sizes.values(),default=0),
        'grouping':['writId','ajId','case number references scoped to issuing court',
                    'normalized fulltext exact hash','normalized facts exact hash',
                    'sampled fact shingle SimHash, Hamming <=3 and length ratio >0.9'],
        'limitations':[
            'These are source judgments selected by structural heuristics, not proven-solvable mysteries.',
            'No document has undergone human sufficient-evidence or ground-truth review.',
            'Raw fields contain outcomes and must NEVER be provided directly to a blind-test agent.',
            'Regex gold can omit actors or misparse compound charges; it is not trusted evaluation truth.',
            'Case numbers scoped to issuing court avoid nationwide short-number collisions; cross-court appeals/retrials, missing references and SimHash misses can still cross splits.',
            'Case grouping must be reviewed before a definitive held-out evaluation.',
            'Public source documents may already exist in model pretraining.',
            'Raw and processed texts are local-only and must not be committed to a public repository.'
        ]
    }
    report_path=ROOT/'data/dataset_report.json'
    if args.preserve_splits and report_path.exists():
        previous=json.loads(report_path.read_text())
        # Candidate expansion must not reset ongoing machine/human review counts.
        for key in ('status','qualified','human_reviewed'):
            report.pop(key,None)
        previous.update(report)
        report=previous
    report['split_assignment_mode']='frozen_existing_stable_hash_new' if args.preserve_splits else 'initial_group_deficit_allocation'
    report['preserved_cases']=len(frozen)
    temporary=report_path.with_suffix('.json.prepare.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(report_path)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
