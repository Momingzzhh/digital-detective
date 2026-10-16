import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def certify():
    spec=importlib.util.spec_from_file_location('certify_dataset',ROOT/'scripts/certify_dataset.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def test_two_distinct_source_bound_human_reviews_required(tmp_path):
    c=certify()
    record={'case_id':'one','group_id':'group-one','split':'test','source':{'sha256_fulltext':'abc'},'evidence':[{'text':'source quote'}],
            'hidden_gold':{'charges':['盗窃罪']}}
    base={'case_id':'one','source_sha256':'abc','evidence_sha256':c.digest(record['evidence']),
          'gold_sha256':c.digest(record['hidden_gold']),'group_split_sha256':c.group_binding(record),
          'reviewed_at':'2026-10-15T12:00:00+08:00',
          'decision':'approve','checks':{k:True for k in c.CHECKS}}
    paths=[]
    for i in range(2):
        p=tmp_path/f'{i}.json';p.write_text(json.dumps({**base,'reviewer_id':f'reviewer-{i}'}));paths.append(p)
    assert not c.verify_reviews(record,paths[:1])
    assert c.verify_reviews(record,paths)
    tampered=json.loads(paths[1].read_text());tampered['gold_sha256']='wrong'
    paths[1].write_text(json.dumps(tampered))
    try:c.verify_reviews(record,paths)
    except ValueError:pass
    else:assert False,'A changed gold must invalidate attestations'
