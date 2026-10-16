import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def test_source_spans_preserve_every_character():
    b=module('build_benchmark')
    text=' 证人称：甲在场。\n乙没有出现；'+('长材料'*400)+'末尾'
    spans=b.source_spans(text,'F')
    assert ''.join(x['text'] for x in spans)==text
    assert all(text[x['start']:x['end']]==x['text'] for x in spans)


def test_visible_view_cannot_leak_gold_or_source():
    import json
    e=module('evaluate')
    record={'case_id':'secret-id','source':{'url':'secret-source'},'hidden_gold':{'charges':['secret-answer']},
            'review':{'reason':'secret-review'},'extraction':{'secret':'secret-extraction'},
            'evidence':[{'id':'E1','text':'证人称甲进入仓库。','quote':'被告人甲进入仓库。','kind':'证言','source':{'secret':'secret'}}]}
    view=e.visible_case(record)
    rendered=json.dumps(view,ensure_ascii=False)
    assert 'secret' not in rendered
    assert '被告人' not in rendered
    assert view['evidence'][0]['quote']=='证人称甲进入仓库。'


def test_wilson_interval_does_not_claim_ninety_with_one_success():
    e=module('evaluate')
    assert e.wilson(1,1)[0]<.9
    assert e.wilson(90,100)[0]<.9
    assert e.normalized(['盗窃罪',' 诈骗 '],True)=={'盗窃','诈骗'}


def test_target_claim_requires_thousand_reviewed_groups_and_full_test():
    from types import SimpleNamespace
    e=module('evaluate')
    args=SimpleNamespace(pilot=False,split='test',limit=None)
    rows=[{'group_id':str(i),'status':'independently_verified','audit':{'human_reviewed':True}}
          for i in range(1000)]
    assert e.eligible_for_target_claim(rows,args,150)
    rows[-1]['stratum']='england_historical_1834_1913'
    assert not e.eligible_for_target_claim(rows,args,150)
    rows[-1].pop('stratum')
    assert not e.eligible_for_target_claim(rows[:-1],args,150)
    assert not e.eligible_for_target_claim(rows,args,149)
    rows[-1]['audit']['human_reviewed']=False
    assert not e.eligible_for_target_claim(rows,args,150)
    rows[-1]['audit']['human_reviewed']=True
    args.limit=150
    assert not e.eligible_for_target_claim(rows,args,150)
