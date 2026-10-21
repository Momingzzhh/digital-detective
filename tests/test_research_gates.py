import copy
import hashlib
import json
from pathlib import Path

import pytest
from app import research
from app.five_dimensions import record_binding, fingerprint
from test_system import client, account


def fixture_row(split='dev'):
    # Synthetic mechanism fixture; never added to real-case data.
    quotes = [f'证人称甲与乙在第{i}个测试地点出现，以下为合成机制测试记录。' for i in range(6)]
    fact = '\n'.join(quotes); full = fact + '\n隐藏测试裁决，不能进入模型。'
    evidence = []; start = 0
    for i, text in enumerate(quotes):
        evidence.append({'id': f'E{i+1}', 'text': text, 'quote': text, 'kind': ['证言','书证','物证'][i%3],
                         'start': start, 'end': start+len(text)})
        start += len(text) + 1
    row = {'case_id':'fixture','group_id':'fixture-group','split':split,'status':'machine_annotated_unverified',
        'evidence':evidence,'source':{'sha256_fulltext':hashlib.sha256(full.encode()).hexdigest()},
        'source_fact_sha256':hashlib.sha256(fact.encode()).hexdigest(),
        'raw_record':{'qw':full,'ajjbqk':fact},
        'hidden_gold':{'charges':['合成测试罪名'],'perpetrators':['甲'],
            'chronology':[{'summary':quotes[0],'evidence_ids':['E1'],'sentence_start':0,'sentence_end':len(quotes[0])}]},
        'five_dimensions':{'fixture':True}}
    row['record_binding']=record_binding(row)
    return row


@pytest.fixture
def setup_research(client, tmp_path, monkeypatch):
    row=fixture_row(); monkeypatch.setattr(research,'ROOT',tmp_path)
    monkeypatch.setattr(research,'records',lambda:[row])
    monkeypatch.setattr(research,'model_binding',lambda:'fixed-test-model')
    monkeypatch.setattr(research,'acceptance_model_ready',lambda:True)
    monkeypatch.setenv('RESEARCH_REVIEW_ENABLED','true')
    (tmp_path/'data').mkdir()
    (tmp_path/'data/five_dimensions_report.json').write_text(json.dumps({'dataset_target_assembled':True}))
    headers=account(client)
    return client,headers,row,tmp_path


def approve(client, headers, row):
    return client.post('/api/research/items/fixture/review',headers=headers,json={
        'record_binding':row['record_binding'],'decision':'approve',
        'checks':{k:True for k in research.CHECKS},'notes':'仅供自动化机制测试的人工签署模拟，不计入真实复核。'})


def test_review_requires_complete_explicit_checks(setup_research):
    client,h,row,_=setup_research
    body={'record_binding':row['record_binding'],'decision':'approve','checks':{},'notes':'合成测试记录，不作为真实审核成果。'}
    assert client.post('/api/research/items/fixture/review',headers=h,json=body).status_code==422
    assert client.post('/api/research/items/fixture/session',headers=h,json={}).status_code==409
    assert approve(client,h,row).status_code==200


def test_human_session_only_receives_source_evidence_and_forbids_free_text(setup_research):
    client,h,row,_=setup_research; assert approve(client,h,row).status_code==200
    response=client.post('/api/research/items/fixture/session',headers=h,json={})
    assert response.status_code==200
    sid=response.json()['session_case_id']
    session=client.get('/api/cases/'+sid).json()
    assert len(session['evidence'])==2 and session['sources']==[]
    assert 'hidden_gold' not in session and '合成测试罪名' not in json.dumps(session,ensure_ascii=False)
    assert client.post('/api/cases/'+sid+'/messages',headers=h,json={'text':'未经来源支持的指纹'}).status_code==409
    assert client.post('/api/research/sessions/'+sid+'/release',headers=h,json={'evidence_ids':['NOT-REAL']}).status_code==422
    assert client.post('/api/research/sessions/'+sid+'/release',headers=h,json={'evidence_ids':['E3']}).status_code==200
    session=client.get('/api/cases/'+sid).json()
    assert len(session['evidence'])==3 and session['evidence'][2]['quote']==row['evidence'][2]['text']


def test_source_change_invalidates_review(setup_research):
    client,h,row,_=setup_research; assert approve(client,h,row).status_code==200
    row['raw_record']['qw']+='被替换的原文'
    assert client.get('/api/research/items/fixture').status_code==409


def test_changed_annotation_cannot_score_an_old_session(setup_research):
    from app.store import load_case, save_case
    client,h,row,_=setup_research
    assert approve(client,h,row).status_code==200
    sid=client.post('/api/research/items/fixture/session',headers=h,json={}).json()['session_case_id']
    me=client.get('/api/me').json()
    # Test mechanism only: no model or real human outcome is being simulated.
    with research.db() as con:
        user_id=con.execute('SELECT id FROM users WHERE name=?',(me['name'],)).fetchone()[0]
    case=load_case(user_id,sid);case['research']['analysis_attempts']=1;save_case(user_id,case)
    row['five_dimensions']['changed']=True;row['record_binding']=record_binding(row)
    assert approve(client,h,row).status_code==200
    response=client.post('/api/research/sessions/'+sid+'/score',headers=h,json={
        'chronology_all_correct':True,'no_unsupported_events':True,'notes':'旧版本会话必须被拒绝，不能用于本轮正式评分。'})
    assert response.status_code==409
    assert '数据版本' in response.json()['detail']


def test_test_set_stays_closed_until_full_validation_and_repeated_runs_forbidden(setup_research):
    client,h,row,root=setup_research
    row['split']='test';row['record_binding']=record_binding(row)
    assert approve(client,h,row).status_code==200
    assert client.post('/api/research/items/fixture/session',headers=h,json={}).status_code==409
    (root/'data/validation_acceptance_report.json').write_text(json.dumps({'passed':True,'model_binding':'fixed-test-model',
        'dataset_binding':fingerprint([row['record_binding']])}))
    first=client.post('/api/research/items/fixture/session',headers=h,json={})
    second=client.post('/api/research/items/fixture/session',headers=h,json={})
    assert first.status_code==second.status_code==200
    assert first.json()['session_case_id']==second.json()['session_case_id']
    assert second.json()['resumed'] is True


def test_missing_scores_remain_in_denominator_and_do_not_pass(monkeypatch):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import summarize_acceptance as summary
    rows=[{'case_id':str(i),'group_id':str(i),'record_binding':'b'+str(i),
        'split':'train' if i<420 else ('dev' if i<510 else 'test')} for i in range(600)]
    monkeypatch.setattr(summary,'approved',lambda row:True)
    scores={}
    for row in rows[420:490]:
        scores[('dev',row['case_id'])]={'record_binding':row['record_binding'],'model_binding':'m',
            'mode':'human_source_bound','reviewer_id':'synthetic-test','human_scored_at':'fixture',
            'session_sha256':'fixture','all_correct':True,'human_analysis_rounds':2}
    out=summary.summarize(rows,scores,'m',True)
    assert out['dev']['cases']==90 and out['dev']['missing_or_failed']==20
    assert out['dev']['whole_case_accuracy']==70/90
    assert not out['dev']['passed'] and not out['training_complete']
