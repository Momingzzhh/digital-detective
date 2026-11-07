import json
import pytest
from scripts import screen_training_complexity as screen


def fixture():
    evidence=[{'id':f'E{i}','kind':['证人证言','物证','书证'][i%3],'text':f'合成机制测试材料{i}。'} for i in range(1,7)]
    row={'case_id':'synthetic','group_id':'synthetic-group','split':'train','source_binding':'source',
        'evidence':evidence,'sentence_spans':[{'id':f'E{i}:S1','evidence_id':f'E{i}',
            'text':evidence[i-1]['text'],'start':0,'end':len(evidence[i-1]['text'])} for i in range(1,7)]}
    value={'source_sufficient':True,'real_conflict_or_multiple_behavior_actors':True,
        'steps':[{'id':f'L{i}','claim':f'仅为合成测试主张{i}','sentence_ids':[f'E{i}:S1'],
            'independent':i==1,'parents':[] if i==1 else [{'id':f'L{i-1}','necessary':True,'contribution':'仅为合成依赖机制'}]} for i in range(1,4)],
        'gaps':[],'reason':'仅为合成机制核验，不能证明实际推理必要。'}
    return row,value


def test_mechanical_source_candidate_never_becomes_a_human_or_formal_approval():
    row,value=fixture();result=screen.validate_screen(row,value)
    assert result['graph_depth']==3 and result['machine_candidate'] is True
    assert result['human_reviewed'] is False and result['formal_approval'] is False
    assert result['qualified_cases_added']==0


@pytest.mark.parametrize('defect',['heldout','independent','cycle','unknown','repeated','offset'])
def test_source_screen_cannot_pass_an_inconsistent_candidate(defect):
    row,value=fixture()
    if defect=='heldout':row['split']='test'
    elif defect=='independent':value['steps'][1]['independent']=True
    elif defect=='cycle':value['steps'][0]['parents']=[{'id':'L3','necessary':True,'contribution':'合成错误前提'}]
    elif defect=='unknown':value['steps'][1]['sentence_ids']=['NOT-A-SENTENCE']
    elif defect=='repeated':value['steps'][1]['claim']=value['steps'][0]['claim']
    else:row['sentence_spans'][0]['text']='不匹配的合成引句'
    with pytest.raises(ValueError):screen.validate_screen(row,value)


def test_shallow_rejection_can_be_saved_without_manufacturing_a_three_step_graph():
    row,value=fixture();value.update(steps=[],reason='仅为合成短链拒绝机制。')
    result=screen.validate_screen(row,value)
    assert result['graph_depth']==0 and not result['machine_candidate']


@pytest.mark.parametrize('defect',['short','gap','unnecessary'])
def test_status_is_conservatively_derived_instead_of_asking_for_a_second_verdict(defect):
    row,value=fixture()
    if defect=='short':value['steps']=value['steps'][:2]
    elif defect=='gap':value['gaps']=['仅为合成来源缺口']
    else:value['steps'][1]['parents'][0]['necessary']=False
    result=screen.validate_screen(row,value)
    assert not result['machine_candidate'] and not result['formal_approval']
    assert result['decision']==('too_shallow' if defect=='short' else 'source_gap')


def test_generation_schema_enforces_root_and_parent_direction():
    row,value=fixture();schema=screen.schema_for(row)
    value['steps'][0]['independent']=False
    with pytest.raises(ValueError):schema.model_validate(value)
    value['steps'][0]['independent']=True
    value['steps'][1]['parents'][0]['id']='L3'
    with pytest.raises(ValueError):schema.model_validate(value)


def test_canonical_order_cannot_invent_a_missing_parent():
    row,value=fixture();value['steps'].reverse()
    result=screen.validate_screen(row,value)
    assert [s['id'] for s in result['steps']]==['L1','L2','L3']
    value['steps']=[s for s in value['steps'] if s['id']!='L1']
    with pytest.raises(ValueError,match='earlier nodes'):screen.validate_screen(row,value)


def test_exact_quoted_facts_cannot_be_strung_into_a_false_three_level_candidate():
    row,value=fixture()
    value['steps'][1]['claim']=row['sentence_spans'][1]['text']
    with pytest.raises(ValueError,match='already explicitly stated'):
        screen.validate_screen(row,value)


def test_cohort_cannot_move_a_heldout_or_changed_source_into_training(tmp_path,monkeypatch):
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    row,_=fixture();folder=tmp_path/'data/processed/five_dimensions_v2';folder.mkdir(parents=True)
    pool=folder/'candidates.private.jsonl';pool.write_text(json.dumps(row)+'\n')
    manifest={'protocol':screen.PROTOCOL,'selection':'synthetic mechanism only','reference_binding':'synthetic',
        'cases':[{k:row[k] for k in ('case_id','group_id','split','source_binding')}]}
    manifest['cohort_binding']=screen.fingerprint(manifest)
    path=tmp_path/'cohort.json';path.write_text(json.dumps(manifest))
    assert screen.load_cohort(path)[1][0]['case_id']=='synthetic'
    for mutation in ({'split':'test'},{'source_binding':'changed'}):
        pool.write_text(json.dumps(row|mutation)+'\n')
        with pytest.raises(ValueError,match='frozen training sources'):screen.load_cohort(path)


def test_reference_example_is_train_only_and_cannot_be_its_own_target():
    row,value=fixture()
    reference={'case_id':'other-synthetic','split':'train','source_evidence':row['evidence'],
        'sentence_spans':row['sentence_spans'],'expected_screen':value|{'decision':'candidate'}}
    messages,_=screen.requests_for(row,reference)
    assert len(messages)==4 and 'hidden_gold' not in str(messages)
    with pytest.raises(ValueError,match='training-only'):screen.requests_for(row|{'split':'test'},reference)
    with pytest.raises(ValueError,match='own diagnostic target'):
        screen.requests_for(row,reference|{'case_id':row['case_id']})


def test_reference_acceptance_does_not_upgrade_two_levels_to_a_complex_case(tmp_path,monkeypatch):
    monkeypatch.setattr(screen,'ROOT',tmp_path)
    row,value=fixture();value.update(decision='too_shallow',steps=value['steps'][:2])
    relative='data/review/five_dimensions_v2_calibration_revision/synthetic/draft.json'
    path=tmp_path/relative;path.parent.mkdir(parents=True)
    draft={k:row[k] for k in ('case_id','source_binding')}|{'human_reviewed_maximum_depth':2}
    path.write_text(json.dumps(draft))
    reference={'case_id':row['case_id'],'split':'train','source_binding':row['source_binding'],
        'formal_approval':False,'maximum_reviewed_depth':2,'corrected_draft':relative,
        'corrected_draft_binding':screen.fingerprint(draft),'source_evidence':row['evidence'],
        'sentence_spans':row['sentence_spans'],'expected_screen':value,
        'user_acceptance':{'source':'human_user_message_in_this_chat','quote':'synthetic mechanism acceptance only'}}
    reference['reference_binding']=screen.fingerprint(reference)
    p=tmp_path/'reference.json';p.write_text(json.dumps(reference))
    assert screen.load_reference(p,{'reference_binding':reference['reference_binding']})['formal_approval'] is False
    reference['expected_screen']['decision']='candidate'
    reference['reference_binding']=screen.fingerprint({k:v for k,v in reference.items() if k!='reference_binding'})
    p.write_text(json.dumps(reference))
    with pytest.raises(ValueError):screen.load_reference(p,{'reference_binding':reference['reference_binding']})
