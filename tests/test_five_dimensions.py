import copy
import pytest
from app.five_dimensions import validate_annotation, training_examples, validate_splits, ground_truth


def fixture():
    # Mechanism fixture only, never counted as a real criminal case.
    evidence = [{'id': f'E{i}', 'kind': ['证言','物证','书证'][i % 3], 'text': '甲称看到乙。合成测试片段。'} for i in range(1,7)]
    row = {'case_id': 'fixture', 'group_id': 'group', 'split': 'train', 'evidence': evidence,
           'hidden_gold': {'charges': ['测试罪名'], 'perpetrators': ['甲'],
                           'chronology': [{'summary': '合成测试片段。', 'evidence_ids': ['E1']}]}}
    ann = {'evidence_hierarchy': [{'evidence_id': e['id'], 'tier': '间接印证', 'reason': '机制测试'} for e in evidence],
           'logic_chain': [{'id': 'L1', 'claim': '测试1', 'evidence_ids': ['E1'], 'depends_on_steps': []},
                           {'id': 'L2', 'claim': '测试2', 'evidence_ids': ['E2'], 'depends_on_steps': ['L1']},
                           {'id': 'L3', 'claim': '测试3', 'evidence_ids': ['E3'], 'depends_on_steps': ['L2']}],
           'contradictions': [], 'candidate_names': ['甲','乙'], 'event_sentence_ids': ['S1','S2','S3'],
           'actor_assessments':[{'name':n,'aliases':[],'participation':role,'role':'仅为机制测试描述',
                                 'sentence_ids':['S1']} for n,role in [('甲','直接实施'),('乙','帮助配合')]],
           'events':[{'claim':'仅为机制测试经过','sentence_ids':[f'S{i}']} for i in range(1,4)],
           'complete_actor_coverage':True,'complete_event_coverage':True}
    spans = [{'id': f'S{i}','evidence_id':f'E{i}','text':evidence[i-1]['text'],
              'start':0,'end':len(evidence[i-1]['text'])} for i in range(1,4)]
    row['sentence_spans']=spans
    return row, ann, spans


def test_five_dimensions_derive_graph_depth_and_keep_round_types_separate():
    row, ann, spans = fixture()
    out = validate_annotation(row, ann, spans)
    assert out['reasoning_depth'] == 3
    assert out['dialogue_rounds']['observed_human'] is None
    row['five_dimensions'] = out
    row['hidden_gold'] = ground_truth(row)
    examples = training_examples(row)
    simulated = next(r for r in examples if r['variant'] == 'source_bound_simulated_dialogue')
    assert simulated['simulated_rounds'] == 3
    import json
    partial = json.loads(simulated['messages'][4]['content'])
    assert partial['status']=='证据不足' and not partial['charges'] and not partial['perpetrators']
    assert partial['findings'] and all(set(f['evidence_ids']) <= {'E1','E2','E3'} for f in partial['findings'])
    withheld = next(r for r in examples if r['variant'] == 'all_evidence_withheld')
    assert '测试罪名' not in str(withheld)
    assert all(e['group_id'] == 'group' for e in examples)


@pytest.mark.parametrize('defect', ['missing_tier','cycle','unknown_reference','one_candidate','fake_quote','shallow'])
def test_invalid_complexity_and_fabricated_support_rejected(defect):
    row, ann, spans = fixture()
    if defect == 'missing_tier': ann['evidence_hierarchy'].pop()
    elif defect == 'cycle': ann['logic_chain'][0]['depends_on_steps'] = ['L3']
    elif defect == 'unknown_reference': ann['logic_chain'][0]['evidence_ids'] = ['NOT-REAL']
    elif defect == 'one_candidate': ann['candidate_names'] = ['甲']
    elif defect == 'shallow':
        for step in ann['logic_chain']: step['depends_on_steps'] = []
    else:
        ann['contradictions'] = [{'left_id': 'E1', 'right_id': 'E2', 'left_quote': '虚构指纹',
            'right_quote': '乙', 'issue': '测试', 'resolution': '未解决', 'resolution_evidence_ids': []}]
    with pytest.raises(ValueError): validate_annotation(row, ann, spans)


def test_heldout_groups_cannot_generate_training_examples():
    row, ann, spans = fixture(); row['five_dimensions'] = validate_annotation(row, ann, spans)
    for split in ('dev','test'):
        row['split'] = split
        with pytest.raises(ValueError): training_examples(row)


def test_identifier_repair_cannot_manufacture_a_second_person():
    from scripts.annotate_five_dimensions import normalize_references
    row,ann,spans=fixture()
    ann['candidate_names']=['甲','甲']
    normalized,repairs=normalize_references(ann,[])
    assert normalized['candidate_names']==['甲']
    assert repairs[-1]['rule']=='exact_duplicate_name_removal'
    with pytest.raises(ValueError,match='multiple investigated persons'):
        validate_annotation(row,normalized,spans)


def test_exact_split_and_related_groups():
    rows = [{'case_id': str(i), 'group_id': str(i), 'split': 'train' if i < 420 else ('dev' if i < 510 else 'test')} for i in range(600)]
    assert validate_splits(rows, True) == {'train':420,'dev':90,'test':90}
    broken=copy.deepcopy(rows);broken[-1]['group_id']='0'
    with pytest.raises(ValueError):validate_splits(broken, True)
    broken=copy.deepcopy(rows);broken[-1]['split']='dev'
    with pytest.raises(ValueError):validate_splits(broken, True)


@pytest.mark.parametrize('defect',['incomplete_actors','incomplete_events','offset','alias','event_index','role_person'])
def test_full_case_support_is_required(defect):
    row,ann,spans=fixture()
    if defect=='incomplete_actors':ann['complete_actor_coverage']=False
    elif defect=='incomplete_events':ann['complete_event_coverage']=False
    elif defect=='offset':spans[0]['end']-=1
    elif defect=='alias':ann['actor_assessments'][0]['aliases']=['乙']
    elif defect=='event_index':ann['event_sentence_ids']=['S1','S2']
    else:ann['actor_assessments'][0]['name']='丙'
    with pytest.raises(ValueError):validate_annotation(row,ann,spans)


def test_full_case_roles_are_separate_from_current_judgment_names():
    row,ann,spans=fixture();row['five_dimensions']=validate_annotation(row,ann,spans)
    gold=ground_truth(row)
    assert gold['adjudicated_perpetrators']==['甲']
    assert gold['perpetrators']==['乙','甲']
    assert gold['actor_roles'][1]['participation']=='帮助配合'
    assert gold['chronology'][0]['supporting_sentences'][0]['text']==spans[0]['text']


def test_role_identity_requires_cited_sentence_or_explicit_statement_speaker():
    row,ann,spans=fixture()
    row['evidence'][0]['text']='甲描述独立事项。稍后在另一件事中提到乙。'
    spans[0].update(text='甲描述独立事项。',start=0,end=len('甲描述独立事项。'))
    with pytest.raises(ValueError,match='Role support'):
        validate_annotation(row,ann,spans)
    row['evidence'][0]['text']='1.当事人甲供述：我搬走了物品。'
    start=row['evidence'][0]['text'].index('我')
    spans[0].update(text='我搬走了物品。',start=start,end=len(row['evidence'][0]['text']))
    ann['actor_assessments'][1]['sentence_ids']=['S2']
    assert validate_annotation(row,ann,spans)['reasoning_depth']==3


def test_compact_wire_derives_indexes_without_inventing_names_or_quotes():
    from scripts.compact_annotation import canonical,schema_for
    row,ann,spans=fixture()
    wire={'t':{e['id']:1 for e in row['evidence']},'logic_chain':ann['logic_chain'],'c':[],
          **{k:ann[k] for k in ('actor_assessments','events','complete_actor_coverage','complete_event_coverage')}}
    # Short claims in this fixture are lengthened to satisfy the wire schema.
    for step in wire['logic_chain']:step['claim']='仅为机制测试主张'
    result=canonical(schema_for(row).model_validate(wire).model_dump(),row)
    assert result['candidate_names']==['甲','乙']
    assert result['event_sentence_ids']==['S1','S2','S3']
    assert validate_annotation(row,result,spans)['reasoning_depth']==3


def test_aliases_require_explicit_identity_mapping_not_cooccurrence():
    from app.five_dimensions import explicit_alias
    assert not explicit_alias('甲和乙到场作证。','甲','乙')
    assert explicit_alias('甲（绰号“小甲”）作了陈述。','甲','小甲')
    assert explicit_alias('甲，又名小甲，到场。','甲','小甲')
