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
           'logic_chain': [{'id': 'L1', 'claim': '测试1', 'evidence_ids': ['E1'], 'sentence_ids':['S1'],
                            'independently_supported':True,'dependency_reason':'','depends_on_steps': []},
                           {'id': 'L2', 'claim': '测试2', 'evidence_ids': ['E2'], 'sentence_ids':['S2'],
                            'independently_supported':False,'dependency_reason':'仅为合成机制依赖','depends_on_steps': ['L1']},
                           {'id': 'L3', 'claim': '测试3', 'evidence_ids': ['E3'], 'sentence_ids':['S3'],
                            'independently_supported':False,'dependency_reason':'仅为合成机制依赖','depends_on_steps': ['L2']}],
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


def test_conflict_resolution_cites_its_own_support_not_only_conflicting_sources():
    from app.five_dimensions import conclusion
    row, ann, spans = fixture()
    # Synthetic unresolved difference plus an independent checking record.
    ann['contradictions'] = [{'left_id': 'E1', 'right_id': 'E2',
        'left_quote': row['evidence'][0]['text'], 'right_quote': row['evidence'][1]['text'],
        'issue': '合成陈述差异', 'resolution': '依据第三份合成记录核对，仍保留未决状态',
        'resolution_evidence_ids': ['E3', 'E1']}]
    row['five_dimensions'] = validate_annotation(row, ann, spans)
    row['hidden_gold'] = ground_truth(row)
    target = conclusion(row)
    assert set(target['conflicts'][0]['evidence_ids']) == {'E1', 'E2', 'E3'}
    assert len(target['conflicts'][0]['evidence_ids']) == 3


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
    for index, step in enumerate(wire['logic_chain'], start=1):
        step['claim']=f'仅为机制测试主张{index}'
        step.pop('evidence_ids')
    result=canonical(schema_for(row).model_validate(wire).model_dump(),row)
    assert result['candidate_names']==['甲','乙']
    assert result['event_sentence_ids']==['S1','S2','S3']
    assert [step['evidence_ids'] for step in result['logic_chain']]==[['E1'],['E2'],['E3']]
    assert validate_annotation(row,result,spans)['reasoning_depth']==3


def test_compact_references_cannot_add_an_unrelated_evidence_list():
    from scripts.compact_annotation import schema_for
    from pydantic import ValidationError
    row,ann,_=fixture()
    wire={'t':{e['id']:1 for e in row['evidence']},'logic_chain':ann['logic_chain'],'c':[],
          **{k:ann[k] for k in ('actor_assessments','events','complete_actor_coverage','complete_event_coverage')}}
    with pytest.raises(ValidationError,match='evidence_ids'):
        schema_for(row).model_validate(wire)


def test_aliases_require_explicit_identity_mapping_not_cooccurrence():
    from app.five_dimensions import explicit_alias
    assert not explicit_alias('甲和乙到场作证。','甲','乙')
    assert explicit_alias('甲（绰号“小甲”）作了陈述。','甲','小甲')
    assert explicit_alias('甲，又名小甲，到场。','甲','小甲')


@pytest.mark.parametrize('text',[
    '其辨认出当事人甲就是“阿甲”。',
    '其辨认出“阿甲”就是当事人甲。',
    '其辨认出当事人甲就是负责引路的“阿甲”。',
    '其辨认出当事人甲就是该名男经理。',
])
def test_aliases_accept_explicit_source_identifications_in_either_direction(text):
    from app.five_dimensions import explicit_alias
    alias='男经理' if '男经理' in text else '阿甲'
    assert explicit_alias(text,'甲',alias)


@pytest.mark.parametrize('text',[
    '甲与阿甲一起到场。',
    '其未能辨认出甲就是阿甲。',
    '其辨认出甲不是阿甲。',
    '其辨认出甲，阿甲住在隔壁。',
    '其辨认出甲似乎是阿甲。',
])
def test_aliases_reject_unbound_or_uncertain_identifications(text):
    from app.five_dimensions import explicit_alias
    assert not explicit_alias(text,'甲','阿甲')


def test_multiple_explicit_alias_attributes_keep_each_identity_binding():
    from app.five_dimensions import explicit_alias
    text='当事人甲（曾用名乙、绰号“丙”，手机号隐去）的手机话单。甲（即丁）有多次通话。'
    assert all(explicit_alias(text,'甲',alias) for alias in ('乙','丙','丁'))
    assert explicit_alias('乙（曾用名甲、绰号“丙”）。','甲','乙')


@pytest.mark.parametrize('text',[
    '甲（同行者乙、绰号“丙”）到场。',
    '甲（曾用名可能是乙、绰号“丙”）到场。',
    '甲（不是乙、绰号“丙”）到场。',
    '甲（曾用名乙某、绰号“丙”）到场。',
    '甲（即乙或丙）到场。',
])
def test_parenthetical_attributes_do_not_manufacture_ambiguous_aliases(text):
    from app.five_dimensions import explicit_alias
    assert not explicit_alias(text,'甲','乙')


def test_independent_result_cannot_be_counted_as_a_third_dependency_level():
    row,ann,spans=fixture()
    ann['logic_chain'][2]['independently_supported']=True
    with pytest.raises(ValueError,match='Independent source facts'):
        validate_annotation(row,ann,spans)
    ann['logic_chain'][2]['depends_on_steps']=[]
    with pytest.raises(ValueError,match='three linked'):
        validate_annotation(row,ann,spans)


@pytest.mark.parametrize('defect',['unknown_sentence','wrong_evidence','duplicate_sentence','missing_reason'])
def test_logic_citations_and_dependency_explanations_are_checked(defect):
    row,ann,spans=fixture()
    if defect=='unknown_sentence':ann['logic_chain'][1]['sentence_ids']=['S-UNKNOWN']
    elif defect=='wrong_evidence':ann['logic_chain'][1]['sentence_ids']=['S1']
    elif defect=='duplicate_sentence':ann['logic_chain'][1]['sentence_ids']=['S2','S2']
    else:ann['logic_chain'][1]['dependency_reason']=''
    with pytest.raises(ValueError):validate_annotation(row,ann,spans)


def test_short_chain_can_be_recorded_honestly_but_is_not_admitted():
    from app.five_dimensions import FiveDimensions
    row,ann,spans=fixture();ann['logic_chain']=ann['logic_chain'][:2]
    assert len(FiveDimensions.model_validate(ann).logic_chain)==2
    with pytest.raises(ValueError,match='three linked'):
        validate_annotation(row,ann,spans)


def test_an_exact_source_restatement_cannot_become_a_dependent_inference():
    row,ann,spans=fixture()
    ann['logic_chain'][1]['claim']=spans[1]['text']
    with pytest.raises(ValueError,match='already explicitly stated'):
        validate_annotation(row,ann,spans)


def test_verbatim_facts_remain_valid_as_independent_nodes():
    from app.five_dimensions import require_nonverbatim_inference
    # This checks whole-sentence equality, never a truth/semantic score.
    text='甲称看到乙搬走物品。'
    require_nonverbatim_inference(text,True,[text])
    require_nonverbatim_inference('另有材料支持的关联主张',False,[text])


def test_a_conditional_consequence_is_not_mistaken_for_an_explicit_source_fact():
    from app.five_dimensions import require_nonverbatim_inference
    require_nonverbatim_inference('两笔记录属于同一交易。',False,
        ['如果账号、金额和发生时间均一致，两笔记录属于同一交易。'])
    with pytest.raises(ValueError,match='already explicitly stated'):
        require_nonverbatim_inference('两笔记录属于同一交易。',False,
            ['2、两笔记录属于同一交易。'])


def test_repeated_claim_cannot_inflate_graph_depth():
    row,ann,spans=fixture()
    ann['logic_chain'][2]['claim']=ann['logic_chain'][1]['claim']
    with pytest.raises(ValueError,match='Repeated logic claim'):
        validate_annotation(row,ann,spans)


def test_dependent_findings_cite_transitive_premises_without_mutating_the_graph():
    from app.five_dimensions import logic_support_ids, conclusion
    row, ann, spans = fixture()
    row['five_dimensions'] = validate_annotation(row, ann, spans)
    row['hidden_gold'] = ground_truth(row)
    original = copy.deepcopy(row['five_dimensions']['logic_chain'])
    support = logic_support_ids(original)
    assert support['L1'] == ['E1']
    assert set(support['L2']) == {'E1', 'E2'}
    assert set(support['L3']) == {'E1', 'E2', 'E3'}
    assert conclusion(row)['findings'][2]['evidence_ids'] == support['L3']
    assert row['five_dimensions']['logic_chain'] == original
    import json
    for example in training_examples(row):
        visible = set()
        for message in example['messages']:
            if message['role'] == 'user':
                payload = json.loads(message['content'])
                visible.update(e['id'] for e in payload.get('evidence', payload.get('additional_evidence', [])))
            elif message['role'] == 'assistant':
                for finding in json.loads(message['content'])['findings']:
                    step = next(s for s in original if s['claim'] == finding['claim'])
                    assert finding['evidence_ids'] == support[step['id']]
                    assert set(finding['evidence_ids']) <= visible


def test_single_source_examples_are_opt_in_and_never_assert_whole_case_accusations():
    import json
    row, ann, spans = fixture()
    row['five_dimensions'] = validate_annotation(row, ann, spans)
    row['hidden_gold'] = ground_truth(row)
    assert not any(e['variant'] == 'single_source_incomplete_case' for e in training_examples(row))
    examples = [e for e in training_examples(row, include_singletons=True)
                if e['variant'] == 'single_source_incomplete_case']
    assert len(examples) == len(row['evidence'])
    for e in examples:
        material = json.loads(e['messages'][1]['content'])['evidence']
        result = json.loads(e['messages'][-1]['content'])
        assert result['status'] == '证据不足' and not result['charges'] and not result['perpetrators']
        assert result['questions'] and len(material) == 1
        assert all(set(x['evidence_ids']) <= {material[0]['id']}
                   for key in ('findings', 'reconstruction', 'participants') for x in result[key])
