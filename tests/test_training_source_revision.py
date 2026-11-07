import copy
import hashlib
import json

import pytest

from app.five_dimensions import fingerprint
from scripts import prepare_training_source_revision as revision
from scripts.build_v3 import extract_items, sentences


def fixture():
    texts = [f'{i}、证人甲的证言：仅为机制测试，甲看见乙将登记物品搬上车辆，随后驾车离开，现场有完整的登记和交接记录。'
             for i in range(1,8)]
    fact = '下列证据证实：' + ' '.join(texts)
    items = extract_items(fact, '合成测试罪')
    assert len(items) == 7
    sha = hashlib.sha256(fact.encode()).hexdigest()
    row = {'case_id':'synthetic','group_id':'synthetic-group','split':'train',
           'source':{'sha256_fulltext':sha},'source_fact_sha256':sha,
           'raw_record':{'qw':fact,'ajjbqk':fact},'hidden_gold':{'charges':['合成测试罪']},
           'evidence':items[:6], 'sentence_spans':sentences(items[:6])}
    row['source_binding'] = fingerprint({k:row[k] for k in revision.SOURCE_KEYS})
    plan = {'case_id':row['case_id'],'parent_source_binding':row['source_binding'],
            'append_ranges':[{'start':items[6]['start'],'end':items[6]['end'],
                              'kind':'证人证言','reason':'合成补充机制'}]}
    return row, plan


def test_supplement_preserves_every_existing_item_and_creates_no_approval():
    row, plan = fixture(); before = copy.deepcopy(row)
    out = revision.revise(row, plan)
    assert row == before and out['evidence'][:6] == row['evidence']
    assert out['source_binding'] != row['source_binding']
    assert out['group_id'] == row['group_id'] and out['split'] == 'train'
    assert out['formal_approval'] is False and out['human_reviewed'] is False
    assert out['training_source_revision']['qualified_cases_added'] == 0


@pytest.mark.parametrize('defect',['heldout','parent','offset','overlap','cross_item','kind','court'])
def test_supplement_cannot_move_sets_or_invent_source_fragments(defect):
    row, plan = fixture(); a = plan['append_ranges'][0]
    if defect == 'heldout': row['split'] = 'test'
    elif defect == 'parent': plan['parent_source_binding'] = 'other'
    elif defect == 'offset': a['start'] += 1
    elif defect == 'overlap': a.update(start=row['evidence'][0]['start'],end=row['evidence'][0]['end'])
    elif defect == 'cross_item': a.update(start=row['evidence'][0]['start'])
    elif defect == 'kind': a['kind'] = '物证'
    else:
        row['evidence'][0]['text'] += '本院综合分析评判如下：'
    with pytest.raises(ValueError): revision.revise(row,plan)


def test_confirmed_human_hold_cannot_be_bypassed_by_assistant_supplement(monkeypatch):
    row, plan = fixture()
    monkeypatch.setattr(revision,'source_review_hold',lambda _:True)
    with pytest.raises(ValueError,match='human-feedback revision workflow'):
        revision.revise(row,plan)


def test_current_answer_label_cannot_be_added_through_source_supplement():
    row,plan=fixture()
    row['hidden_gold']['charges']=['登记物品']  # Synthetic answer label only.
    with pytest.raises(ValueError,match='Target charge'):
        revision.revise(row,plan)


def test_revision_loader_rechecks_original_source_plan_and_output(tmp_path):
    row, plan = fixture()
    def write(name, value, jsonl=False):
        path=tmp_path/name;path.write_text(json.dumps(value)+ ('\n' if jsonl else ''))
        return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest={'protocol':revision.PROTOCOL,'qualified_cases_added':0,
              'parent_pool':write('pool.jsonl',row,True),
              'plan':write('plan.json',{'cases':[plan]}),
              'dataset':write('data.jsonl',revision.revise(row,plan),True)}
    entry=write('manifest.json',manifest)
    assert revision.load_revision(tmp_path,entry)[0]['split']=='train'
    (tmp_path/'plan.json').write_text('{}')
    with pytest.raises(ValueError,match='provenance file changed'):
        revision.load_revision(tmp_path,entry)


def nested_report_fixture():
    fact=('下列证据证实：1、鉴定意见：本段仅为合成机制材料，记录检验所见、检材编号和相关观察，不能用于真实案件判断。'
          '论证：1、第一项记录说明检验过程及结果；2、第二项分析说明相关观察的含义；3、第三项分析保留完整的最终结论。'
          ' 2、证人乙的证言：这是一份独立来源的合成陈述，证人记录了现场具体观察，不属于上一份报告的内部分析。')
    start=len('下列证据证实：');old_end=fact.index('；2、');end=fact.index(' 2、证人乙')
    quote=fact[start:old_end]
    e={'id':'E1','kind':'鉴定意见','start':start,'end':old_end,'text':quote,'quote':quote}
    sha=hashlib.sha256(fact.encode()).hexdigest()
    row={'case_id':'synthetic-nested','group_id':'synthetic-nested-group','split':'train',
         'source':{'sha256_fulltext':sha},'source_fact_sha256':sha,
         'raw_record':{'qw':fact,'ajjbqk':fact},'hidden_gold':{'charges':['合成测试罪']},
         'evidence':[e],'sentence_spans':sentences([e])}
    row['source_binding']=fingerprint({k:row[k] for k in revision.SOURCE_KEYS})
    marker=revision.START.search(fact)
    starts=[marker.end()+m.start() for m in revision.ITEM.finditer(fact[marker.end():])]
    plan={'case_id':row['case_id'],'parent_source_binding':row['source_binding'],'append_ranges':[],
          'extend_ranges':[{'evidence_id':'E1','end':end,
              'nested_item_starts':[s for s in starts if old_end<=s<end],'reason':'合成报告内部编号恢复'}]}
    return row,plan


def test_explicit_extension_restores_nested_report_without_losing_old_text():
    row,plan=nested_report_fixture();out=revision.revise(row,plan)
    assert out['evidence'][0]['quote'].startswith(row['evidence'][0]['quote'])
    assert '最终结论' in out['evidence'][0]['quote']
    assert '证人乙' not in out['evidence'][0]['quote']
    assert row['evidence'][0]['end'] < out['evidence'][0]['end']
    assert out['source_binding'] != row['source_binding'] and not out['formal_approval']


def test_extension_requires_declared_nested_markers():
    row,plan=nested_report_fixture();plan['extend_ranges'][0]['nested_item_starts']=[]
    with pytest.raises(ValueError,match='explicitly declared'):
        revision.revise(row,plan)


def test_extension_cannot_absorb_an_independent_source_that_was_not_previously_selected():
    row,plan=nested_report_fixture();fact=row['raw_record']['ajjbqk'];a=plan['extend_ranges'][0]
    a['end']=len(fact)
    marker=revision.START.search(fact)
    a['nested_item_starts']=[marker.end()+m.start() for m in revision.ITEM.finditer(fact[marker.end():])
                            if row['evidence'][0]['end']<=marker.end()+m.start()<len(fact)]
    with pytest.raises(ValueError,match='recognized source document'):
        revision.revise(row,plan)


def type_correction_fixture():
    texts = [f'{i}、证人甲的证言：仅为合成机制测试，甲看见乙搬走登记物品，现场记录了车辆、时间和后续经过。'
             for i in range(1, 7)]
    texts.append('7、某公安局物证鉴定室鉴定书，证实送检的合成检材编号和检测结果相互对应，本段仅为类型识别机制测试。')
    fact = '下列证据证实：' + ' '.join(texts)
    items = extract_items(fact, '合成测试罪')
    assert len(items) == 7 and items[-1]['kind'] == '鉴定意见'
    items[-1]['kind'] = '物证'  # Simulate the old issuing-institution error.
    sha = hashlib.sha256(fact.encode()).hexdigest()
    row = {'case_id': 'synthetic-type', 'group_id': 'synthetic-type-group', 'split': 'train',
           'source': {'sha256_fulltext': sha}, 'source_fact_sha256': sha,
           'raw_record': {'qw': fact, 'ajjbqk': fact}, 'hidden_gold': {'charges': ['合成测试罪']},
           'evidence': items, 'sentence_spans': sentences(items)}
    row['source_binding'] = fingerprint({k: row[k] for k in revision.SOURCE_KEYS})
    plan = {'case_id': row['case_id'], 'parent_source_binding': row['source_binding'], 'append_ranges': [],
            'reclassify_items': [{'evidence_id': 'E7', 'previous_kind': '物证',
                                  'kind': '鉴定意见', 'reason': '机构名称中的物证不是文书类型'}]}
    return row, plan


def test_type_only_revision_preserves_quotes_offsets_sentences_and_original_pool():
    row, plan = type_correction_fixture(); before = copy.deepcopy(row)
    out = revision.revise(row, plan)
    assert row == before and out['sentence_spans'] == row['sentence_spans']
    assert out['evidence'][:-1] == row['evidence'][:-1]
    assert out['evidence'][-1] == row['evidence'][-1] | {'kind': '鉴定意见'}
    assert out['source_binding'] != row['source_binding']
    assert not out['human_reviewed'] and not out['formal_approval']


@pytest.mark.parametrize('defect', ['previous_type', 'new_type', 'reason', 'repeated_item', 'heldout'])
def test_type_plan_cannot_relabel_without_a_matching_source_title(defect):
    row, plan = type_correction_fixture(); action = plan['reclassify_items'][0]
    if defect == 'previous_type': action['previous_kind'] = '书证'
    elif defect == 'new_type': action['kind'] = '证人证言'
    elif defect == 'reason': action['reason'] = ''
    elif defect == 'repeated_item': plan['reclassify_items'] *= 2
    else: row['split'] = 'test'
    with pytest.raises(ValueError): revision.revise(row, plan)


def judicial_tail_fixture(tail='上述事实清楚，证据充分，'):
    fact = '下列证据证实：' + ' '.join(
        f'{i}、证人甲的证言：仅为合成机制测试，甲看见乙搬走登记物品，现场记录了车辆、时间和后续经过。'
        for i in range(1, 7)) + tail
    items = extract_items(fact, '合成测试罪')
    # Reproduce the old extractor's untrimmed last evidence item.
    items[-1].update(end=len(fact), quote=fact[items[-1]['start']:], text=fact[items[-1]['start']:])
    sha = hashlib.sha256(fact.encode()).hexdigest()
    row = {'case_id': 'synthetic-tail', 'group_id': 'synthetic-tail-group', 'split': 'train',
           'source': {'sha256_fulltext': sha}, 'source_fact_sha256': sha,
           'raw_record': {'qw': fact, 'ajjbqk': fact}, 'hidden_gold': {'charges': ['合成测试罪']},
           'evidence': items, 'sentence_spans': sentences(items)}
    row['source_binding'] = fingerprint({k: row[k] for k in revision.SOURCE_KEYS})
    plan = {'case_id': row['case_id'], 'parent_source_binding': row['source_binding'], 'append_ranges': [],
            'trim_judicial_tails': [{'evidence_id': items[-1]['id'], 'previous_end': len(fact),
                                    'end': len(fact) - len(tail), 'reason': '移除已识别的司法结语，保留原始来源'}]}
    return row, plan


def test_judicial_trim_only_removes_known_tail_and_preserves_all_evidence_ids_and_raw_record():
    row, plan = judicial_tail_fixture(); before = copy.deepcopy(row)
    out = revision.revise(row, plan)
    assert row == before and out['raw_record'] == row['raw_record']
    assert out['evidence'][:-1] == row['evidence'][:-1]
    assert [e['id'] for e in out['evidence']] == [e['id'] for e in row['evidence']]
    assert row['evidence'][-1]['quote'].startswith(out['evidence'][-1]['quote'])
    assert out['training_source_revision']['changes'][0]['removed_characters'] == 12
    assert out['human_reviewed'] is False


@pytest.mark.parametrize('defect', ['before_boundary', 'after_boundary', 'previous_end', 'reason', 'duplicate', 'unrecognized'])
def test_source_trim_cannot_delete_arbitrary_original_claims(defect):
    row, plan = judicial_tail_fixture('证人又补充了一段具体观察。' if defect == 'unrecognized' else '上述事实清楚，证据充分，')
    action = plan['trim_judicial_tails'][0]
    if defect == 'before_boundary': action['end'] -= 3
    elif defect == 'after_boundary': action['end'] += 3
    elif defect == 'previous_end': action['previous_end'] -= 1
    elif defect == 'reason': action['reason'] = ''
    elif defect == 'duplicate': plan['trim_judicial_tails'] *= 2
    with pytest.raises(ValueError, match='boundary|distinct existing'):
        revision.revise(row, plan)
