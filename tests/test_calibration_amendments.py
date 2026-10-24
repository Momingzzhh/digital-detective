import copy
import pytest
from scripts.prepare_calibration_amendments import revise
from scripts.import_word_calibration import fingerprint


def fixture():
    quote='1、证人甲的证言：我看见乙把仓库电脑装车离开，后来收到丙发来的付款记录，具体金额需要核对。'
    bank=' 2、银行交易明细：2017年2月1日乙的账户收到丙转入的人民币两万元，随后转出一万元。'
    fact='经审理查明。下列证据证实： '+quote+bank
    full='首页。'+fact+'本院认为：留作隐藏。'
    row={'case_id':'synthetic','group_id':'g','split':'train','raw_record':{'qw':full,'ajjbqk':fact},
        'source':{'sha256_fulltext':__import__('hashlib').sha256(full.encode()).hexdigest()},
        'source_fact_sha256':__import__('hashlib').sha256(fact.encode()).hexdigest(),
        'source_binding':'original','evidence':[],'hidden_gold':{}}
    plan={'source_binding_before':'original','annotation_result_binding':'ann','changes':[
        {'source_item':2,'kind':'银行记录','reason':'仅为测试的具体记录补充'}]}
    feedback=[{'case_id':'synthetic','human_attestation':True,'source_binding':'original','annotation_result_binding':'ann'}]
    return row,plan,feedback


def test_source_revision_has_exact_offsets_and_never_changes_parent():
    row,plan,feedback=fixture();before=copy.deepcopy(row)
    revised=revise(row,plan,feedback)
    assert row==before and revised['source_binding']!='original'
    item=revised['evidence'][0]
    assert row['raw_record']['ajjbqk'][item['start']:item['end']]==item['quote']
    assert revised['human_reviewed'] is False and revised['split']=='train'


@pytest.mark.parametrize('defect',['heldout','unsigned','different_source','different_annotation'])
def test_revision_requires_matching_training_feedback(defect):
    row,plan,feedback=fixture()
    if defect=='heldout':row['split']='test'
    elif defect=='unsigned':feedback[0]['human_attestation']=False
    elif defect=='different_source':feedback[0]['source_binding']='changed'
    else:feedback[0]['annotation_result_binding']='changed'
    with pytest.raises(ValueError):revise(row,plan,feedback)


def test_source_policy_rejects_hidden_label_in_filename_and_judicial_shortcuts():
    from app.source_policy import require_clean_evidence
    for text in ('将数据存入张三涉嫌测试犯罪案.rar文件。','当事人对上述认定事实所作的一致供述。',
                 '经审理查明：甲实施了相关行为。'):
        with pytest.raises(ValueError):require_clean_evidence([{'text':text}])
    require_clean_evidence([{'text':'将采集到的交易流水保存到数据.zip，并核对日期金额。'}])


def test_reannotation_rejects_heldout_parent_even_if_child_claims_train():
    from scripts.annotate_calibration_revision import check_revision_rows
    row,plan,feedback=fixture();child=revise(row,plan,feedback)
    parent=copy.deepcopy(row);parent['split']='test'
    with pytest.raises(ValueError,match='training groups'):
        check_revision_rows([child],{'synthetic':parent})


def test_reannotation_cannot_use_unsigned_or_changed_feedback(tmp_path,monkeypatch):
    from scripts import annotate_calibration_revision as annotation
    from scripts.import_word_calibration import fingerprint
    row,plan,feedback=fixture();value=feedback[0]|{'split':'train','human_attestation':False}
    folder=tmp_path/'data/review/five_dimensions_v2_calibration/synthetic';folder.mkdir(parents=True)
    (folder/'feedback.json').write_text(__import__('json').dumps(value))
    child=revise(row,plan,feedback)
    child['source_revision']['feedback_bindings']=[fingerprint(value)]
    monkeypatch.setattr(annotation,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='confirmed'):
        annotation.confirmed_feedback(child)
