import json
from argparse import Namespace
import pytest
from app import complexity_review as review
from app.five_dimensions import fingerprint


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'ROOT',tmp_path)
    row={'case_id':'synthetic','group_id':'g','split':'train','source_binding':'original-source'}
    feedback={'case_id':'synthetic','split':'train','source_binding':'revised-source','human_attestation':True,
        'reviewer_id':'synthetic-test-only','reviewed_date':'2026-10-26','decision':'needs_revision',
        'answers':{'depth':'仅为合成测试：不足三层'}}
    relative='data/review/five_dimensions_v2_calibration/synthetic/feedback.json'
    path=tmp_path/relative;path.parent.mkdir(parents=True);path.write_text(json.dumps(feedback))
    exclusion={'case_id':'synthetic','group_id':'g','split':'train','maximum_reviewed_depth':2,
        'excluded_source_bindings':['original-source','revised-source'],'reviewed_source_binding':'revised-source',
        'feedback_path':relative,'feedback_binding':fingerprint(feedback)}
    target=tmp_path/'data/review/five_dimensions_v2_complexity_exclusions/synthetic.json'
    target.parent.mkdir(parents=True);target.write_text(json.dumps(exclusion))
    return row,feedback,exclusion,path,target


def test_exclusion_only_applies_to_reviewed_material_versions(tmp_path,monkeypatch):
    row,_,_,_,_=fixture(tmp_path,monkeypatch)
    assert review.complexity_exclusion(row)['maximum_reviewed_depth']==2
    assert review.complexity_exclusion(row|{'source_binding':'revised-source'})
    assert review.complexity_exclusion(row|{'source_binding':'new-evidence-version'}) is None


@pytest.mark.parametrize('defect',['unsigned','changed_feedback','heldout','wrong_source','no_depth'])
def test_exclusion_cannot_reuse_unsigned_or_unrelated_feedback(tmp_path,monkeypatch,defect):
    row,feedback,exclusion,path,target=fixture(tmp_path,monkeypatch)
    if defect=='unsigned':feedback['human_attestation']=False
    elif defect=='heldout':row['split']='test'
    elif defect=='wrong_source':feedback['source_binding']='unrelated'
    elif defect=='no_depth':feedback['answers']={}
    else:feedback['decision']='uncertain'
    path.write_text(json.dumps(feedback))
    if defect!='changed_feedback':exclusion['feedback_binding']=fingerprint(feedback)
    target.write_text(json.dumps(exclusion))
    with pytest.raises(ValueError):review.complexity_exclusion(row)


def test_training_rejects_human_excluded_source_before_loading_model(tmp_path,monkeypatch):
    from scripts import train_local
    row,_,_,_,_=fixture(tmp_path,monkeypatch)
    dataset=tmp_path/'data.jsonl';dataset.write_text(json.dumps(row)+'\n')
    monkeypatch.setattr(train_local,'ROOT',tmp_path)
    monkeypatch.setattr(train_local,'record_binding',lambda r:None)
    monkeypatch.setattr(train_local,'verify_model_files',lambda:pytest.fail('Must not load excluded training data'))
    with pytest.raises(ValueError,match='insufficient complexity'):
        train_local.main(Namespace(dataset=str(dataset),pilot=True))


def test_confirmed_revision_feedback_holds_only_the_bound_source(tmp_path,monkeypatch):
    row,_,_,_,_=fixture(tmp_path,monkeypatch)
    word=b'synthetic Word package bytes for test only'
    document=tmp_path/'data/review/five_dimensions_v2_calibration/documents/feedback.docx'
    document.parent.mkdir(parents=True);document.write_bytes(word)
    value={'case_id':'synthetic','split':'train','source_binding':'original-source',
           'feedback_protocol':'word-calibration-feedback-v1',
           'human_attestation':True,'reviewer_id':'synthetic-test-only','reviewed_date':'2026-10-26',
           'decision':'needs_revision','original_document':str(document.relative_to(tmp_path)),
           'input_file_sha256':__import__('hashlib').sha256(word).hexdigest()}
    path=tmp_path/'data/review/five_dimensions_v2_calibration/synthetic/revision.json'
    path.write_text(json.dumps(value))
    assert review.source_review_hold(row)['decision']=='needs_revision'
    assert review.source_review_hold(row|{'source_binding':'new-evidence-version'}) is None
    document.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash differs'):review.source_review_hold(row)
