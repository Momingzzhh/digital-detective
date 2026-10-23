import copy
from argparse import Namespace
import json
import pytest
from scripts import prepare_calibration_review as calibration


def fixture():
    case={'case_id':'synthetic','split':'train','source_binding':'source',
          'annotation_result_binding':'annotation'}
    manifest={'rubric':calibration.RUBRIC,'cases':[case]}
    manifest['bundle_binding']=calibration.fingerprint(manifest)
    value={'rubric':calibration.RUBRIC,'bundle_binding':manifest['bundle_binding'],'reviews':[case|{
        'decision':'needs_revision','human_attestation':True,'reviewer_id':'synthetic-test-only',
        'reviewed_at':'2026-10-24T00:00:00+08:00',
        'answers':{key:'仅为机制测试' for key in calibration.QUESTIONS}}]}
    return manifest,value


@pytest.mark.parametrize('change',['heldout','source','annotation','identity','attestation','missing_answer','duplicate'])
def test_calibration_rejects_wrong_binding_or_missing_human_attestation(change):
    manifest,value=fixture();r=value['reviews'][0]
    if change=='heldout':r['split']='test'
    elif change=='source':r['source_binding']='changed'
    elif change=='annotation':r['annotation_result_binding']='changed'
    elif change=='identity':r['reviewer_id']=''
    elif change=='attestation':r['human_attestation']=False
    elif change=='missing_answer':del r['answers']['depth']
    elif change=='duplicate':value['reviews'].append(copy.deepcopy(r))
    with pytest.raises(ValueError):calibration.validate_submission(value,manifest)


def test_calibration_import_cannot_write_formal_approval_or_overwrite_feedback(tmp_path,monkeypatch):
    manifest,value=fixture()
    folder=tmp_path/'artifacts/calibration'/calibration.RUBRIC/manifest['bundle_binding'][:16];folder.mkdir(parents=True)
    (folder/'manifest.private.json').write_text(json.dumps(manifest))
    submission=tmp_path/'human.json';submission.write_text(json.dumps(value))
    monkeypatch.setattr(calibration,'ROOT',tmp_path)
    args=Namespace(import_review=str(submission))
    calibration.main(args);calibration.main(args)
    assert len(list((tmp_path/'data/review/five_dimensions_v2_calibration/synthetic').glob('*.json')))==1
    assert not (tmp_path/'data/review/five_dimensions_v2_human').exists()


def test_source_text_is_escaped_before_inclusion_in_review_html():
    manifest,_=fixture();case=manifest['cases'][0]|{'group_id':'g','source':{},
        'source_fulltext':'<script>alert(1)</script>','evidence':[],'sentence_spans':[],
        'annotation':{'name':'</script><script>alert(2)</script>'},
        'machine_status':'synthetic','machine_feedback':'untrusted'}
    result=calibration.page([case],'binding')
    assert '<script>alert(1)</script>' not in result
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in result
    assert '<script>alert(2)</script>' not in result
    assert 'fetch(' not in result and 'https://' not in result


def test_calibration_pack_never_accepts_heldout_cases():
    with pytest.raises(ValueError,match='only expose training'):
        calibration.pack_case({'split':'test'},{'split':'test'})


def test_changed_snapshot_cannot_accept_a_review():
    manifest,value=fixture();manifest['cases'][0]['source_binding']='changed'
    with pytest.raises(ValueError,match='snapshot changed'):
        calibration.validate_submission(value,manifest)


def test_preview_server_only_serves_fixed_bound_package(tmp_path,monkeypatch):
    from scripts import calibration_server
    from fastapi import HTTPException
    monkeypatch.setattr(calibration_server,'ROOT',tmp_path)
    with pytest.raises(HTTPException):calibration_server.index()
    manifest,_=fixture();folder=tmp_path/'artifacts/calibration'/calibration.RUBRIC
    target=folder/manifest['bundle_binding'][:16];target.mkdir(parents=True)
    (folder/'manifest.private.json').write_text(json.dumps(manifest))
    (target/'index.html').write_text('<html><script>\nconst synthetic=1;\n</script></html>')
    response=calibration_server.index()
    assert response.headers['cache-control']=='no-store'
    assert "connect-src 'none'" in response.headers['content-security-policy']
    assert "script-src 'sha256-" in response.headers['content-security-policy']
