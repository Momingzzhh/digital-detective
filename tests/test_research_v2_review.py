import json
import pytest
from app import research_v2_review as review
from app.experiment_manifest import digest
from test_system import client, account
from test_research_gates import fixture_row
from test_interactive_curation import synthetic_case


@pytest.fixture
def bundle(client, monkeypatch, tmp_path):
    base = tmp_path/'bundle'; directory = base/'1234567890abcdef'; directory.mkdir(parents=True)
    monkeypatch.setattr(review, 'BASE', base)
    monkeypatch.setattr(review, 'RECEIPTS', tmp_path/'receipts')
    monkeypatch.setenv('RESEARCH_REVIEW_ENABLED', 'true')
    case = synthetic_case(); row = fixture_row('train'); row['case_id'] = case.case_id
    row['sentence_spans'] = []
    case.source_row_sha256 = digest(row)
    raw = [case.model_dump()]
    binding = digest({'source_file':'fixture','cases':raw})
    for path, value in [(base/'current.json', {'directory':directory.name,'binding':binding}),
                        (directory/'cases.private.json',raw),
                        (directory/'manifest.private.json',{'binding':binding,'source_sha256':'fixture'}),
                        (directory/'sources.private.json',{case.case_id:row})]:
        path.write_text(json.dumps(value))
    return client, directory, tmp_path/'receipts'


def test_review_auth_csrf_and_item_checks(bundle):
    client, _, receipts = bundle
    endpoint = '/api/research/v2/items/synthetic-only'
    assert client.get(endpoint).status_code == 401
    h = account(client)
    value = client.get(endpoint).json()
    assert value['receipt']['status'] == 'not_reviewed'
    body = {'review_binding':value['review_binding'], 'decision':'approve_curation',
            'checks':{}, 'notes':'合成测试签署，仅用于测试，不计真实复核',
            'human_minutes':1, 'personally_reviewed':True}
    assert client.post(endpoint+'/review',json=body).status_code == 403
    assert client.post(endpoint+'/review',headers=h,json=body).status_code == 422
    body['checks'] = {k:False for k in value['required_checks']}
    assert client.post(endpoint+'/review',headers=h,json=body).status_code == 422
    body['checks'] = {k:True for k in value['required_checks']}
    body['reviewer_user_id'] = 'spoofed'
    assert client.post(endpoint+'/review',headers=h,json=body).status_code == 422
    body.pop('reviewer_user_id')
    assert client.post(endpoint+'/review',headers=h,json=body).status_code == 200
    record = json.loads(next(receipts.rglob('*.json')).read_text())
    assert record['reviewer_user_id'] and record['formal_benchmark_eligible'] is False
    assert client.get(endpoint).json()['receipt']['status'] == 'approve_curation'


def test_changed_source_fails_closed_and_new_audit_invalidates_receipt(bundle):
    client, directory, _ = bundle; h = account(client)
    endpoint = '/api/research/v2/items/synthetic-only'
    content = client.get(endpoint).json()
    body = {'review_binding':content['review_binding'], 'decision':'reject',
        'checks':{k:False for k in content['required_checks']},
        'notes':'合成测试记录，不是真实人工核验', 'human_minutes':2, 'personally_reviewed':True}
    assert client.post(endpoint+'/review',headers=h,json=body).status_code == 200
    audit = directory/'local_audits/synthetic-only'; audit.mkdir(parents=True)
    (audit/'new.json').write_text('{"status":"new-machine-suggestion"}')
    assert client.get(endpoint).json()['receipt']['status'] == 'stale'
    assert client.post(endpoint+'/review',headers=h,json=body).status_code == 409
    source_path = directory/'sources.private.json'
    source = json.loads(source_path.read_text()); source['synthetic-only']['raw_record']['qw'] += 'changed'
    source_path.write_text(json.dumps(source))
    assert client.get(endpoint).status_code == 409


def test_default_disabled_and_train_only(bundle, monkeypatch):
    client, directory, _ = bundle; account(client)
    monkeypatch.setenv('RESEARCH_REVIEW_ENABLED','false')
    assert client.get('/api/research/v2/items').status_code == 403
    monkeypatch.setenv('RESEARCH_REVIEW_ENABLED','true')
    raw = json.loads((directory/'cases.private.json').read_text()); raw[0]['split'] = 'test'
    (directory/'cases.private.json').write_text(json.dumps(raw))
    assert client.get('/api/research/v2/items').status_code == 409
