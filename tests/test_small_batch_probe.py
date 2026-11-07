"""Source probes stay training-only and never turn a boundary into accuracy."""
import hashlib

import pytest

from app.five_dimensions import fingerprint
from app.models import Conclusion
from scripts.probe_small_batch_adapter import PROTOCOL, prepare_inputs, check_boundary
from test_small_batch_training import automatic_policy, freeze


def inputs(tmp_path):
    value, _ = automatic_policy(tmp_path)
    rows, manifest = freeze(tmp_path, value)
    row, = rows
    run = {'run_id': 'synthetic-run', 'dataset_binding': manifest['dataset_binding'],
           'source_manifest': {'path': 'small-manifest.json',
               'sha256': hashlib.sha256((tmp_path / 'small-manifest.json').read_bytes()).hexdigest()}}
    evidence = [{k: row['evidence'][0][k] for k in ('id', 'kind', 'text')}]
    plan = {'protocol': PROTOCOL, 'author': 'assistant_source_reader', 'human_reviewed': False,
            'run_id': run['run_id'], 'dataset_binding': run['dataset_binding'], 'probes': [{
                'case_id': row['case_id'], 'source_binding': row['source_binding'],
                'evidence_ids': ['E1'], 'visible_evidence_binding': fingerprint(evidence),
                'expected_boundary': 'insufficient_no_accusation', 'source_reader_reason': '合成测试说明'}]}
    return run, plan


def test_probe_payload_has_only_preselected_visible_training_source_not_gold(tmp_path):
    run, plan = inputs(tmp_path)
    payload, = prepare_inputs(tmp_path, run, plan)
    assert set(payload) == {'evidence', 'untrusted_note'}
    assert len(payload['evidence']) == 1 and set(payload['evidence'][0]) == {'id', 'kind', 'text'}


@pytest.mark.parametrize('defect', ['heldout', 'source', 'evidence', 'all_evidence', 'binding', 'rationale'])
def test_probe_cannot_switch_sources_or_reinterpret_an_arbitrary_full_case_as_insufficient(tmp_path, defect):
    run, plan = inputs(tmp_path); probe = plan['probes'][0]
    if defect == 'heldout': probe['case_id'] = 'heldout-not-in-training'
    elif defect == 'source': probe['source_binding'] = 'changed'
    elif defect == 'evidence': probe['evidence_ids'] = ['E99']
    elif defect == 'all_evidence': probe['evidence_ids'] = [f'E{i}' for i in range(1, 7)]
    elif defect == 'binding': probe['visible_evidence_binding'] = 'changed'
    else: probe['source_reader_reason'] = ''
    with pytest.raises(ValueError): prepare_inputs(tmp_path, run, plan)


def test_duplicate_probe_is_not_an_extra_case(tmp_path):
    run, plan = inputs(tmp_path); plan['probes'] *= 2
    with pytest.raises(ValueError, match='Repeated probe'): prepare_inputs(tmp_path, run, plan)


def test_partial_boundary_counts_unknown_citations_and_accusations_as_failures():
    payload = {'evidence': [{'id': 'E1'}]}
    result = Conclusion(status='证据不足', summary='合成测试', charges=[], perpetrators=[],
                        participants=[], reconstruction=[], findings=[], conflicts=[], hypotheses=[],
                        questions=['请补充原始材料'], missing_information=['原始材料'], confidence=0.2)
    assert all(check_boundary(result, payload).values())
    wrong = result.model_copy(update={'charges': ['合成测试罪']})
    assert not check_boundary(wrong, payload)['insufficient_no_accusation']
    wrong = result.model_copy(update={'questions': []})
    assert not check_boundary(wrong, payload)['asks_for_missing_evidence']
    wrong = result.model_copy(deep=True)
    from app.models import Finding
    wrong.findings = [Finding(claim='合成引用', evidence_ids=['E99'])]
    assert not check_boundary(wrong, payload)['citations_within_provided_evidence']
