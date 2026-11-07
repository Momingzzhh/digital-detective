"""Synthetic protocol integrity checks; no accuracy or real approvals."""
import copy
import json

import pytest

from app.annotation_quality import AUDIT_CHECKS, quality_passed
from app.field_semantic import PROTOCOL, binding, normalize, receipt_valid, schema_for
from app.five_dimensions import validate_small_batch_annotation
from test_auxiliary_training import inputs


def receipt():
    row, draft = inputs()
    ann = validate_small_batch_annotation(row, draft['raw_annotation'], row['sentence_spans'])
    wire = {'checks': {k: {'passed': True, 'sentence_ids': ['S1'], 'note': '仅为合成机制核对'}
                       for k in AUDIT_CHECKS if k != 'logic_dependencies_necessary'},
            'dependency_checks': [{'step_id': 'L2', 'parent_id': 'L1', 'necessary': True,
                                   'reason': '仅为合成依赖核对'}]}
    saved = {'five_dimensions': ann, 'semantic_audit_protocol': PROTOCOL,
             'semantic_audit_wire': wire, 'semantic_audit_raw': json.dumps(wire),
             'semantic_audit': normalize(wire, row, ann), 'semantic_audit_binding': binding(row, ann)}
    return row, saved


def test_field_verdicts_derive_totals_and_never_request_conflicting_error_lists():
    row, saved = receipt()
    assert receipt_valid(saved, row) and quality_passed(saved, row)
    for check in saved['semantic_audit_wire']['checks']:
        bad = copy.deepcopy(saved)
        bad['semantic_audit_wire']['checks'][check]['passed'] = False
        bad['semantic_audit_raw'] = json.dumps(bad['semantic_audit_wire'])
        bad['semantic_audit'] = normalize(bad['semantic_audit_wire'], row, bad['five_dimensions'])
        assert check in bad['semantic_audit']['issue_codes']
        assert not quality_passed(bad, row)
    bad = copy.deepcopy(saved)
    bad['semantic_audit_wire']['dependency_checks'][0]['necessary'] = False
    bad['semantic_audit_raw'] = json.dumps(bad['semantic_audit_wire'])
    bad['semantic_audit'] = normalize(bad['semantic_audit_wire'], row, bad['five_dimensions'])
    assert bad['semantic_audit']['logic_dependencies_necessary'] is False
    assert not quality_passed(bad, row)


def test_manual_verdict_flip_cannot_override_original_wire_or_source_binding():
    row, saved = receipt()
    bad = copy.deepcopy(saved); bad['semantic_audit']['events_fully_supported'] = False
    assert not receipt_valid(bad, row)
    bad = copy.deepcopy(saved); bad['semantic_audit_wire']['checks']['events_fully_supported']['passed'] = False
    assert not receipt_valid(bad, row)
    assert not receipt_valid(saved, row | {'source_binding': 'different-source'})


@pytest.mark.parametrize('defect', ['unknown_sentence', 'evidence_parent', 'extra_edge', 'global_flag', 'error_list'])
def test_wire_rejects_invented_references_and_unrequested_global_decisions(defect):
    row, saved = receipt(); wire = copy.deepcopy(saved['semantic_audit_wire'])
    if defect == 'unknown_sentence': wire['checks']['roles_match_actions']['sentence_ids'] = ['NO-SUCH-SENTENCE']
    elif defect == 'evidence_parent': wire['dependency_checks'][0]['parent_id'] = 'E7'
    elif defect == 'extra_edge': wire['dependency_checks'] *= 2
    elif defect == 'global_flag': wire['logic_dependencies_necessary'] = False
    else: wire['issue_codes'] = ['I1', 'I2']
    with pytest.raises(ValueError): schema_for(row, saved['five_dimensions']).model_validate(wire)


def test_positive_substantive_verdict_needs_a_source_reference_while_uncertainty_can_stay_empty():
    row, saved = receipt(); wire = copy.deepcopy(saved['semantic_audit_wire'])
    wire['checks']['events_fully_supported']['sentence_ids'] = []
    with pytest.raises(ValueError): schema_for(row, saved['five_dimensions']).model_validate(wire)
    wire['checks']['events_fully_supported']['passed'] = False
    assert schema_for(row, saved['five_dimensions']).model_validate(wire)
