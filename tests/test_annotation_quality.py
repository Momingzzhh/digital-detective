"""Synthetic checks for quarantine gates; these never count as real cases."""
from argparse import Namespace
import copy
from app.annotation_quality import (SemanticAudit,AUDIT_CHECKS,audit_binding,quality_passed,
    obvious_quality_errors)
from scripts.annotate_five_dimensions_batch import retry_checkpoint,PROTOCOL


def fixture():
    row={'source_binding':'synthetic-source-only'}
    ann={'actor_assessments':[{'name':'甲','participation':'直接实施','role':'仅为机制测试'}],
         'contradictions':[]}
    audit={k:True for k in AUDIT_CHECKS}|{'issue_codes':[],'reason':'仅为合成机制测试','dependency_checks':[]}
    saved={'five_dimensions':ann,'semantic_audit':audit,'semantic_audit_binding':audit_binding(row,ann)}
    return row,saved


def test_format_only_and_self_inconsistent_annotations_cannot_be_exported():
    row,saved=fixture();assert quality_passed(saved,row)
    del saved['semantic_audit'];assert not quality_passed(saved,row)
    row,saved=fixture();saved['five_dimensions']['actor_assessments'][0]['role']='在场见证，未参与交易'
    assert obvious_quality_errors(saved['five_dimensions']) and not quality_passed(saved,row)
    row,saved=fixture();saved['five_dimensions']['actor_assessments'][0]['role']='未直接参与交易，但明知而帮助接送并保管货物'
    assert not obvious_quality_errors(saved['five_dimensions'])
    row,saved=fixture();saved['five_dimensions']['contradictions']=[{'resolution':'两处一致，均描述同一动作，无矛盾'}]
    assert obvious_quality_errors(saved['five_dimensions']) and not quality_passed(saved,row)


def test_every_semantic_check_and_source_binding_are_required():
    row,saved=fixture()
    for field in AUDIT_CHECKS:
        bad=copy.deepcopy(saved);bad['semantic_audit'][field]=False
        assert not quality_passed(bad,row)
    assert not quality_passed(saved,{'source_binding':'different-source'})
    saved['semantic_audit']['issue_codes']=['SYNTHETIC_ERROR']
    assert not quality_passed(saved,row)


def test_retry_budget_cannot_loop_forever_or_retry_semantic_failures():
    args=Namespace(retry_errors=True,retry_rejected=True,max_case_attempts=2)
    old={'status':'local_annotation_error','generation_protocol':PROTOCOL,'protocol_attempts':1}
    assert retry_checkpoint(old,args)
    assert not retry_checkpoint(old|{'protocol_attempts':2},args)
    assert not retry_checkpoint(old|{'status':'semantic_quality_rejected'},args)
    assert retry_checkpoint(old|{'generation_protocol':'older'},args)


def test_semantic_revision_is_explicit_and_hard_bounded():
    args=Namespace(retry_errors=True,retry_rejected=True,max_case_attempts=2,repair_rejected=True)
    old={'status':'semantic_quality_rejected','generation_protocol':PROTOCOL,'semantic_repair_attempts':0,
         'raw_annotation':{'synthetic_mechanism_only':True}}
    assert retry_checkpoint(old,args)
    assert not retry_checkpoint(old|{'semantic_repair_attempts':1},args)
    assert not retry_checkpoint(old|{'status':'complexity_or_reference_rejected'},args)


def test_every_dependency_edge_needs_a_unique_positive_semantic_check():
    row,saved=fixture()
    saved['five_dimensions']['logic_chain']=[{'id':'L1','depends_on_steps':[]},
        {'id':'L2','depends_on_steps':['L1']},{'id':'L3','depends_on_steps':['L2']}]
    saved['semantic_audit_binding']=audit_binding(row,saved['five_dimensions'])
    checks=[{'step_id':'L2','parent_id':'L1','necessary':True,'reason':'仅为合成前提核验'},
        {'step_id':'L3','parent_id':'L2','necessary':True,'reason':'仅为合成前提核验'}]
    assert not quality_passed(saved,row)
    saved['semantic_audit']['dependency_checks']=checks
    assert quality_passed(saved,row)
    for bad in (checks[:1],checks+[checks[0]],checks[:1]+[checks[1]|{'necessary':False}],
                checks[:1]+[checks[1]|{'parent_id':'UNKNOWN'}]):
        saved['semantic_audit']['dependency_checks']=bad
        assert not quality_passed(saved,row)


def test_bound_audit_schema_forbids_evidence_parents_and_extra_edges():
    import pytest
    from app.annotation_quality import semantic_schema, audit_structure_errors
    ann = {'logic_chain': [{'id': 'L1', 'depends_on_steps': []},
                           {'id': 'L2', 'depends_on_steps': ['L1']}]}
    audit = {k: True for k in AUDIT_CHECKS} | {'issue_codes': [], 'reason': '合成格式核验',
        'dependency_checks': [{'step_id': 'L2', 'parent_id': 'L1', 'necessary': True, 'reason': '仅为合成前提测试'}]}
    schema = semantic_schema(ann)
    assert schema.model_validate(audit)
    assert not audit_structure_errors(audit, ann)
    bad = copy.deepcopy(audit)
    bad['dependency_checks'].append({'step_id': 'L2', 'parent_id': 'E7', 'necessary': False, 'reason': '错误地把证据当父节点'})
    assert audit_structure_errors(bad, ann)
    with pytest.raises(ValueError): schema.model_validate(bad)
    bad = copy.deepcopy(audit); bad['dependency_checks'][0]['parent_id'] = 'E7'
    with pytest.raises(ValueError): schema.model_validate(bad)


def test_real_negative_dependency_verdict_is_not_a_format_retry_reason():
    from app.annotation_quality import audit_structure_errors
    ann = {'logic_chain': [{'id': 'L1', 'depends_on_steps': []},
                           {'id': 'L2', 'depends_on_steps': ['L1']}]}
    audit = {k: True for k in AUDIT_CHECKS} | {'logic_dependencies_necessary': False,
        'issue_codes': ['ACTUAL_DEPENDENCY_FAILURE'], 'reason': '合成来源不足，不应因负面结论而重试',
        'dependency_checks': [{'step_id': 'L2', 'parent_id': 'L1', 'necessary': False, 'reason': '合成主张无需先前前提'}]}
    assert not audit_structure_errors(audit, ann)
