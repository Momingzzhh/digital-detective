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
    audit={k:True for k in AUDIT_CHECKS}|{'issue_codes':[],'reason':'仅为合成机制测试'}
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
