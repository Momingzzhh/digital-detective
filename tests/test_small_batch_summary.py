import pytest

from scripts.summarize_small_batch import summarize


def test_changed_audit_protocol_or_retry_is_not_an_additional_case():
    first = {'case_id': 'synthetic', 'group_id': 'group', 'source_binding': 'source',
             'split': 'train', 'finished_at': '2026-11-07T01:00:00+00:00',
             'status': 'semantic_quality_rejected', 'semantic_audit_raw': 'synthetic'}
    later = first | {'finished_at': '2026-11-07T02:00:00+00:00',
                     'semantic_audit_protocol': 'synthetic-new-protocol',
                     'audit_format_repair_attempts': 1}
    report = summarize([later, first])
    assert report['diagnostic_result_files'] == 2
    assert report['distinct_training_groups'] == 1 and report['distinct_source_versions'] == 1
    assert report['latest_status_counts'] == {'semantic_quality_rejected': 1}
    assert report['semantic_audit_calls_including_format_repair'] == 3


def test_summary_rejects_accidentally_included_heldout_semantics():
    with pytest.raises(ValueError, match='training diagnostics only'):
        summarize([{'split': 'test'}])
