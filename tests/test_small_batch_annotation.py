"""Training-only selection and new court-boundary regression checks."""
import copy
import json

import pytest

from app.five_dimensions import fingerprint
from app.source_policy import court_boundary, require_clean_evidence
from scripts import annotate_small_batch as worker
from test_auxiliary_training import inputs


def setup_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'source_review_hold', lambda row: None)
    monkeypatch.setattr(worker, 'complexity_exclusion', lambda row: None)
    source, _ = inputs()
    # Long enough for this cohort's explicit source-length band.
    source['evidence'][0]['text'] += '这里是用于来源长度选择的合成材料。' * 120
    second = copy.deepcopy(source)
    second.update(case_id='synthetic-second', group_id='synthetic-second-group', source_binding='synthetic-second-source')
    heldout = {'case_id': 'synthetic-test', 'split': 'test'}
    path = tmp_path / worker.POOL
    path.parent.mkdir(parents=True); path.write_text(''.join(json.dumps(r) + '\n' for r in [source, second, heldout]))
    return source, second


def test_cohort_uses_fixed_training_identity_and_never_needs_heldout_semantics(tmp_path, monkeypatch):
    first, second = setup_pool(tmp_path, monkeypatch)
    path, value = worker.create_cohort(2)
    assert len(value['cases']) == 2
    assert all(c['split'] == 'train' for c in value['cases'])
    expected = sorted([first, second], key=lambda row: fingerprint(row['group_id']))
    assert [c['case_id'] for c in value['cases']] == [r['case_id'] for r in expected]
    _, rows = worker.load_cohort(path)
    assert [r['case_id'] for r in rows] == [r['case_id'] for r in expected]
    # A changed split cannot become a new training identity after freezing.
    value['cases'][0]['split'] = 'test'
    value['cohort_binding'] = fingerprint({k: v for k, v in value.items() if k != 'cohort_binding'})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='original training identities'): worker.load_cohort(path)


def test_prior_processed_and_source_feedback_are_excluded_before_generation(tmp_path, monkeypatch):
    first, second = setup_pool(tmp_path, monkeypatch)
    processed = tmp_path / 'data/review/five_dimensions_v2' / (first['case_id'] + '.json')
    processed.parent.mkdir(parents=True); processed.write_text('not read as source semantics')
    monkeypatch.setattr(worker, 'source_review_hold', lambda row: {'decision': 'needs_revision'} if row['case_id'] == second['case_id'] else None)
    with pytest.raises(ValueError, match='No unprocessed'): worker.create_cohort(2)


def test_court_legality_assessment_is_not_a_source_confession():
    leaked = '当事人陈述了经过。以上各项证据经查来源合法，内容真实客观，可以证明本案事实。'
    assert court_boundary(leaked)['rule'] == 'court_evidence_confirmation'
    with pytest.raises(ValueError, match='judicial reasoning'): require_clean_evidence([{'text': leaked}])
    assert court_boundary('证人说自己没有看见有关行为，记录由其当场签字。') is None
    assert court_boundary('供述的具体内容。上述事实清楚，证据充分，')['rule'] == 'court_fact_certainty_summary'
    assert court_boundary('警方以涉嫌合成罪为由带回调查。')['rule'] == 'police_current_offence_label'


def test_institution_name_cannot_override_explicit_report_type():
    from scripts.build_v3 import evidence_kind
    assert evidence_kind('。5、某公安局物证鉴定室鉴定书，证实合成检验所见。') == '鉴定意见'
    assert evidence_kind('。2、证人甲的证言，证实其曾听说物证鉴定室出具鉴定书。') == '证人证言'


@pytest.mark.parametrize('split,batch_size', [('test', 1), ('dev', 1), ('train', 4)])
def test_shared_annotation_entry_cannot_open_heldout_small_batch_or_oversized_worker(split, batch_size):
    from argparse import Namespace
    from scripts.annotate_five_dimensions_batch import main
    with pytest.raises(ValueError, match='training-only'):
        main(Namespace(small_batch_cohort='synthetic-not-read', split=split, batch_size=batch_size, limit=1))


def test_assistant_draft_preserves_authorship_and_needs_exact_source_and_support(tmp_path, monkeypatch):
    source, draft = inputs()
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    value = {k: source[k] for k in ('case_id', 'group_id', 'split', 'source_binding')}
    value.update(author='assistant_source_reader', human_reviewed=False,
                 formal_approval=False, raw_annotation=draft['raw_annotation'])
    path = tmp_path / 'data/review/draft.private.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    ann, entry = worker.load_draft(source, path)
    assert ann['reasoning_depth'] == 2 and entry['path'] == 'data/review/draft.private.json'
    for changed in (value | {'human_reviewed': True}, value | {'source_binding': 'wrong'},
                    value | {'author': 'human_reviewer'}):
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='exact training source'): worker.load_draft(source, path)
    bad = copy.deepcopy(value); bad['raw_annotation']['complete_event_coverage'] = False
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='coverage'): worker.load_draft(source, path)
