"""Synthetic admission/isolation checks, never real cases or human approvals."""
import copy
import hashlib
import json

import pytest

from app.annotation_quality import AUDIT_CHECKS, audit_binding
from app.five_dimensions import (FiveDimensions, fingerprint, validate_annotation,
                                validate_small_batch_annotation)
from app.small_batch_training import (POOL, PROTOCOL, authorized_rows, build_row,
                                      load_dataset, public_counts, require_revision_identity)
from test_auxiliary_training import inputs, package as auxiliary_package


def write(root, name, value, jsonl=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + ('\n' if jsonl else ''))
    return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def policy():
    return {'protocol': PROTOCOL, 'authority': 'human_user_message_in_this_chat',
            'minimum_depth': 2, 'preferred_depth': 3, 'allow_small_batch': True,
            'allow_automatic_review_pending_human': True, 'require_source_and_answer_support': True,
            'preserve_existing_splits': True, 'human_approvals_created': 0,
            'user_statements': ['合成测试授权，不是真实用户签署']}


def freeze(root, value):
    entry = write(root, 'small-policy.json', value)
    rows, _ = authorized_rows(root, entry)
    manifest = public_counts(rows) | {'policy_file': entry,
        'dataset': write(root, 'small-data.jsonl', rows[0], True),
        'dataset_binding': fingerprint([r['record_binding'] for r in rows])}
    write(root, 'small-manifest.json', manifest)
    return rows, manifest


def corrected_package(root):
    auxiliary_package(root)
    manifest_entry = {'path': 'manifest.json', 'sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()}
    return freeze(root, policy() | {'accepted_corrected_manifests': [manifest_entry]})


def automatic_policy(root):
    source, draft = inputs()
    write(root, POOL, source, True)
    ann = validate_small_batch_annotation(source, draft['raw_annotation'], source['sentence_spans'])
    audit = {k: True for k in AUDIT_CHECKS} | {'issue_codes': [], 'reason': '仅为合成核验测试',
        'dependency_checks': [{'step_id': s['id'], 'parent_id': p, 'necessary': True,
                               'reason': '仅为合成依赖必要性测试'}
                              for s in ann['logic_chain'] for p in s['depends_on_steps']]}
    saved = {k: source[k] for k in ('case_id', 'source_binding', 'split')}
    saved.update(five_dimensions=ann, semantic_audit=audit,
                 semantic_audit_binding=audit_binding(source, ann),
                 status='automatically_audited_pending_human_review')
    entry = write(root, 'automatic.json', saved)
    return policy() | {'automatically_audited_annotations': [{'case_id': source['case_id'], 'annotation_file': entry}]}, saved


def test_two_levels_allow_training_without_changing_three_level_benchmark(tmp_path):
    rows, manifest = corrected_package(tmp_path)
    loaded, _ = load_dataset(tmp_path, tmp_path / 'small-manifest.json')
    assert loaded == rows
    row = rows[0]
    assert row['five_dimensions']['reasoning_depth'] == 2
    assert row['dataset_role'] == 'small_batch_formal_training'
    assert not row['human_reviewed'] and not row['formal_approval']
    assert not row['eligible_for_formal_benchmark']
    assert manifest['strict_three_level_benchmark_cases_added'] == 0
    with pytest.raises(ValueError, match='three linked'):
        validate_annotation(row, {k: row['five_dimensions'][k] for k in FiveDimensions.model_fields}, row['sentence_spans'])


def test_single_actor_without_conflict_needs_explicit_small_batch_scope():
    source, draft = inputs()
    raw = draft['raw_annotation']
    raw['candidate_names'] = raw['candidate_names'][:1]
    raw['actor_assessments'] = raw['actor_assessments'][:1]
    raw['contradictions'] = []
    with pytest.raises(ValueError, match='conflict|multiple'):
        build_row(source, raw, 'policy', {}, 'pending')
    row = build_row(source, raw, 'policy', {}, 'pending', allow_single_actor_without_conflict=True)
    assert row['single_actor_without_conflict'] is True
    assert row['five_dimensions']['reasoning_depth'] == 2
    assert public_counts([row])['single_actor_without_conflict_cases'] == 1
    assert not row['eligible_for_formal_benchmark']
    with pytest.raises(ValueError):
        validate_annotation(source, raw, source['sentence_spans'])


@pytest.mark.parametrize('value', ['true', 1, None])
def test_single_actor_scope_cannot_be_a_truthy_nonboolean(value):
    from app.small_batch_training import validate_policy
    with pytest.raises(ValueError, match='explicit boolean'):
        validate_policy(policy() | {'allow_single_actor_without_conflict': value})
    source, draft = inputs()
    with pytest.raises(ValueError, match='explicit boolean'):
        validate_small_batch_annotation(source, draft['raw_annotation'], source['sentence_spans'],
                                        allow_single_actor_without_conflict=value)


def test_automatic_single_actor_admission_requires_policy_not_just_positive_audit(tmp_path):
    value, saved = automatic_policy(tmp_path)
    raw = {k: saved['five_dimensions'][k] for k in FiveDimensions.model_fields}
    raw.update(candidate_names=raw['candidate_names'][:1],
               actor_assessments=raw['actor_assessments'][:1], contradictions=[])
    source = json.loads((tmp_path / POOL).read_text())
    ann = validate_small_batch_annotation(source, raw, source['sentence_spans'],
                                         allow_single_actor_without_conflict=True)
    saved.update(five_dimensions=ann, semantic_audit_binding=audit_binding(source, ann))
    value['automatically_audited_annotations'][0]['annotation_file'] = write(tmp_path, 'automatic.json', saved)
    with pytest.raises(ValueError, match='conflict|multiple'):
        freeze(tmp_path, value)
    rows, manifest = freeze(tmp_path, value | {'allow_single_actor_without_conflict': True})
    assert manifest['single_actor_without_conflict_cases'] == 1
    assert rows[0]['human_reviewed'] is False


def test_inherited_source_flag_cannot_change_single_actor_statistics():
    source, draft = inputs()
    source['single_actor_without_conflict'] = True
    row = build_row(source, draft['raw_annotation'], 'policy', {}, 'pending')
    assert 'single_actor_without_conflict' not in row
    assert public_counts([row]).get('single_actor_without_conflict_cases', 0) == 0


def test_old_manifest_without_single_actor_rows_cannot_claim_extra_single_actor_cases(tmp_path):
    _, manifest = corrected_package(tmp_path)
    manifest['single_actor_without_conflict_cases'] = 1
    write(tmp_path, 'small-manifest.json', manifest)
    with pytest.raises(ValueError, match='Single-actor manifest count'):
        load_dataset(tmp_path, tmp_path / 'small-manifest.json')


def test_even_a_three_level_small_batch_cannot_overwrite_legacy_reports_via_pilot(tmp_path):
    from types import SimpleNamespace
    from scripts.train_local import main
    source, draft = inputs()
    draft['raw_annotation']['logic_chain'][2].update(
        independently_supported=False, depends_on_steps=['L2'], dependency_reason='合成三层入口隔离检查')
    row = build_row(source, draft['raw_annotation'], 'policy', {}, 'pending')
    assert row['five_dimensions']['reasoning_depth'] == 3
    path = tmp_path / 'small.jsonl'; path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='explicit small-batch entry point'):
        main(SimpleNamespace(dataset=str(path), pilot=True))


def test_discovered_support_defect_blocks_reuse_without_rewriting_the_old_audit(tmp_path):
    value, saved = automatic_policy(tmp_path)
    binding = fingerprint(saved['five_dimensions'])
    hold = {'author':'assistant_source_reader','human_reviewed':False,
            'case_id':saved['case_id'],'source_binding':saved['source_binding'],
            'annotation_binding':binding,'source_support_defect':'合成遗漏引用问题'}
    hold['hold_binding'] = fingerprint(hold)
    write(tmp_path, f'data/review/small_batch_support_holds/{saved["case_id"]}/{binding}.json', hold)
    with pytest.raises(ValueError, match='Reported source-support defect'): freeze(tmp_path, value)
    assert json.loads((tmp_path / 'automatic.json').read_text()) == saved


@pytest.mark.parametrize('defect', ['dev', 'test', 'one_level', 'offset', 'incomplete', 'unsupported_actor'])
def test_revised_depth_never_relaxes_source_support_or_split_isolation(defect):
    source, draft = inputs()
    if defect in ('dev', 'test'): source['split'] = defect
    elif defect == 'one_level':
        for s in draft['raw_annotation']['logic_chain']:
            s.update(independently_supported=True, depends_on_steps=[], dependency_reason='')
    elif defect == 'offset': source['sentence_spans'][0]['end'] -= 1
    elif defect == 'incomplete': draft['raw_annotation']['complete_event_coverage'] = False
    else: draft['raw_annotation']['actor_assessments'][0]['name'] = '原文没有的人'
    with pytest.raises(ValueError): build_row(source, draft['raw_annotation'], 'policy', {}, 'pending')


@pytest.mark.parametrize('mutation', ['dataset_label', 'dataset_answer', 'counts', 'policy', 'source'])
def test_dataset_regenerated_from_provenance_even_after_hash_rewrite(tmp_path, mutation):
    rows, manifest = corrected_package(tmp_path)
    if mutation.startswith('dataset_'):
        if mutation == 'dataset_label': rows[0]['human_reviewed'] = True
        else: rows[0]['hidden_gold']['perpetrators'] = ['伪造人物']
        manifest['dataset'] = write(tmp_path, 'small-data.jsonl', rows[0], True)
    elif mutation == 'counts': manifest['training_cases'] = 99
    elif mutation == 'policy':
        p = tmp_path / 'small-policy.json'
        value = json.loads(p.read_text()); value['minimum_depth'] = 1
        manifest['policy_file'] = write(tmp_path, 'small-policy.json', value)
    else: (tmp_path / 'source.jsonl').write_text('changed source')
    write(tmp_path, 'small-manifest.json', manifest)
    with pytest.raises(ValueError): load_dataset(tmp_path, tmp_path / 'small-manifest.json')


def test_new_automatic_case_requires_completed_exact_source_audit(tmp_path):
    value, saved = automatic_policy(tmp_path)
    rows, _ = freeze(tmp_path, value)
    assert rows[0]['review_status'] == 'automatically_audited_pending_human_review'
    assert rows[0]['human_reviewed'] is False
    saved['semantic_audit']['events_fully_supported'] = False
    value['automatically_audited_annotations'][0]['annotation_file'] = write(tmp_path, 'automatic.json', saved)
    with pytest.raises(ValueError, match='incomplete or failed'): freeze(tmp_path, value)


def test_automatic_route_cannot_bypass_signed_source_hold(tmp_path, monkeypatch):
    value, _ = automatic_policy(tmp_path)
    monkeypatch.setattr('app.small_batch_training.source_review_hold', lambda row: {'decision': 'needs_revision'})
    with pytest.raises(ValueError, match='signed source feedback'): freeze(tmp_path, value)


def test_heldout_metadata_cannot_be_relabelled_for_automatic_training(tmp_path):
    value, saved = automatic_policy(tmp_path)
    source = json.loads((tmp_path / POOL).read_text())
    source['split'] = 'test'
    write(tmp_path, POOL, source, True)
    with pytest.raises(ValueError, match='existing training source'): freeze(tmp_path, value)


def test_opened_independent_test_blocks_new_small_batch_runs(tmp_path):
    from scripts.train_small_batch import require_unopened_test
    require_unopened_test(tmp_path)
    path = tmp_path / 'runtime/five_dimensions_v2_test_freeze.json'
    path.parent.mkdir(); path.write_text('{}')
    with pytest.raises(ValueError, match='test already opened'): require_unopened_test(tmp_path)


def test_duplicate_groups_in_two_admission_routes_are_rejected(tmp_path):
    value, _ = automatic_policy(tmp_path)
    value['automatically_audited_annotations'] *= 2
    with pytest.raises(ValueError, match='Duplicate case'): freeze(tmp_path, value)


@pytest.mark.parametrize('defect', ['heldout', 'group', 'source', 'raw', 'parent'])
def test_revision_cannot_smuggle_in_relabelled_heldout_or_replaced_raw_records(defect):
    source, _ = inputs()
    revised = copy.deepcopy(source)
    revised['training_source_revision'] = {'parent_source_binding': source['source_binding']}
    if defect == 'heldout': source['split'] = 'test'
    elif defect == 'group': revised['group_id'] = 'another-group'
    elif defect == 'source': revised['source'] = {'sha256_fulltext': 'different'}
    elif defect == 'raw': revised['raw_record'] = {'qw': 'substituted'}
    else: revised['training_source_revision']['parent_source_binding'] = 'another-version'
    with pytest.raises(ValueError, match='original training identity'): require_revision_identity(source, revised)


def test_audit_format_repair_cannot_change_candidate_or_retry_a_substantive_failure(tmp_path):
    value, saved = automatic_policy(tmp_path)
    prior = copy.deepcopy(saved)
    prior['status'] = 'semantic_quality_rejected'
    prior['semantic_audit']['dependency_checks'].append(
        {'step_id': 'L2', 'parent_id': 'E7', 'necessary': False, 'reason': '合成格式错误'})
    saved['audit_format_repair_attempts'] = 1
    saved['previous_invalid_audit_result'] = write(tmp_path, 'prior-invalid.json', prior)
    value['automatically_audited_annotations'][0]['annotation_file'] = write(tmp_path, 'automatic.json', saved)
    rows, _ = freeze(tmp_path, value)
    assert rows[0]['human_reviewed'] is False
    prior['five_dimensions']['logic_chain'][0]['claim'] = '不同的旧候选'
    saved['previous_invalid_audit_result'] = write(tmp_path, 'prior-invalid.json', prior)
    value['automatically_audited_annotations'][0]['annotation_file'] = write(tmp_path, 'automatic.json', saved)
    with pytest.raises(ValueError, match='unchanged candidate'): freeze(tmp_path, value)


def test_new_signed_feedback_cannot_be_ignored_by_reusing_a_prior_accepted_correction(tmp_path, monkeypatch):
    corrected_package(tmp_path)
    monkeypatch.setattr('app.small_batch_training.source_review_hold',
                        lambda row: {'reviewed_date': '2026-11-07', 'received_at': '2026-11-07T01:00:00+00:00'})
    with pytest.raises(ValueError, match='New signed source feedback'):
        load_dataset(tmp_path, tmp_path / 'small-manifest.json')
