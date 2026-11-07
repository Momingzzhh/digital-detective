"""Source-bound small-batch formal training under an explicit revised policy.

The 600-case benchmark and its human acceptance gates remain independently
measured. A user-approved corrected version is reusable; other new labels need
a completed, source-bound semantic audit. Neither route creates a human review.
"""
from collections import Counter
import copy
from datetime import date, datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from .auxiliary_training import SOURCE_KEYS, bound_file, load_dataset as load_auxiliary
from .annotation_quality import quality_passed, audit_structure_errors
from .complexity_review import complexity_exclusion, source_review_hold
from .five_dimensions import (VERSION, FiveDimensions, fingerprint, ground_truth,
                              record_binding, validate_small_batch_annotation,
                              validate_splits)

PROTOCOL = 'small-batch-formal-training-v1'
POOL = 'data/processed/five_dimensions_v2/candidates.private.jsonl'


def require_no_reported_support_defect(root, source, saved):
    binding = fingerprint(saved['five_dimensions'])
    path = root / 'data/review/small_batch_support_holds' / source['case_id'] / (binding + '.json')
    if not path.exists():
        return
    hold = json.loads(path.read_text())
    if (hold.get('author') != 'assistant_source_reader' or hold.get('human_reviewed') is not False or
        hold.get('case_id') != source['case_id'] or hold.get('source_binding') != source['source_binding'] or
        hold.get('annotation_binding') != binding or not hold.get('source_support_defect') or
        hold.get('hold_binding') != fingerprint({k:v for k,v in hold.items() if k != 'hold_binding'})):
        raise ValueError('Invalid assistant source-support defect record')
    raise ValueError('Reported source-support defect: preserve the old label and audit a corrected version')


def require_revision_identity(original, revised):
    if (not original or original.get('split') != 'train' or revised.get('split') != 'train' or
        any(original[k] != revised.get(k) for k in
            ('case_id', 'group_id', 'source', 'source_fact_sha256', 'raw_record')) or
        revised.get('training_source_revision', {}).get('parent_source_binding') != original['source_binding']):
        raise ValueError('Source revision cannot replace an original training identity or raw record')


def validate_policy(value):
    required = {'protocol': PROTOCOL, 'authority': 'human_user_message_in_this_chat',
                'minimum_depth': 2, 'preferred_depth': 3, 'allow_small_batch': True,
                'allow_automatic_review_pending_human': True,
                'require_source_and_answer_support': True,
                'preserve_existing_splits': True, 'human_approvals_created': 0}
    if any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in required.items()):
        raise ValueError('Explicit revised small-batch training policy is required')
    if not value.get('user_statements'):
        raise ValueError('Training policy must retain the actual user instructions')
    if 'allow_single_actor_without_conflict' in value and type(value['allow_single_actor_without_conflict']) is not bool:
        raise ValueError('Single-actor admission must be an explicit boolean policy')
    return value


def build_row(source, annotation, policy_binding, provenance, review_status, *, allow_single_actor_without_conflict=False):
    if source.get('split') != 'train':
        raise ValueError('Small-batch training cannot consume validation or test cases')
    if source.get('source_binding') != fingerprint({k: source[k] for k in SOURCE_KEYS}):
        raise ValueError('Small-batch source binding changed')
    from scripts.export_review_bundle import verify
    verify(source, {'raw_record': source['raw_record']})
    ann = validate_small_batch_annotation(source, annotation, source['sentence_spans'],
        allow_single_actor_without_conflict=allow_single_actor_without_conflict)
    row = copy.deepcopy(source)
    # Keep historical auxiliary files immutable; this is a newly authorized
    # dataset version with a separate role, provenance and policy binding.
    for key in ('auxiliary_protocol', 'auxiliary_provenance', 'authorization_binding',
                'user_accepted_for_auxiliary_training', 'single_actor_without_conflict'):
        row.pop(key, None)
    row.update(annotation_version=VERSION, five_dimensions=ann,
               dataset_role='small_batch_formal_training', training_protocol=PROTOCOL,
               training_policy_binding=policy_binding, training_provenance=provenance,
               review_status=review_status, status='admitted_for_small_batch_training',
               human_reviewed=False, formal_approval=False,
               eligible_for_formal_benchmark=False)
    row['hidden_gold'] = ground_truth(row)
    if len(ann['candidate_names']) == 1 and not ann['contradictions']:
        row['single_actor_without_conflict'] = True
    row['record_binding'] = record_binding(row)
    return row


def authorized_rows(root, policy_entry):
    policy = validate_policy(json.loads(bound_file(root, policy_entry)))
    binding = fingerprint(policy)
    allow_single = policy.get('allow_single_actor_without_conflict', False)
    rows = []
    for entry in policy.get('accepted_corrected_manifests', []):
        bound_file(root, entry)
        accepted, auxiliary_manifest = load_auxiliary(root, root / entry['path'])
        for source in accepted:
            # This route replays the exact subsequently user-accepted corrected
            # annotation. Historical feedback remains intact; it must not be
            # mistaken for a prohibition on the accepted correction itself.
            # Automatic admission below has no such exception.
            hold = source_review_hold(source)
            if hold:
                original_acceptance = json.loads(bound_file(root, auxiliary_manifest['authorization_file']))
                cutoff = original_acceptance.get('client_date')
                received = hold.get('received_at')
                received_date = (datetime.fromisoformat(received.replace('Z', '+00:00'))
                                 .astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()) if received else ''
                if (not cutoff or not hold.get('reviewed_date') or
                    hold['reviewed_date'] >= cutoff or received_date >= cutoff):
                    raise ValueError('New signed source feedback supersedes the prior corrected-version acceptance')
                date.fromisoformat(cutoff)
            raw = {k: source['five_dimensions'][k] for k in FiveDimensions.model_fields}
            provenance = {'accepted_corrected_manifest': entry,
                          'historical_source_feedback_retained': bool(hold),
                          'historical_three_level_exclusion_retained': bool(complexity_exclusion(source))}
            rows.append(build_row(source, raw, binding, provenance,
                                  'user_accepted_corrected_version_pending_benchmark_review',
                                  allow_single_actor_without_conflict=allow_single))
    automated = policy.get('automatically_audited_annotations', [])
    if automated:
        # Resolve frozen training sources or rederive an explicit source-only
        # revision from its parent pool and immutable plan.
        pool = {r['case_id']: r for r in map(json.loads, (root / POOL).open()) if r['split'] == 'train'}
        for entry in automated:
            saved = json.loads(bound_file(root, entry['annotation_file']))
            source = pool.get(entry['case_id'])
            if entry.get('source_revision_manifest'):
                from scripts.prepare_training_source_revision import load_revision
                matches = [r for r in load_revision(root, entry['source_revision_manifest'])
                           if r['case_id'] == entry['case_id']]
                if len(matches) != 1:
                    raise ValueError('Revised automatic source must resolve exactly once')
                require_revision_identity(source, matches[0])
                source = matches[0]
            if (not source or saved.get('case_id') != source['case_id'] or
                saved.get('split') != 'train' or saved.get('source_binding') != source['source_binding']):
                raise ValueError('Automatic annotation must match its existing training source')
            if complexity_exclusion(source) or source_review_hold(source):
                raise ValueError('Automatic admission cannot override signed source feedback')
            if (saved.get('status') != 'automatically_audited_pending_human_review' or
                not quality_passed(saved, source)):
                raise ValueError('Source-bound semantic audit is incomplete or failed')
            require_no_reported_support_defect(root, source, saved)
            if saved.get('audit_format_repair_attempts'):
                prior = json.loads(bound_file(root, saved['previous_invalid_audit_result']))
                if (saved['audit_format_repair_attempts'] != 1 or prior.get('audit_format_repair_attempts') or
                    prior.get('source_binding') != source['source_binding'] or
                    fingerprint(prior.get('five_dimensions')) != fingerprint(saved['five_dimensions']) or
                    not audit_structure_errors(prior['semantic_audit'], prior['five_dimensions'])):
                    raise ValueError('Audit format repair must preserve one malformed audit and the unchanged candidate')
            raw = {k: saved['five_dimensions'][k] for k in FiveDimensions.model_fields}
            rows.append(build_row(source, raw, binding, entry,
                                  'automatically_audited_pending_human_review',
                                  allow_single_actor_without_conflict=allow_single))
    if not rows:
        raise ValueError('No source-verified cases admitted for small-batch training')
    validate_splits(rows)
    rows.sort(key=lambda r: (-r['five_dimensions']['reasoning_depth'], fingerprint(r['group_id'])))
    return rows, policy


def public_counts(rows):
    result = {'protocol': PROTOCOL, 'training_stage': 'small_batch_formal',
            'training_cases': len(rows), 'split_counts': dict(Counter(r['split'] for r in rows)),
            'actual_depth_counts': dict(Counter(str(r['five_dimensions']['reasoning_depth']) for r in rows)),
            'review_status_counts': dict(Counter(r['review_status'] for r in rows)),
            'minimum_depth': 2, 'preferred_depth': 3, 'human_approvals_created': 0,
            'strict_three_level_benchmark_cases_added': 0,
            'heldout_cases_used_for_training': 0, 'independent_accuracy_verified': False}
    single_count = sum(len(r['five_dimensions']['candidate_names']) == 1 and
                       not r['five_dimensions']['contradictions'] for r in rows)
    if single_count:
        result['single_actor_without_conflict_cases'] = single_count
    return result


def load_dataset(root, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('protocol') != PROTOCOL or manifest.get('training_stage') != 'small_batch_formal':
        raise ValueError('Expected the revised small-batch manifest')
    expected, _ = authorized_rows(root, manifest['policy_file'])
    rows = [json.loads(l) for l in bound_file(root, manifest['dataset']).decode().splitlines() if l.strip()]
    if fingerprint(rows) != fingerprint(expected):
        raise ValueError('Small-batch rows differ from their authorized source versions')
    if manifest.get('single_actor_without_conflict_cases', 0) != sum(
        len(r['five_dimensions']['candidate_names']) == 1 and not r['five_dimensions']['contradictions']
        for r in rows):
        raise ValueError('Single-actor manifest count differs from the actual annotation')
    if (manifest.get('dataset_binding') != fingerprint([r['record_binding'] for r in rows]) or
        any(manifest.get(k) != v for k, v in public_counts(rows).items())):
        raise ValueError('Small-batch manifest counts or binding changed')
    return rows, manifest
