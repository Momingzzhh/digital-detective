"""Explicitly accepted corrected drafts, isolated from the complex benchmark."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from .five_dimensions import (VERSION, fingerprint, ground_truth, record_binding,
                              validate_auxiliary_annotation, validate_splits)

PROTOCOL = 'user-accepted-auxiliary-calibration-v1'
SOURCE_KEYS = ('case_id', 'group_id', 'split', 'source', 'evidence',
               'sentence_spans', 'source_fact_sha256')


def bound_file(root, entry):
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Auxiliary provenance must stay inside the project')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry['sha256']:
        raise ValueError('Auxiliary provenance file changed')
    return content


def build_row(source, draft, authorization_binding, provenance):
    if source['split'] != 'train' or draft['split'] != 'train':
        raise ValueError('Auxiliary data must use existing training groups')
    if source['source_binding'] != fingerprint({k: source[k] for k in SOURCE_KEYS}):
        raise ValueError('Auxiliary source binding changed')
    if any(source[k] != draft[k] for k in ('case_id', 'group_id', 'source_binding')):
        raise ValueError('Corrected draft belongs to a different source')
    if draft.get('formal_approval') or draft.get('human_reviewed'):
        raise ValueError('Auxiliary acceptance cannot synthesize formal review')
    from scripts.export_review_bundle import verify
    verify(source, {'raw_record': source['raw_record']})
    annotation = validate_auxiliary_annotation(source, draft['raw_annotation'], source['sentence_spans'])
    depth = annotation['reasoning_depth']
    if depth != draft['computed_graph_depth'] or depth != draft['human_reviewed_maximum_depth']:
        raise ValueError('Corrected draft depth differs from the recorded actual depth')
    row = copy.deepcopy(source)
    row.update(annotation_version=VERSION, five_dimensions=annotation,
               dataset_role='auxiliary_calibration', auxiliary_protocol=PROTOCOL,
               eligible_for_formal_benchmark=False, formal_approval=False,
               human_reviewed=False, user_accepted_for_auxiliary_training=True,
               authorization_binding=authorization_binding,
               auxiliary_provenance=provenance, status='auxiliary_user_accepted')
    row['hidden_gold'] = ground_truth(row)
    row['record_binding'] = record_binding(row)
    verify(row, {'raw_record': row['raw_record']})
    return row


def authorized_rows(root, authorization_entry):
    authorization = json.loads(bound_file(root, authorization_entry))
    if (authorization.get('protocol') != PROTOCOL or
        authorization.get('authority') != 'human_user_message_in_this_chat' or
        authorization.get('scope') != 'auxiliary_training_preserve_actual_depth' or
        authorization.get('formal_benchmark_minimum_depth') != 3 or
        authorization.get('formal_approvals_created') != 0 or
        not authorization.get('user_statements')):
        raise ValueError('Explicit auxiliary training authorization is required')
    binding = fingerprint(authorization)
    rows = []
    for entry in authorization['accepted_corrected_drafts']:
        sources = [json.loads(line) for line in bound_file(root, entry['source_dataset']).decode().splitlines() if line.strip()]
        matches = [r for r in sources if r['source_binding'] == entry['source_binding']]
        if len(matches) != 1:
            raise ValueError('Accepted corrected source must resolve exactly once')
        draft = json.loads(bound_file(root, entry['corrected_draft']))
        rows.append(build_row(matches[0], draft, binding, entry))
    if not rows:
        raise ValueError('No concrete accepted corrected drafts')
    validate_splits(rows)
    return rows, authorization


def load_dataset(root, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest.get('protocol') != PROTOCOL or manifest.get('formal_cases_added') != 0 or
        manifest.get('eligible_for_formal_benchmark') is not False):
        raise ValueError('Auxiliary manifest must remain separate from formal data')
    expected, _ = authorized_rows(root, manifest['authorization_file'])
    rows = [json.loads(line) for line in bound_file(root, manifest['dataset']).decode().splitlines() if line.strip()]
    if fingerprint(rows) != fingerprint(expected):
        raise ValueError('Auxiliary rows differ from the authorized corrected versions')
    if manifest.get('dataset_binding') != fingerprint([r['record_binding'] for r in rows]):
        raise ValueError('Auxiliary dataset binding changed')
    return rows, manifest


def public_counts(rows):
    return {'protocol': PROTOCOL, 'auxiliary_cases': len(rows),
            'split_counts': dict(Counter(r['split'] for r in rows)),
            'actual_depth_counts': dict(Counter(str(r['five_dimensions']['reasoning_depth']) for r in rows)),
            'eligible_for_formal_benchmark': False, 'formal_cases_added': 0,
            'formal_approvals_created': 0}
