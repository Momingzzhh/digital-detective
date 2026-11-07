import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from app.auxiliary_training import (PROTOCOL, SOURCE_KEYS, authorized_rows, build_row,
                                    load_dataset, public_counts)
from app.five_dimensions import fingerprint, validate_annotation, validate_auxiliary_annotation
from test_five_dimensions import fixture


def inputs():
    source, ann, spans = fixture()
    ann['logic_chain'][2].update(independently_supported=True, depends_on_steps=[], dependency_reason='')
    full = ''.join(e['text'] for e in source['evidence'])
    cursor = 0
    for e in source['evidence']:
        e.update(start=cursor, end=cursor+len(e['text']), quote=e['text'])
        cursor = e['end']
    sha = hashlib.sha256(full.encode()).hexdigest()
    source.update(source={'sha256_fulltext': sha}, source_fact_sha256=sha,
                  raw_record={'qw': full, 'ajjbqk': full},
                  hidden_gold={'charges': ['测试罪名'], 'adjudicated_perpetrators': ['甲']})
    source['source_binding'] = fingerprint({k: source[k] for k in SOURCE_KEYS})
    draft = {k: source[k] for k in ('case_id', 'group_id', 'source_binding', 'split')}
    draft.update(raw_annotation=ann, computed_graph_depth=2, human_reviewed_maximum_depth=2,
                 formal_approval=False, human_reviewed=False)
    return source, draft


def test_auxiliary_keeps_actual_depth_while_formal_gate_stays_three():
    source, draft = inputs()
    with pytest.raises(ValueError, match='three linked'):
        validate_annotation(source, draft['raw_annotation'], source['sentence_spans'])
    assert validate_auxiliary_annotation(source, draft['raw_annotation'], source['sentence_spans'])['reasoning_depth'] == 2


@pytest.mark.parametrize('defect', ['heldout', 'offset', 'role', 'incomplete', 'false_depth', 'different_source'])
def test_auxiliary_does_not_relax_source_and_coverage_checks(defect):
    source, draft = inputs()
    if defect == 'heldout': source['split'] = draft['split'] = 'test'
    elif defect == 'offset': source['sentence_spans'][0]['end'] -= 1
    elif defect == 'role': draft['raw_annotation']['actor_assessments'][0]['name'] = '不存在'
    elif defect == 'incomplete': draft['raw_annotation']['complete_actor_coverage'] = False
    elif defect == 'false_depth': draft['human_reviewed_maximum_depth'] = 3
    else: draft['source_binding'] = 'other'
    with pytest.raises(ValueError): build_row(source, draft, 'auth', {})


def package(root):
    source, draft = inputs()
    def write(name, value, jsonl=False):
        path = root / name
        path.write_text(json.dumps(value, ensure_ascii=False) + ('\n' if jsonl else ''))
        return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    entry = {'source_binding': source['source_binding'],
             'source_dataset': write('source.jsonl', source, True),
             'corrected_draft': write('draft.json', draft)}
    authorization = {'protocol': PROTOCOL, 'authority': 'human_user_message_in_this_chat',
        'scope': 'auxiliary_training_preserve_actual_depth', 'formal_benchmark_minimum_depth': 3,
        'formal_approvals_created': 0, 'user_statements': ['机制测试授权，不是真实人工签字'],
        'accepted_corrected_drafts': [entry]}
    authorization_file = write('authorization.json', authorization)
    rows, _ = authorized_rows(root, authorization_file)
    manifest = public_counts(rows) | {'authorization_file': authorization_file,
        'dataset': write('data.jsonl', rows[0], True),
        'dataset_binding': fingerprint([r['record_binding'] for r in rows])}
    write('manifest.json', manifest)
    return rows, manifest


def test_exact_authorized_draft_is_bound_without_new_formal_approval(tmp_path):
    rows, manifest = package(tmp_path)
    loaded, _ = load_dataset(tmp_path, tmp_path / 'manifest.json')
    assert loaded == rows
    row = rows[0]
    assert row['five_dimensions']['reasoning_depth'] == 2
    assert row['user_accepted_for_auxiliary_training'] is True
    assert not row['formal_approval'] and not row['human_reviewed']
    assert not row['eligible_for_formal_benchmark']
    assert manifest['formal_cases_added'] == 0


@pytest.mark.parametrize('mutation', ['source', 'draft', 'dataset_flags', 'dataset_claim'])
def test_auxiliary_rejects_changed_files_even_when_dataset_hash_is_rewritten(tmp_path, mutation):
    rows, manifest = package(tmp_path)
    if mutation in ('source', 'draft'):
        path = tmp_path / ('source.jsonl' if mutation == 'source' else 'draft.json')
        path.write_text(path.read_text() + ' ')
    else:
        row = rows[0]
        if mutation == 'dataset_flags': row['eligible_for_formal_benchmark'] = True
        else: row['five_dimensions']['logic_chain'][0]['claim'] = '伪造训练标签'
        path = tmp_path / 'data.jsonl'
        path.write_text(json.dumps(row) + '\n')
        manifest['dataset']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): load_dataset(tmp_path, tmp_path / 'manifest.json')


def test_auxiliary_never_enters_formal_training_even_in_pilot_mode(tmp_path):
    rows, _ = package(tmp_path)
    from scripts.train_local import main
    with pytest.raises(ValueError, match='separate training entry point'):
        main(SimpleNamespace(dataset=str(tmp_path / 'data.jsonl'), pilot=True))


def test_simulated_prefixes_supervise_only_available_source_and_keep_refusals(tmp_path):
    rows, _ = package(tmp_path)
    from scripts.train_auxiliary import expanded_examples
    expanded = expanded_examples(rows)
    variants = {e['variant'] for e in expanded}
    assert 'all_evidence_withheld' in variants and 'synthetic_instruction_injection' in variants
    for example in expanded:
        assert example['messages'][-1]['role'] == 'assistant'
        visible = set()
        for message in example['messages']:
            if message['role'] != 'user': continue
            payload = json.loads(message['content'])
            visible.update(e['id'] for e in payload.get('evidence', payload.get('additional_evidence', [])))
        target = json.loads(example['messages'][-1]['content'])
        for field in ('findings', 'participants', 'reconstruction'):
            assert all(set(item['evidence_ids']) <= visible for item in target[field])
        if target['status'] == '证据不足':
            assert target['charges'] == [] and target['perpetrators'] == []
