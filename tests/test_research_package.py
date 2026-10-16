import json
import zipfile

import pytest
from test_v3_source import module


def fixture_export(tmp_path):
    pack = module('package_research_bundle')
    row = {'case_id': 'fixture', 'group_id': 'group', 'split': 'test',
           'evidence': [{'id': 'E1', 'kind': '证言', 'text': '合成机制测试线索'}],
           'hidden_gold': {'charges': ['隐藏罪名']}, 'source': {'title': '隐藏全文'}}
    view = {k: row[k] for k in ('case_id', 'group_id', 'split', 'evidence')}
    pack.write_rows(tmp_path / 'machine_curated.unverified.jsonl', [row])
    for split in pack.SPLITS:
        pack.write_rows(tmp_path / 'visible' / (split + '.jsonl'), [view] if split == 'test' else [])
        pack.write_rows(tmp_path / 'hidden' / (split + '.jsonl'), [
            {k: row[k] for k in ('case_id', 'hidden_gold', 'source')}] if split == 'test' else [])
    manifest = {'machine_curated_cases': 1, 'artifacts': {
        str(p.relative_to(tmp_path)): {'sha256': pack.digest(p), 'bytes': p.stat().st_size}
        for p in tmp_path.rglob('*.jsonl')}}
    pack.write_json(tmp_path / 'manifest.json', manifest)
    return pack, view


def test_archive_and_public_report_do_not_add_private_answers(tmp_path):
    pack, view = fixture_export(tmp_path)
    pack.load_export(tmp_path, False)
    text = pack.visible_text(view, False)
    assert '合成机制测试线索' in text
    assert '隐藏罪名' not in text and '隐藏全文' not in text
    (tmp_path / 'visible/fixture.txt').write_text(text)
    target = tmp_path / 'visible-only.zip'
    pack.archive(tmp_path / 'visible', target)
    with zipfile.ZipFile(target) as archive:
        assert all(not name.startswith('hidden/') for name in archive.namelist())
        assert all('隐藏罪名' not in archive.read(name).decode() for name in archive.namelist())
    public = pack.public_report({'artifacts': {
        'visible-only.zip': {}, 'review-private.zip': {},
        'visible/text/test/fixture.txt': {}, 'review_private/test.jsonl': {}}})
    assert set(public['artifacts']) == {'visible-only.zip', 'review-private.zip'}


def test_package_rejects_changed_frozen_export(tmp_path):
    pack, _ = fixture_export(tmp_path)
    path = tmp_path / 'visible/test.jsonl'
    path.write_text(path.read_text().replace('合成机制测试线索', '被替换的线索'))
    with pytest.raises(ValueError, match='Export changed'):
        pack.load_export(tmp_path, False)


def test_package_rejects_hidden_fields_in_visible_input(tmp_path):
    pack, view = fixture_export(tmp_path)
    view['hidden_gold'] = {'charges': ['隐藏罪名']}
    with pytest.raises(ValueError, match='Unexpected visible fields'):
        pack.visible_text(view, False)


def test_package_rejects_group_shared_across_splits(tmp_path):
    pack, _ = fixture_export(tmp_path)
    path = tmp_path / 'machine_curated.unverified.jsonl'
    row = json.loads(path.read_text())
    pack.write_rows(path, [row, row | {'case_id': 'other', 'split': 'train'}])
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['artifacts'][path.name] = {'sha256': pack.digest(path), 'bytes': path.stat().st_size}
    pack.write_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Duplicate case/group'):
        pack.load_export(tmp_path, False)
