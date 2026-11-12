import copy
import json
import pytest
from app.experiment_manifest import (REQUIRED, build_manifest, canonical,
                                     verify_resume, cache_read, cache_write)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('synthetic evidence only')
    config = {name: {'version': 'synthetic-v1'} for name in REQUIRED}
    config['seeds'] = {'python': 42, 'model': 42, 'environment': 42}
    return config, {'source': source}


@pytest.mark.parametrize('section', sorted(REQUIRED))
def test_each_critical_section_invalidates_resume(inputs, section):
    config, files = inputs
    saved = build_manifest(config, files)
    updated = copy.deepcopy(config)
    if section == 'seeds':
        updated[section]['model'] += 1
    else:
        updated[section]['version'] = 'different'
    assert build_manifest(updated, files)['run_id'] != saved['run_id']
    with pytest.raises(ValueError, match='changed'):
        verify_resume(saved, updated, files)


def test_order_and_relocation_do_not_change_identity(inputs, tmp_path):
    config, files = inputs
    saved = build_manifest(config, files)
    moved = tmp_path / 'moved.txt'
    moved.write_bytes(files['source'].read_bytes())
    assert build_manifest(dict(reversed(list(config.items()))), {'source': moved}) == saved
    assert verify_resume(saved, config, files) == saved


def test_cache_rejects_changed_input_and_tampered_payload(inputs, tmp_path):
    config, files = inputs
    saved = build_manifest(config, files)
    path = cache_write(tmp_path, saved, config, files, '../case', {'score': 0})
    assert path.parent.parent == tmp_path
    assert cache_read(tmp_path, saved, config, files, '../case') == {'score': 0}
    assert cache_read(tmp_path, saved, config, files, 'new-case') is None
    with pytest.raises(FileExistsError):
        cache_write(tmp_path, saved, config, files, '../case', {'score': 1})
    original = files['source'].read_bytes()
    files['source'].write_text('changed source')
    with pytest.raises(ValueError, match='changed'):
        cache_read(tmp_path, saved, config, files, '../case')
    files['source'].write_bytes(original)
    record = json.loads(path.read_text()); record['result']['score'] = 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='payload'):
        cache_read(tmp_path, saved, config, files, '../case')


def test_missing_files_and_forged_manifest_fail_closed(inputs):
    config, files = inputs
    saved = build_manifest(config, files)
    forged = copy.deepcopy(saved); forged['run_id'] += 'x'
    with pytest.raises(ValueError, match='digest'):
        verify_resume(forged, config, files)
    files['source'].unlink()
    with pytest.raises(FileNotFoundError):
        verify_resume(saved, config, files)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {1: 'lossy'}, (1, 2)])
def test_noncanonical_values_rejected(value):
    with pytest.raises(ValueError):
        canonical(value)


def test_config_cannot_silently_omit_sections(inputs):
    config, files = inputs
    del config['adapter']
    with pytest.raises(ValueError):
        build_manifest(config, files)
