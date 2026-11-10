"""Release readiness must describe and enforce the actual configured weights."""
import json
import pytest
from app.model_release import file_sha256, inspect_adapter, require_adapter
from scripts.activate_local_adapter import validate_release, update_env


def fixture_release(root):
    run = root / 'artifacts/small_batch_formal/test-run'
    adapter = run / 'adapters'
    adapter.mkdir(parents=True)
    (adapter / 'adapters.safetensors').write_bytes(b'synthetic-weight-fixture')
    (adapter / 'adapter_config.json').write_text('{"num_layers":8}')
    digest = file_sha256(adapter / 'adapters.safetensors')
    report = {'run_id': 'test-run', 'stage': 'completed', 'small_batch_training_completed': True,
              'actual_weight_update_verified': True, 'nonzero_lora_b_tensors': 56,
              'completed_micro_iterations': 88, 'iters': 88, 'heldout_cases_used_for_training': 0,
              'adapter_sha256': digest, 'training_cases': 5, 'train_examples': 86, 'optimizer_updates': 22}
    (run / 'run.json').write_text(json.dumps(report))
    smoke = root / 'smoke.json'
    smoke.write_text(json.dumps({'passed': True, 'adapter_sha256': digest, 'network_attempts': 0,
                                'cloud_ledger_unchanged': True, 'empty_evidence_injection_boundary_passed': True}))
    return run, adapter, smoke


def test_ready_adapter_and_pinned_files(tmp_path):
    run, adapter, smoke = fixture_release(tmp_path)
    report, status = validate_release(tmp_path, run, smoke)
    assert status['training_verified'] and status['training_cases'] == 5
    assert require_adapter(adapter, status['adapter_sha256'], status['config_sha256'])['ready']
    (adapter / 'adapter_config.json').write_text('{"num_layers":4}')
    with pytest.raises(ValueError):
        require_adapter(adapter, status['adapter_sha256'], status['config_sha256'])
    (adapter / 'adapters.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError): require_adapter(adapter, report['adapter_sha256'])


@pytest.mark.parametrize('defect', ['missing', 'incomplete', 'heldout', 'smoke', 'smoke_weights'])
def test_activation_rejects_unverified_or_mismatched_runs(tmp_path, defect):
    run, adapter, smoke = fixture_release(tmp_path)
    if defect == 'missing': (adapter / 'adapters.safetensors').unlink()
    elif defect in ('incomplete', 'heldout'):
        report = json.loads((run / 'run.json').read_text())
        report['stage' if defect == 'incomplete' else 'heldout_cases_used_for_training'] = 'running' if defect == 'incomplete' else 1
        (run / 'run.json').write_text(json.dumps(report))
    else:
        report = json.loads(smoke.read_text())
        report['passed' if defect == 'smoke' else 'adapter_sha256'] = False if defect == 'smoke' else 'wrong'
        smoke.write_text(json.dumps(report))
    with pytest.raises(ValueError): validate_release(tmp_path, run, smoke)


def test_base_and_unverified_adapter_are_not_labelled_trained(tmp_path):
    assert inspect_adapter(None) == {'ready': True, 'mode': 'base', 'training_verified': False}
    run, adapter, _ = fixture_release(tmp_path)
    (run / 'run.json').unlink()
    assert inspect_adapter(adapter)['ready'] and not inspect_adapter(adapter)['training_verified']
    assert not inspect_adapter(tmp_path / 'missing')['ready']


def test_atomic_config_update_preserves_unrelated_settings(tmp_path):
    path = tmp_path / '.env'
    path.write_text('# local\nALLOW_REGISTRATION=false\nLOCAL_ADAPTER_PATH=old\nLOCAL_ADAPTER_PATH=duplicate\n')
    update_env(path, {'LOCAL_ADAPTER_PATH': 'new', 'LOCAL_ADAPTER_SHA256': 'test'})
    content = path.read_text()
    assert 'ALLOW_REGISTRATION=false' in content and content.count('LOCAL_ADAPTER_PATH=') == 1
    assert 'LOCAL_ADAPTER_PATH=new' in content and path.stat().st_mode & 0o777 == 0o600
