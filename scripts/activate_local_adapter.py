#!/usr/bin/env python3
"""Activate a completed small-batch run after matching its offline proof."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.model_release import file_sha256, require_adapter


def validate_release(root, run_dir, smoke_path):
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to((root / 'artifacts/small_batch_formal').resolve()):
        raise ValueError('Select a run inside artifacts/small_batch_formal')
    run = json.loads((run_dir / 'run.json').read_text())
    if (run.get('stage') != 'completed' or not run.get('small_batch_training_completed')
            or not run.get('actual_weight_update_verified') or not run.get('nonzero_lora_b_tensors')
            or run.get('completed_micro_iterations') != run.get('iters')
            or run.get('heldout_cases_used_for_training') != 0):
        raise ValueError('The run must have completed verified training without heldout data')
    directory = run_dir / 'adapters'
    status = require_adapter(directory, run['adapter_sha256'])
    smoke = json.loads(smoke_path.read_text())
    if (smoke.get('adapter_sha256') != status['adapter_sha256'] or smoke.get('passed') is not True
            or smoke.get('network_attempts') != 0 or smoke.get('cloud_ledger_unchanged') is not True
            or smoke.get('empty_evidence_injection_boundary_passed') is not True):
        raise ValueError('A passing offline and empty-evidence check for these exact weights is required')
    return run, status


def update_env(path, values):
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines if line.split('=', 1)[0].strip() not in values]
    lines.extend(f'{key}={value}' for key, value in values.items())
    temp = path.with_suffix('.activation.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write('\n'.join(lines) + '\n')
    temp.replace(path)


def main(args):
    directory = (ROOT / args.run).resolve()
    smoke = (ROOT / args.smoke_report).resolve()
    run, status = validate_release(ROOT, directory, smoke)
    from app.five_dimensions import fingerprint
    from app.local_model import verify_model_files
    from app.local_resource import model_resource
    os.environ['LOCAL_MODEL_PATH'] = str(ROOT / run['model_path'])
    with model_resource('activate_adapter'):
        if fingerprint(verify_model_files()) != run['base_manifest_binding']:
            raise ValueError('Base model differs from the training base')
        import numpy as np
        from safetensors.numpy import load_file
        tensors = load_file(str(directory / 'adapters/adapters.safetensors'))
        learned = sum(bool(np.any(value != 0)) for key, value in tensors.items() if key.endswith('lora_b'))
        if learned != run['nonzero_lora_b_tensors'] or not all(np.isfinite(v).all() for v in tensors.values()):
            raise ValueError('Learned tensors do not match the completed training report')
        # Save the old local configuration; never publish its contents.
        config = ROOT / '.env'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = ROOT / 'runtime/activation_backups' / stamp
        backup.mkdir(parents=True)
        if config.exists():
            target = backup / 'env.private'
            target.write_bytes(config.read_bytes()); target.chmod(0o600)
        update_env(config, {'MODEL_BACKEND': 'local', 'LOCAL_MODEL_PATH': run['model_path'],
                           'LOCAL_ADAPTER_PATH': str((directory / 'adapters').relative_to(ROOT)),
                           'LOCAL_ADAPTER_SHA256': status['adapter_sha256'],
                           'LOCAL_ADAPTER_CONFIG_SHA256': status['config_sha256']})
        report = {'activated_at': datetime.now(timezone.utc).isoformat(),
                  'stage': 'local_usable_small_batch', 'adapter_activated': True,
                  'adapter_path': str((directory / 'adapters').relative_to(ROOT)),
                  **{k: v for k, v in status.items() if k not in ('ready', 'mode')},
                  'base_manifest_binding': run['base_manifest_binding'],
                  'nonzero_lora_b_tensors_rechecked': learned,
                  'offline_report_sha256': file_sha256(smoke),
                  'independent_accuracy_verified': False, 'benchmark_acceptance_completed': False}
        (ROOT / 'data/local_release_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--smoke-report', required=True)
    main(parser.parse_args())
