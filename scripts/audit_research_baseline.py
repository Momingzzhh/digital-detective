#!/usr/bin/env python3
"""Read-only baseline inventory; private contents and group identities stay in artifacts/."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    result = {'generated_at': datetime.now(timezone.utc).isoformat(),
              'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'worktree_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
              'files': {}, 'historical_training_groups': {}, 'missing': [], 'hash_mismatches': []}

    def inspect(relative, expected=None):
        path = ROOT / relative
        key = str(path.relative_to(ROOT))
        if key not in result['files']:
            result['files'][key] = {'accessible': path.is_file()}
            if path.is_file():
                result['files'][key].update(bytes=path.stat().st_size, sha256=sha(path))
            else:
                result['missing'].append(key)
        if expected and result['files'][key].get('sha256') != expected:
            result['hash_mismatches'].append(key)
        return path

    for manifest, directory in [('local_model_manifest.json', 'qwen3-4b-instruct-2507-4bit'),
                                ('local_curator_manifest.json', 'qwen3-14b-mlx-4bit')]:
        data = json.loads(inspect('data/' + manifest).read_text())
        for name, info in data['files'].items():
            inspect('models/' + directory + '/' + name, info['sha256'])
    for report in ['local_release_report.json', 'latest_small_batch_training_report.json',
                   'latest_auxiliary_training_report.json', 'latest_training_report.json',
                   'dataset_report.json', 'five_dimensions_report.json', 'small_batch_dataset_report.json',
                   'follow_up_aggregate_report.json', 'small_batch_run_history.json',
                   'mnbvc_download_manifest.json', 'oldbailey_download_manifest.json']:
        inspect('data/' + report)
    for report in ['local_release_report.json', 'latest_small_batch_training_report.json']:
        data = json.loads((ROOT / 'data' / report).read_text())
        inspect(data['adapter_path'] + '/adapters.safetensors', data['adapter_sha256'])
        inspect(data['adapter_path'] + '/adapter_config.json', data.get('config_sha256'))
    # Hash all historical train views without displaying or publishing source text.
    patterns = ['data/processed/**/manifest*.json', 'data/processed/**/pool_manifest.json',
                'data/processed/*candidates.jsonl', 'artifacts/**/data/train.jsonl',
                'data/processed/**/data.private.jsonl']
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            if 'research_v2' in path.parts:
                continue
            inspect(path.relative_to(ROOT))
            if path.suffix != '.jsonl':
                continue
            for line in path.open():
                row = json.loads(line)
                if row.get('split') == 'train' or path.name == 'train.jsonl':
                    group = row.get('group_id')
                    if group:
                        entry = result['historical_training_groups'].setdefault(group, {'case_ids': [], 'files': []})
                        if row.get('case_id') and row['case_id'] not in entry['case_ids']:
                            entry['case_ids'].append(row['case_id'])
                        relative = str(path.relative_to(ROOT))
                        if relative not in entry['files']:
                            entry['files'].append(relative)
    result['note'] = 'Includes frozen training candidates conservatively; not all groups were used for weight updates. No heldout semantics reviewed.'
    destination = ROOT / 'artifacts/research_v2/baseline_manifest.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'manifest': str(destination.relative_to(ROOT)), 'sha256': sha(destination),
                      'files': len(result['files']), 'training_group_identities': len(result['historical_training_groups']),
                      'missing': result['missing'], 'hash_mismatches': result['hash_mismatches']}))
    return int(bool(result['missing'] or result['hash_mismatches']))


if __name__ == '__main__':
    raise SystemExit(main())
