#!/usr/bin/env python3
"""Freeze a new policy-bound training-only dataset without opening heldout text."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.five_dimensions import fingerprint
from app.small_batch_training import authorized_rows, load_dataset, public_counts


def main(args):
    path = (ROOT / args.policy).resolve()
    entry = {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    rows, policy = authorized_rows(ROOT, entry)
    out = ROOT / 'data/processed/small_batch_formal' / fingerprint(policy)[:16]
    out.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
    dataset = out / 'data.private.jsonl'
    manifest_path = out / 'manifest.private.json'
    manifest = public_counts(rows) | {'policy_file': entry,
        'dataset': {'path': str(dataset.relative_to(ROOT)), 'sha256': hashlib.sha256(content).hexdigest()},
        'dataset_binding': fingerprint([r['record_binding'] for r in rows])}
    for file, value in ((dataset, content), (manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())):
        if file.exists() and file.read_bytes() != value:
            raise ValueError('Refusing to overwrite a different frozen small-batch artifact')
        file.write_bytes(value)
    load_dataset(ROOT, manifest_path)
    report = public_counts(rows) | {'manifest': str(manifest_path.relative_to(ROOT)),
                                  'dataset_binding': manifest['dataset_binding']}
    (ROOT / 'data/small_batch_dataset_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return manifest_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    main(parser.parse_args())
