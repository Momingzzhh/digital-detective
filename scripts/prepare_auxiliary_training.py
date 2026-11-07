#!/usr/bin/env python3
"""Bind explicit user acceptance to concrete corrected training-only drafts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.auxiliary_training import authorized_rows, load_dataset, public_counts
from app.five_dimensions import fingerprint


def main(args):
    path = (ROOT / args.authorization).resolve()
    entry = {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    rows, authorization = authorized_rows(ROOT, entry)
    out = ROOT / 'data/processed/auxiliary_calibration' / fingerprint(authorization)[:16]
    out.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
    dataset = out / 'data.private.jsonl'
    manifest_path = out / 'manifest.private.json'
    manifest = public_counts(rows) | {'authorization_file': entry,
        'dataset': {'path': str(dataset.relative_to(ROOT)), 'sha256': hashlib.sha256(content).hexdigest()},
        'dataset_binding': fingerprint([r['record_binding'] for r in rows])}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    for file, value in ((dataset, content), (manifest_path, manifest_bytes)):
        if file.exists() and file.read_bytes() != value:
            raise ValueError('Refusing to overwrite a different auxiliary artifact')
        file.write_bytes(value)
    load_dataset(ROOT, manifest_path)
    report = public_counts(rows) | {'manifest': str(manifest_path.relative_to(ROOT)),
                                 'dataset_binding': manifest['dataset_binding']}
    (ROOT / 'data/auxiliary_calibration_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', required=True)
    main(parser.parse_args())
