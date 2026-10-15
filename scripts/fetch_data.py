#!/usr/bin/env python3
"""Fetch pinned LeCaRD originals locally; audit structure, NEVER certify solvability.

No API key or model calls. Raw judgments must stay out of public Git.
Usage: python scripts/fetch_data.py [--audit-only]
"""
import argparse
import base64
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/lecard'
REV = 'b1a4b72b2050a4aa643e6360d5ff3a6b1fd221e3'
REPO = 'https://github.com/myx666/LeCaRD'
SIZES = {'candidates1.zip': 54875800, 'candidates2.zip': 67985309}
GIT_BLOBS = {'candidates1.zip': 'd67d7c5eb0f6934c64db8adbcfc183451c1ce5f3',
             'candidates2.zip': '5c5be213e5fac875af4a378fcd04e031d6e2313a'}


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    artifacts, unique, duplicate_count = [], {}, 0
    for name, size in SIZES.items():
        path = RAW / name
        url = f'https://raw.githubusercontent.com/myx666/LeCaRD/{REV}/data/candidates/{name}'
        if not args.audit_only and not path.exists():
            temporary = path.with_suffix('.download')
            subprocess.run(['curl', '--fail', '--location', '--retry', '2',
                            '--connect-timeout', '15', '--max-time', '300',
                            '--output', str(temporary), url], check=True)
            data = temporary.read_bytes()
            if len(data) != size or blob_sha(data) != GIT_BLOBS[name]:
                raise ValueError(f'Integrity check failed: {name}')
            temporary.replace(path)
        if not path.exists():
            continue
        data = path.read_bytes()
        if len(data) != size or blob_sha(data) != GIT_BLOBS[name]:
            raise ValueError(f'Integrity check failed: {name}')
        count = 0
        with zipfile.ZipFile(path) as z:
            for member in sorted(z.namelist()):
                if not member.endswith('.json'):
                    continue
                d = json.loads(z.read(member))
                count += 1
                identifier = d.get('writId') or d.get('ajId')
                if not identifier:
                    identifier = hashlib.sha256(d['qw'].encode()).hexdigest()
                if identifier in unique:
                    duplicate_count += 1
                else:
                    unique[identifier] = (d, name, member)
        artifacts.append({'filename': name, 'url': url, 'bytes': size,
                          'sha256': hashlib.sha256(data).hexdigest(),
                          'git_blob_sha1': GIT_BLOBS[name], 'json_records': count})
    stats = collections.Counter()
    for d, _, _ in unique.values():
        for k, v in d.items():
            stats[f'nonempty_{k}'] += int(bool(v))
        full = d.get('qw', '')
        for k in ['证据', '证言', '供述', '鉴定', '监控', '指纹', 'DNA']:
            stats[f'mentions_{k}'] += int(k in full)
        stats['detailed_evidence_marker_heuristic'] += int(bool(re.search(
            r'(?:证言|供述|鉴定意见|辨认笔录|勘验笔录)[，,:：。].{20}', full)))
        stats['judgment_crime_regex'] += int(bool(re.search(
            r'犯[^，。；：]{1,30}罪', d.get('pjjg', ''))))
    audit = {
        'dataset': 'LeCaRD v1', 'repository': REPO, 'revision': REV,
        'upstream_license': 'MIT (repository); underlying judgments require separate rights/privacy assessment',
        'license_url': f'{REPO}/blob/{REV}/LICENSE',
        'original_provenance': 'China Judgments Online per upstream README; writId/ajId retained',
        'provenance_limit': 'Original court URLs are not supplied or live-verified; dataset archive is the reproducible source.',
        'artifacts': artifacts, 'unique_writ_ids': len(unique),
        'duplicate_writ_ids': duplicate_count, 'statistics': dict(stats),
        'human_verified_solvable_cases': 0,
        'meets_1000_fully_verified_investigation_cases': False,
        'limitations': [
            'This is a legal retrieval corpus, not an investigation or culprit-identification benchmark.',
            'Keyword counts and detailed-evidence regex are structural screening only, not a finding of sufficient evidence.',
            'Facts, titles, procedural history and evidence confessions may directly reveal the perpetrator or charge.',
            'Judicially found facts are retrospective conclusions; removing judgment fields alone does not prevent leakage.',
            'Some documents list evidence names without the evidence contents; do not invent missing contents.',
            'Candidate records repeat across queries; writId deduplication does not eliminate all same-case appeals/retrials.',
            'No human-adjudicated clue spans, perpetrator labels or chronology labels are supplied.',
            'Public pretraining contamination cannot be ruled out, even with a held-out downstream split.'
        ],
        'required_next_steps': [
            'Identify source-backed evidence spans and preserve exact document offsets.',
            'Separate initial observations, available evidence and hidden judicial outcome.',
            'Human review at least 1000 cases for evidence sufficiency, contradictory evidence and label correctness.',
            'Group related case numbers and proceedings before train/dev/test splitting.',
            'Publish code, checksums and source identifiers; keep raw sensitive judgments local.'
        ]
    }
    out = ROOT / 'data/source_audit.json'
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'audit': str(out), 'unique_writ_ids': len(unique),
                      'statistics': dict(stats)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
