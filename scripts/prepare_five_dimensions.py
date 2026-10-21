#!/usr/bin/env python3
"""Prepare modern, group-safe complexity candidates; no model calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.five_dimensions import VERSION, TARGET, fingerprint
from build_v3 import preflight, sentences
from export_review_bundle import write_rows, verify

INPUTS = ('candidates.jsonl', 'lecardv2_candidates.jsonl', 'lecardv2_mteb_candidates.jsonl', 'mnbvc_candidates.jsonl')


def main():
    audit_path = ROOT / 'data/processed/case_group_audit.json'
    audit = json.loads(audit_path.read_text())
    pool = []; groups = set(); counts = Counter()
    for name in INPUTS:
        path = ROOT / 'data/processed' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit['source_files_sha256'].get(name):
            raise ValueError('Cross-source group audit stale')
        for line in path.open():
            source = json.loads(line); row = preflight(source)
            if not row: continue
            counts['source_section_candidates'] += 1
            if len(row['evidence']) < 6 or len({e['kind'] for e in row['evidence']}) < 3: continue
            counts['six_items_three_types'] += 1
            linked = audit['cases'][row['case_id']]
            if linked['cross_split_conflict'] or linked['linked_group_id'] in groups: continue
            # Avoid making the initial Mac recipe depend on silent truncation.
            if sum(len(e['text']) for e in row['evidence']) > 7000:
                counts['long_source_reserved_for_later'] += 1; continue
            verify(row, source)
            row['source_group_id'] = row['group_id']; row['group_id'] = linked['linked_group_id']
            row['sentence_spans'] = sentences(row['evidence'])
            if len(row['sentence_spans']) < 3: continue
            row['annotation_version'] = VERSION
            row['status'] = 'awaiting_five_dimension_annotation'
            row['source_fact_excerpt'] = source['raw_record']['ajjbqk']
            row['source_judgment_excerpt'] = source['raw_record']['pjjg']
            row['raw_record'] = source['raw_record']
            row['source_binding'] = fingerprint({k: row[k] for k in
                ('case_id', 'group_id', 'split', 'source', 'evidence', 'sentence_spans', 'source_fact_sha256')})
            groups.add(row['group_id']); pool.append(row)
    # This order is frozen independently of any agent predictions or test score.
    pool.sort(key=lambda r: fingerprint(VERSION + ':' + r['group_id']))
    out = ROOT / 'data/processed/five_dimensions_v1'
    existing = out / 'pool_manifest.json'
    binding = fingerprint([r['source_binding'] for r in pool])
    if existing.exists() and json.loads(existing.read_text())['pool_binding'] != binding:
        raise ValueError('Candidate pool already frozen; create a new dataset version')
    write_rows(out / 'candidates.private.jsonl', pool)
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'version': VERSION,
        'target': TARGET, 'minimum_cases': 501, 'split_target': {'train': 420, 'dev': 90, 'test': 90},
        'pool_cases': len(pool), 'pool_split_counts': dict(Counter(r['split'] for r in pool)),
        'pool_binding': binding, 'group_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'screening_counts': dict(counts), 'machine_annotated': 0, 'human_reviewed': 0,
        'qualified_complex_cases': 0, 'weights_trained': False, 'validation_passed': False,
        'test_passed': False, 'first_round_threshold': 0.7,
        'note': 'Structural pool only. Real conflict/multiple investigated persons, necessary logic depth and sufficient clues still require annotation and review.'}
    existing.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'data/five_dimensions_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
