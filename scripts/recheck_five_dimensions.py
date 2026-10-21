#!/usr/bin/env python3
"""Revalidate saved outputs with recorded identifier-only repairs; no model call."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.five_dimensions import validate_annotation
from scripts.annotate_five_dimensions import normalize_references
from scripts.annotation_lock import acquire_annotation_lock


def main():
    acquire_annotation_lock()
    pool = ROOT / 'data/processed/five_dimensions_v1/candidates.private.jsonl'
    rows = {r['case_id']: r for r in map(json.loads, pool.open())}
    counts = Counter()
    for path in sorted((ROOT / 'data/review/five_dimensions_v1').glob('*.json')):
        saved = json.loads(path.read_text()); row = rows.get(saved['case_id'])
        if not row or not saved.get('raw_annotation'): continue
        if saved.get('source_binding') != row['source_binding']: raise ValueError('Source binding changed')
        try:
            normalized, repairs = normalize_references(saved['raw_annotation'], row['sentence_spans'])
            annotation = validate_annotation(row, normalized, row['sentence_spans'])
            update = {'status': 'machine_annotated_unverified', 'five_dimensions': annotation}
        except ValueError as exc:
            update = {'status': 'complexity_or_reference_rejected', 'error': str(exc)[:300]}
        except (KeyError, TypeError):
            counts['unrecognized_legacy_structure_unchanged'] += 1
            continue
        if saved['status'] != update['status'] or saved.get('identifier_repairs', []) != repairs:
            before = saved['status']
            saved.update(update, identifier_repairs=repairs,
                         rechecked_at=datetime.now(timezone.utc).isoformat(),
                         previous_status_before_recheck=before)
            if update['status'] == 'machine_annotated_unverified': saved.pop('error', None)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(path)
            counts[f'{before}->{update["status"]}'] += 1
    print(json.dumps({'rechecked_changes': dict(counts), 'model_calls': 0, 'human_approvals': 0}))


if __name__ == '__main__': main()
