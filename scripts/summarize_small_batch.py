#!/usr/bin/env python3
"""Aggregate actual small-batch diagnostics without counting retries as cases."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize(records):
    latest = {}
    for row in records:
        if row.get('split') != 'train':
            raise ValueError('Small-batch summaries must contain training diagnostics only')
        group = row['group_id']
        if group not in latest or row.get('finished_at', '') > latest[group].get('finished_at', ''):
            latest[group] = row
    values = list(latest.values())
    return {'diagnostic_result_files': len(records), 'distinct_training_groups': len(values),
            'distinct_source_versions': len({(r['group_id'], r['source_binding']) for r in records}),
            'latest_status_counts': dict(Counter(r['status'] for r in values)),
            'accepted_actual_depth_counts': dict(Counter(str(r['five_dimensions']['reasoning_depth'])
                for r in values if r['status'] == 'automatically_audited_pending_human_review')),
            'diagnostic_results_by_audit_protocol': dict(Counter(r.get('semantic_audit_protocol', 'legacy-global-flags') for r in records)),
            'model_annotation_generations': sum(bool(r.get('raw_output')) for r in records),
            'semantic_audit_calls_including_format_repair': sum(bool(r.get('semantic_audit_raw')) + r.get('audit_format_repair_attempts', 0) for r in records),
            'archived_runtime_only_failures': sum(bool(r.get('previous_runtime_failure')) for r in records),
            'audit_format_repairs': sum(r.get('audit_format_repair_attempts', 0) for r in records),
            'heldout_semantics_read': False, 'human_approvals_created': 0,
            'independent_accuracy_verified': False}


def main():
    records = [json.loads(p.read_text()) for p in (ROOT / 'data/review/small_batch_annotations').glob('*/*.json')]
    report = summarize(records)
    report['generated_at'] = datetime.now(timezone.utc).isoformat()
    dataset = ROOT / 'data/small_batch_dataset_report.json'
    if dataset.exists():
        data = json.loads(dataset.read_text())
        report.update(current_formal_training_cases=data['training_cases'],
                      current_formal_training_depth_counts=data['actual_depth_counts'],
                      current_formal_training_dataset_binding=data['dataset_binding'])
    target = ROOT / 'data/small_batch_curation_report.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
