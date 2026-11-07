#!/usr/bin/env python3
"""Publish verified local run totals without adding repeated training as cases."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.auxiliary_training import bound_file
from app.five_dimensions import fingerprint, record_binding, validate_splits
from app.small_batch_training import PROTOCOL, public_counts, require_no_reported_support_defect
from scripts.train_local import observed_metrics
from scripts.train_auxiliary import expanded_examples


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run(folder, superseded_training_sha):
    report = json.loads((folder / 'run.json').read_text())
    if report['protocol'] != PROTOCOL or report['run_id'] != folder.name:
        raise ValueError('Run identity differs from its folder')
    if report['stage'] != 'completed':
        return None, set()
    manifest = json.loads(bound_file(ROOT, report['source_manifest']))
    bound_file(ROOT, manifest['policy_file'])
    rows = [json.loads(line) for line in bound_file(ROOT, manifest['dataset']).decode().splitlines()]
    validate_splits(rows)
    if (any(r['split'] != 'train' or r['record_binding'] != record_binding(r) for r in rows) or
        any(report.get(k) != v for k, v in public_counts(rows).items()) or
        report['dataset_binding'] != fingerprint([r['record_binding'] for r in rows])):
        raise ValueError('Historical frozen training rows or counts differ')
    # Verify recorded historical inputs. New human feedback may change present
    # admission; this history does not grant permission to train that case again.
    for relative, sha in report['training_file_sha256'].items():
        if digest(folder / relative) != sha:
            raise ValueError('Historical training input changed')
    if any(digest(folder / 'code' / relative) != sha
           for relative, sha in report['code_file_sha256'].items()):
        raise ValueError('Historical code snapshot changed')
    if (digest(folder / 'policy.private.json') != manifest['policy_file']['sha256'] or
        (folder / 'data/valid.jsonl').read_bytes() != (folder / 'data/train.jsonl').read_bytes().splitlines(keepends=True)[0]):
        raise ValueError('Policy snapshot or training replay probe differs')
    metrics = observed_metrics(folder / 'training.log')
    if (any(report.get(k) != v for k, v in metrics.items()) or
        metrics['completed_micro_iterations'] != report['iters'] or
        report['iters'] % 4 or report['optimizer_updates'] != report['iters'] // 4):
        raise ValueError('Reported updates differ from actual training log')
    from safetensors.numpy import load_file
    adapter = folder / 'adapters/adapters.safetensors'
    tensors = load_file(str(adapter))
    nonzero = sum(bool((v != 0).any()) for k, v in tensors.items() if k.endswith('lora_b'))
    if (not nonzero or nonzero != report['nonzero_lora_b_tensors'] or
        digest(adapter) != report['adapter_sha256'] or not report['actual_weight_update_verified']):
        raise ValueError('Historical adapter differs or has no learned LoRA B weights')
    smoke_path = folder / 'offline_smoke_report.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    if smoke and (smoke.get('adapter_sha256') != report['adapter_sha256'] or
                  not smoke.get('passed') or smoke.get('network_attempts') != 0):
        raise ValueError('Archived offline smoke failed or belongs to another adapter')
    keys = ('run_id', 'training_cases', 'actual_depth_counts', 'review_status_counts',
            'train_examples', 'iters', 'optimizer_updates', 'nonzero_lora_b_tensors',
            'adapter_sha256', 'observed_peak_memory_gb', 'observed_validation_token_losses',
            'dataset_binding', 'finished_at')
    public = {k: report[k] for k in keys}
    held_labels = 0
    for row in rows:
        try:
            require_no_reported_support_defect(ROOT, row, {'five_dimensions': row['five_dimensions']})
        except ValueError as exc:
            if not str(exc).startswith('Reported source-support defect:'):
                raise
            held_labels += 1
    derived = ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in
                      expanded_examples(rows, include_singletons=report.get('single_evidence_training_variants', False))).encode()
    current_sha = hashlib.sha256(derived).hexdigest()
    public.update(single_actor_without_conflict_cases=report.get('single_actor_without_conflict_cases', 0),
                  frozen_input_and_adapter_hashes_verified=True,
                  offline_smoke_passed=bool(smoke),
                  conflict_resolution_citations_superseded=report['training_file_sha256']['data/train.jsonl'] == superseded_training_sha,
                  training_file_sha256=report['training_file_sha256']['data/train.jsonl'],
                  current_derived_training_file_sha256=current_sha,
                  annotation_source_support_holds=held_labels,
                  training_targets_superseded=bool(held_labels) or report['training_file_sha256']['data/train.jsonl'] != current_sha,
                  independent_accuracy_verified=False)
    return public, {r['group_id'] for r in rows}


def main():
    target_audit = json.loads((ROOT / 'data/small_batch_target_audit.json').read_text())
    runs = []; groups = set(); unfinished = []
    for path in sorted((ROOT / 'artifacts/small_batch_formal').glob('*/run.json')):
        run, used = verify_run(path.parent, target_audit['old_training_file_sha256'])
        if run:
            runs.append(run); groups.update(used)
        else:
            unfinished.append({'run_id': path.parent.name, 'stage': json.loads(path.read_text())['stage']})
    report = {'protocol': PROTOCOL, 'generated_at': datetime.now(timezone.utc).isoformat(),
              'completed_runs': len(runs), 'distinct_training_groups_across_completed_runs': len(groups),
              'run_count_is_not_case_count': True, 'human_approvals_created': 0,
              'strict_three_level_benchmark_cases_added': 0, 'heldout_cases_used_for_training': 0,
              'independent_accuracy_verified': False, 'runs': runs, 'unfinished_runs': unfinished}
    (ROOT / 'data/small_batch_run_history.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'verified_completed_runs': len(runs), 'distinct_training_groups': len(groups),
                      'unfinished_runs': len(unfinished)}), flush=True)


if __name__ == '__main__':
    main()
