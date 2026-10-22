#!/usr/bin/env python3
"""Assemble exact 70/15/15 annotated groups, without pretending human approval."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.five_dimensions import fingerprint, record_binding, validate_splits, training_examples, view, validate_annotation, FiveDimensions, conclusion, TRAIN_SYSTEM
from app.five_dimensions import ground_truth
from app.annotation_quality import quality_passed
from app.research import CHECKS as HUMAN_CHECKS
from export_review_bundle import write_rows

BASE = ROOT / 'data/processed/five_dimensions_v2'
QUOTAS = {'train': 420, 'dev': 90, 'test': 90}


def main():
    pool = [json.loads(l) for l in (BASE / 'candidates.private.jsonl').open()]
    selected = []; counts = Counter(); states = Counter()
    from transformers import AutoTokenizer
    from app.local_model import model_path
    tokenizer = AutoTokenizer.from_pretrained(str(model_path()), local_files_only=True, trust_remote_code=False)
    for row in pool:
        path = ROOT / 'data/review/five_dimensions_v2' / (row['case_id'] + '.json')
        if not path.exists(): continue
        saved = json.loads(path.read_text()); states[saved['status']] += 1
        if saved['source_binding'] != row['source_binding']: raise ValueError('Source binding changed')
        if saved['status'] != 'machine_annotated_unverified' or counts[row['split']] >= QUOTAS[row['split']]: continue
        if not quality_passed(saved,row):
            states['semantic_audit_missing_or_failed'] += 1
            continue
        row['five_dimensions'] = saved['five_dimensions']; row['status'] = 'machine_annotated_unverified'
        canonical = {k: row['five_dimensions'][k] for k in FiveDimensions.model_fields}
        validate_annotation(row, canonical, row['sentence_spans'])
        row['hidden_gold'] = ground_truth(row)
        from scripts.export_review_bundle import verify
        verify(row, {'raw_record':row['raw_record']})
        if row['split'] == 'train':
            messages = [{'role':'system','content':TRAIN_SYSTEM},
                {'role':'user','content':json.dumps({'evidence':view(row)},ensure_ascii=False)},
                {'role':'assistant','content':json.dumps(conclusion(row),ensure_ascii=False)}]
            tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
            if tokens > 6144:
                states['train_context_reserved_for_later'] += 1
                continue
        if row['split'] == 'train':
            simulated = next(e for e in training_examples(row) if e['variant'] == 'source_bound_simulated_dialogue')
            row['five_dimensions']['dialogue_rounds']['simulated_training'] = simulated['simulated_rounds']
        row['record_binding'] = record_binding(row)
        review_path = ROOT / 'data/review/five_dimensions_v2_human' / (row['case_id'] + '.json')
        if review_path.exists():
            human = json.loads(review_path.read_text())
            if (human.get('record_binding') == row['record_binding'] and human.get('human_attestation') is True and
                human.get('reviewer_id') and human.get('reviewed_at') and human.get('decision') == 'reject'):
                states['human_rejected'] += 1
                continue
        selected.append(row); counts[row['split']] += 1
    validate_splits(selected)
    freeze_path = ROOT/'runtime/five_dimensions_v2_test_freeze.json'
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        if freeze['data_binding'] != fingerprint([r['record_binding'] for r in selected]):
            raise ValueError('Final tests have opened; this dataset cannot be replaced within the same cycle')
    complete = all(counts[s] == n for s, n in QUOTAS.items())
    write_rows(BASE / 'current.private.jsonl', selected)
    for split in QUOTAS:
        subset = [r for r in selected if r['split'] == split]
        write_rows(BASE / 'visible' / (split + '.jsonl'), [
            {k: r[k] for k in ('case_id', 'group_id', 'split')} | {'evidence': view(r)} for r in subset])
        write_rows(BASE / 'hidden' / (split + '.jsonl'), [
            {k: r[k] for k in ('case_id', 'record_binding', 'hidden_gold', 'five_dimensions', 'source')} for r in subset])
        for row in subset:
            path = ROOT / 'data/review/five_dimensions_v2_human' / (row['case_id'] + '.json')
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'case_id': row['case_id'], 'record_binding': row['record_binding'],
                    'reviewer_id': '', 'reviewed_at': '', 'decision': 'pending',
                    'checks': {k: False for k in HUMAN_CHECKS}, 'notes': ''}, ensure_ascii=False, indent=2) + '\n')
    report = json.loads((ROOT / 'data/five_dimensions_report.json').read_text())
    report.update(generated_at=datetime.now(timezone.utc).isoformat(),
        annotation_status_counts=dict(states), machine_annotated=len(selected),
        current_split_counts=dict(counts), dataset_target_assembled=complete,
        data_sha256=fingerprint([r['record_binding'] for r in selected]),
        dataset_binding=fingerprint([r['record_binding'] for r in selected]))
    if complete:
        validate_splits(selected, require_target=True)
        frozen = BASE / 'frozen' / report['data_sha256'][:16]
        path = frozen / 'dataset.private.jsonl'
        if not path.exists(): write_rows(path, selected)
        if path.read_bytes() != (BASE / 'current.private.jsonl').read_bytes():
            raise ValueError('Frozen dataset bytes differ; no existing snapshot was overwritten')
        report['frozen_dataset'] = str(path.relative_to(ROOT))
        report['frozen_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / 'data/five_dimensions_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('machine_annotated', 'current_split_counts', 'dataset_target_assembled')}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
