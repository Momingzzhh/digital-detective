#!/usr/bin/env python3
"""Aggregate human, source-bound sessions; missing cases stay in denominator."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.five_dimensions import THRESHOLD, fingerprint, validate_splits
from app.research import records, approved, model_binding, acceptance_model_ready, CHECKS
from scripts.evaluate import wilson


def summarize(rows, scores, model, ready):
    validate_splits(rows, require_target=True)
    result = {}
    for split in ('dev', 'test'):
        subset = [r for r in rows if r['split'] == split]
        correct = 0; completed = 0; accepted = 0; rounds = []; components = Counter()
        for row in subset:
            source_ok = approved(row)
            accepted += bool(source_ok)
            score = scores.get((split, row['case_id']))
            if not score or not source_ok: continue
            if (score.get('record_binding') != row['record_binding'] or score.get('model_binding') != model or
                score.get('mode') != 'human_source_bound' or not score.get('reviewer_id') or
                not score.get('human_scored_at') or not score.get('session_sha256')): continue
            completed += 1; correct += score['all_correct'] is True
            for key in ('charge_correct', 'perpetrator_correct', 'chronology_correct', 'refused_or_failed'):
                components[key] += score.get(key) is True
            rounds.append(score['human_analysis_rounds'])
        n = len(subset); value = correct / n if n else 0
        result[split] = {'split': split, 'cases': n, 'human_reviewed': accepted,
            'human_scored': completed, 'missing_or_failed': n - completed, 'whole_case_correct': correct,
            'whole_case_accuracy': value, 'wilson_95_interval': wilson(correct, n),
            'charge_accuracy': components['charge_correct'] / n,
            'perpetrator_accuracy': components['perpetrator_correct'] / n,
            'chronology_accuracy': components['chronology_correct'] / n,
            'refused_or_failed_scored': components['refused_or_failed'],
            'human_rounds': rounds, 'threshold': THRESHOLD,
            'passed': bool(ready and completed == n and accepted == n and value >= THRESHOLD),
            'model_binding': model, 'dataset_binding': fingerprint([r['record_binding'] for r in rows]),
            'generated_at': datetime.now(timezone.utc).isoformat()}
    result['training_complete'] = result['dev']['passed'] and result['test']['passed']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    scores = {}
    for split in ('dev', 'test'):
        for path in (ROOT / 'data/review/five_dimensions_scores' / split).glob('*.json'):
            r = json.loads(path.read_text()); scores[(split, r['case_id'])] = r
    report = summarize(records(), scores, model_binding(), acceptance_model_ready())
    for split, name in (('dev', 'validation'), ('test', 'test')):
        (ROOT / 'data' / (name + '_acceptance_report.json')).write_text(json.dumps(report[split], ensure_ascii=False, indent=2) + '\n')
    state = json.loads((ROOT / 'data/five_dimensions_report.json').read_text())
    state.update(validation_passed=report['dev']['passed'], test_passed=report['test']['passed'],
                 training_complete=report['training_complete'],
                 human_reviewed=sum(approved(r) is True for r in records()))
    state['qualified_complex_cases'] = state['human_reviewed']
    (ROOT / 'data/five_dimensions_report.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
