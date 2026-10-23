#!/usr/bin/env python3
"""Actual QLoRA updates, train-group-only variants and token length checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.five_dimensions import training_examples, conclusion, TRAIN_SYSTEM, view, validate_splits, fingerprint, record_binding, FiveDimensions, validate_annotation, ground_truth, VERSION
from app.local_model import model_path, verify_model_files
from app.local_resource import model_resource
from app.local_context import chat_token_ids, TOKEN_POLICY


def observed_metrics(log_path):
    text = log_path.read_text()
    peaks = [float(v) for v in re.findall(r'Peak mem ([0-9.]+) GB', text)]
    losses = [{'micro_iteration': int(i), 'token_loss': float(v)}
              for i,v in re.findall(r'Iter (\d+): Val loss ([0-9.]+)', text)]
    iterations = [int(i) for i in re.findall(r'Iter (\d+): Train loss', text)]
    return {'observed_peak_memory_gb': max(peaks, default=None),
            'observed_validation_token_losses': losses,
            'completed_micro_iterations': max(iterations, default=0)}


def main(args):
    dataset = args.dataset
    if not dataset:
        state = json.loads((ROOT/'data/five_dimensions_report.json').read_text())
        dataset = state.get('frozen_dataset') if not args.pilot else 'data/processed/five_dimensions_v2/current.private.jsonl'
        if not dataset: raise ValueError('Full dataset is not frozen; formal training cannot start')
    source = (ROOT / dataset).resolve()
    data_bytes = source.read_bytes()
    rows = [json.loads(l) for l in data_bytes.decode().splitlines() if l.strip()]
    if not args.pilot: validate_splits(rows, require_target=True)
    if not rows or not any(r['split'] == 'train' for r in rows): raise ValueError('No training cases')
    if any(r.get('record_binding') != record_binding(r) for r in rows): raise ValueError('Dataset binding changed')
    from scripts.export_review_bundle import verify
    for row in rows:
        if row.get('annotation_version') != VERSION: raise ValueError('Training data uses a superseded annotation version')
        verify(row, {'raw_record': row['raw_record']})
        validate_annotation(row,{k:row['five_dimensions'][k] for k in FiveDimensions.model_fields},row['sentence_spans'])
        if fingerprint(row['hidden_gold']) != fingerprint(ground_truth(row)):
            raise ValueError('Answer key differs from source-bound participant and event annotation')
    if not args.pilot:
        from app.research import approved
        if not all(approved(r) for r in rows if r['split'] in ('dev','test')):
            raise ValueError('Formal training requires source-reviewed heldout data; human approvals cannot be synthesized')
    if (ROOT / 'runtime/five_dimensions_v2_test_freeze.json').exists():
        raise ValueError('Test set already opened. Freeze new independent test data before another training cycle.')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    base_manifest = verify_model_files()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path()), local_files_only=True, trust_remote_code=False)
    data_hash = hashlib.sha256(data_bytes).hexdigest()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'artifacts/training' / run_id
    (out / 'data').mkdir(parents=True)
    code_hashes = {}
    for relative in ('scripts/train_local.py','app/five_dimensions.py','app/local_context.py','requirements-mac.lock'):
        content = (ROOT/relative).read_bytes()
        code_hashes[relative] = hashlib.sha256(content).hexdigest()
        destination = out/'code'/relative
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(content)
    def write(name, examples):
        accepted = []; rejected = []
        for example in examples:
            token_count = len(chat_token_ids(tokenizer, example['messages'], generation=False))
            if token_count > args.max_seq_length:
                rejected.append({'case_id': example['case_id'], 'variant': example.get('variant'), 'tokens': token_count})
            else: accepted.append(example)
        if name == 'train' and not accepted: raise ValueError('All training examples exceed length limit')
        (out / 'data' / (name + '.jsonl')).write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in accepted))
        return accepted, rejected
    all_examples = [e for r in rows if r['split'] == 'train' for e in training_examples(r)]
    train = [e for e in all_examples if e['variant'] == 'source_complete']
    # Every training case is included in the full task; select boundary variants
    # by a fixed hash, with no validation/test prediction used for selection.
    boundary_count = max(1, len(train)//10)
    variants = sorted({e['variant'] for e in all_examples if e['variant'] != 'source_complete'})
    for variant in variants:
        subset = [e for e in all_examples if e['variant'] == variant]
        subset.sort(key=lambda e: fingerprint(e['case_id'] + ':' + variant))
        train.extend(subset[:boundary_count])
    # MLX masks everything before the final assistant response. Materialize each
    # simulated response prefix so questions and refusals receive actual loss.
    expanded = []
    for example in train:
        if example['variant'] != 'source_bound_simulated_dialogue':
            expanded.append(example); continue
        for index, message in enumerate(example['messages']):
            if message['role'] == 'assistant':
                expanded.append(example | {'messages': example['messages'][:index+1],
                                           'supervised_dialogue_turn': (index+1)//2})
    train = expanded
    accepted, excluded = write('train', train)
    # Test text and answers are never copied to the training directory.
    valid = [{'case_id': r['case_id'], 'messages': [{'role': 'system', 'content': TRAIN_SYSTEM},
        {'role': 'user', 'content': json.dumps({'evidence': view(r)}, ensure_ascii=False)},
        {'role': 'assistant', 'content': json.dumps(conclusion(r), ensure_ascii=False)}]}
        for r in rows if r['split'] == 'dev']
    validation_kind = 'heldout_dev_token_loss_not_accuracy'
    if valid: _, dev_excluded = write('valid', valid)
    elif args.pilot:
        # Engineering-only loss probe; explicitly never called heldout validation.
        write('valid', [accepted[0]]); dev_excluded = []
        validation_kind = 'training_replay_engineering_probe_not_validation'
    else: raise ValueError('Missing independent validation cases')
    trained_groups = {e['group_id'] for e in accepted if e['variant'] == 'source_complete'}
    required_groups = {r['group_id'] for r in rows if r['split'] == 'train'}
    if trained_groups != required_groups:
        raise ValueError('Some training groups have no fitting examples; increase context or revise data, never silently drop cases')
    minimum_iters = ((len(accepted)+3)//4)*4
    iters = args.iters if args.iters is not None else max(600, minimum_iters)
    if iters <= 0 or iters % 4: raise ValueError('Iterations must be a positive multiple of gradient accumulation=4')
    if not args.pilot and iters < minimum_iters:
        raise ValueError('Formal training must cover every training example at least once')
    # Pilots verify machinery only and can never set training_complete.
    config = {'model': str(model_path()), 'train': True, 'data': str(out / 'data'),
        'adapter_path': str(out / 'adapters'), 'fine_tune_type': 'lora', 'num_layers': 8,
        'batch_size': 1, 'iters': iters, 'learning_rate': 1e-5,
        'max_seq_length': args.max_seq_length, 'grad_checkpoint': True,
        'grad_accumulation_steps': 4, 'mask_prompt': True, 'seed': 20261021,
        'steps_per_report': 5, 'steps_per_eval': 50, 'save_every': 50,
        'val_batches': min(10,len(valid)) if valid else 1, 'lora_parameters': {'rank': 8, 'dropout': 0.05, 'scale': 16.0},
        'report_to': None, 'test': False}
    import yaml
    (out / 'config.yaml').write_text(yaml.safe_dump(config))
    report = {'run_id': run_id, 'dataset_sha256': data_hash, 'dataset_version':VERSION,
        'dataset_binding': fingerprint([r['record_binding'] for r in rows]),
        'base_manifest_binding': fingerprint(base_manifest),
        'code_file_sha256': code_hashes,
        'token_length_policy': TOKEN_POLICY,
        'model_path': str(model_path().relative_to(ROOT)),
        'pilot': args.pilot, 'train_groups': len(trained_groups), 'train_examples': len(accepted),
        'validation_kind': validation_kind, 'iters': iters, 'minimum_iters_for_full_example_coverage': minimum_iters,
        'boundary_examples_per_variant': boundary_count,
        'excluded_overlength_examples': excluded, 'excluded_overlength_dev': dev_excluded,
        'test_examples_in_training_directory': 0, 'actual_weight_update_verified': False,
        'training_complete': False, 'validation_accuracy_verified': False, 'test_accuracy_verified': False,
        'config_sha256': fingerprint(config)}
    (out / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'run': str(out), 'training_examples': len(accepted), 'pilot': args.pilot}), flush=True)
    bootstrap = 'import mlx.core as mx; mx.set_memory_limit(17*1024**3); from mlx_lm.lora import main; main()'
    command = [sys.executable, '-u', '-c', bootstrap, '--config', str(out / 'config.yaml')]
    with (out / 'training.log').open('w') as log:
        process = subprocess.Popen(command, env=os.environ.copy(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        returncode = process.wait()
    if returncode:
        report.update(failed=True, subprocess_returncode=returncode)
        (out / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        raise RuntimeError(f'Local training failed ({returncode}); see {out / "training.log"}')
    report.update(observed_metrics(out / 'training.log'))
    if report['completed_micro_iterations'] != iters: raise ValueError('Training log does not confirm all requested iterations')
    report['optimizer_updates'] = iters // 4
    adapter = out / 'adapters/adapters.safetensors'
    from safetensors.numpy import load_file
    tensors = load_file(str(adapter))
    nonzero = sum(bool((value != 0).any()) for key, value in tensors.items() if key.endswith('lora_b'))
    if not nonzero: raise ValueError('No nonzero learned LoRA B parameters; cannot claim actual training')
    report.update(actual_weight_update_verified=True, nonzero_lora_b_tensors=nonzero,
                  adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                  adapter_path=str((out / 'adapters').relative_to(ROOT)))
    (out / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'data/latest_training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    state_path = ROOT / 'data/five_dimensions_report.json'
    state = json.loads(state_path.read_text())
    if args.pilot: state['pilot_weight_update_verified'] = True
    else: state['weights_trained'] = True
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'weights_updated': True, 'training_complete': False,
                     'reason': 'Human-verified independent validation and final test are still required'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset')
    parser.add_argument('--iters', type=int)
    parser.add_argument('--max-seq-length', type=int, default=6144)
    parser.add_argument('--pilot', action='store_true')
    with model_resource('training'):
        main(parser.parse_args())
