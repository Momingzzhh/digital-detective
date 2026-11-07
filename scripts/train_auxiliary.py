#!/usr/bin/env python3
"""Bounded Mac LoRA pilot for explicitly accepted auxiliary calibration only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.auxiliary_training import load_dataset, public_counts
from app.five_dimensions import fingerprint, training_examples
from app.local_context import chat_token_ids, TOKEN_POLICY
from app.local_model import model_path, verify_model_files
from app.local_resource import model_resource
from scripts.train_local import observed_metrics

GUARDED = ('.env', 'data/latest_training_report.json', 'data/five_dimensions_report.json')
CODE = ('scripts/train_auxiliary.py', 'scripts/train_local.py', 'scripts/export_review_bundle.py',
        'app/auxiliary_training.py', 'app/five_dimensions.py', 'app/local_context.py',
        'app/local_model.py', 'requirements-mac.lock')


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def expanded_examples(rows):
    result = []
    for row in rows:
        for example in training_examples(row):
            if example['variant'] == 'source_bound_simulated_dialogue':
                for index, message in enumerate(example['messages']):
                    if message['role'] == 'assistant':
                        result.append(example | {'messages': example['messages'][:index+1],
                                                 'supervised_dialogue_turn': (index+1)//2})
            else:
                result.append(example)
    return result


def prepare(args):
    manifest_path = (ROOT / args.manifest).resolve()
    rows, _ = load_dataset(ROOT, manifest_path)
    os.environ.update(LOCAL_MODEL_PATH=str(ROOT / 'models/qwen3-4b-instruct-2507-4bit'),
                      LOCAL_ADAPTER_PATH='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1')
    base = verify_model_files()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path()), local_files_only=True, trust_remote_code=False)
    examples = expanded_examples(rows)
    counts = [len(chat_token_ids(tokenizer, e['messages'], generation=False)) for e in examples]
    if any(n > args.max_seq_length for n in counts):
        raise ValueError('Auxiliary examples exceed context: revise data; never truncate or silently omit them')
    iters = args.iters if args.iters is not None else ((len(examples)+3)//4)*4
    if iters < len(examples) or iters % 4 or not 4 <= iters <= 32:
        raise ValueError('Auxiliary pilot must cover all examples with 4..32 micro iterations, multiple of four')
    out = ROOT / 'artifacts/auxiliary_training' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (out / 'data').mkdir(parents=True)
    content = ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in examples)
    (out / 'data/train.jsonl').write_text(content)
    # Training replay is an engineering loss probe, never independent validation.
    (out / 'data/valid.jsonl').write_text(json.dumps(examples[0], ensure_ascii=False) + '\n')
    config = {'model': str(model_path()), 'train': True, 'data': str(out / 'data'),
        'adapter_path': str(out / 'adapters'), 'fine_tune_type': 'lora', 'num_layers': 8,
        'batch_size': 1, 'iters': iters, 'learning_rate': 1e-5,
        'max_seq_length': args.max_seq_length, 'grad_checkpoint': True,
        'grad_accumulation_steps': 4, 'mask_prompt': True, 'seed': 20261105,
        'steps_per_report': 1, 'steps_per_eval': iters, 'save_every': iters,
        'val_batches': 1, 'lora_parameters': {'rank': 8, 'dropout': 0.05, 'scale': 16.0},
        'report_to': None, 'test': False}
    import yaml
    (out / 'config.yaml').write_text(yaml.safe_dump(config))
    for relative in CODE:
        target = out / 'code' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    report = public_counts(rows) | {'run_id': out.name, 'stage': 'prepared', 'pilot': True,
        'model_path': str(model_path().relative_to(ROOT)), 'base_manifest_binding': fingerprint(base),
        'source_manifest': {'path': str(manifest_path.relative_to(ROOT)), 'sha256': sha_file(manifest_path)},
        'code_file_sha256': {p: sha_file(ROOT / p) for p in CODE},
        'training_file_sha256': {p: sha_file(out / p) for p in ('data/train.jsonl', 'data/valid.jsonl', 'config.yaml')},
        'guarded_file_sha256': {p: sha_file(ROOT / p) for p in GUARDED},
        'token_length_policy': TOKEN_POLICY, 'train_examples': len(examples),
        'variant_counts': dict(Counter(e['variant'] for e in examples)),
        'token_counts': counts, 'max_example_tokens': max(counts), 'iters': iters,
        'grad_accumulation_steps': 4, 'validation_kind': 'training_replay_engineering_probe_not_validation',
        'test_examples_in_training_directory': 0, 'actual_weight_update_verified': False,
        'training_complete': False, 'validation_accuracy_verified': False, 'test_accuracy_verified': False,
        'adapter_activated': False}
    save(out / 'run.json', report)
    print(json.dumps({'prepared_run': str(out.relative_to(ROOT)), 'examples': len(examples),
                      'max_tokens': max(counts), 'iters': iters}), flush=True)
    return out


def run(out):
    report = json.loads((out / 'run.json').read_text())
    if report['stage'] != 'prepared':
        raise ValueError('Refusing to restart a used auxiliary run; prepare a new immutable run')
    from app.auxiliary_training import bound_file
    bound_file(ROOT, report['source_manifest'])
    load_dataset(ROOT, ROOT / report['source_manifest']['path'])
    for key, base in (('code_file_sha256', ROOT), ('guarded_file_sha256', ROOT), ('training_file_sha256', out)):
        if any(sha_file(base / p) != digest for p, digest in report[key].items()):
            raise ValueError('Prepared auxiliary inputs changed; prepare a new run')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    bootstrap = 'import mlx.core as mx; mx.set_memory_limit(17*1024**3); from mlx_lm.lora import main; main()'
    command = [sys.executable, '-u', '-c', bootstrap, '--config', str(out / 'config.yaml')]
    report['stage'] = 'running'
    save(out / 'run.json', report)
    with (out / 'training.log').open('w') as log:
        process = subprocess.Popen(command, env=os.environ.copy(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        returncode = process.wait()
    report.update(observed_metrics(out / 'training.log'))
    if returncode or report['completed_micro_iterations'] != report['iters']:
        report.update(stage='failed', subprocess_returncode=returncode)
        save(out / 'run.json', report)
        raise RuntimeError('Auxiliary training did not finish; inspect its private training log')
    from safetensors.numpy import load_file
    adapter = out / 'adapters/adapters.safetensors'
    tensors = load_file(str(adapter))
    nonzero = sum(bool((v != 0).any()) for k, v in tensors.items() if k.endswith('lora_b'))
    if not nonzero:
        raise ValueError('No learned LoRA B parameters; cannot claim a weight update')
    if any(sha_file(ROOT / p) != digest for p, digest in report['guarded_file_sha256'].items()):
        raise ValueError('Formal state or active configuration changed during auxiliary pilot')
    report.update(stage='completed', actual_weight_update_verified=True,
                  optimizer_updates=report['iters']//4, nonzero_lora_b_tensors=nonzero,
                  adapter_sha256=sha_file(adapter), adapter_path=str(adapter.parent.relative_to(ROOT)),
                  guarded_files_unchanged=True)
    save(out / 'run.json', report)
    save(ROOT / 'data/latest_auxiliary_training_report.json', report)
    print(json.dumps({'auxiliary_weights_updated': True, 'optimizer_updates': report['optimizer_updates'],
                      'formal_cases_added': 0, 'adapter_activated': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--run')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--iters', type=int)
    parser.add_argument('--max-seq-length', type=int, default=6144)
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.run):
        parser.error('Choose one of --manifest or --run')
    out = (ROOT / args.run).resolve() if args.run else prepare(args)
    if not args.prepare_only:
        with model_resource('auxiliary_training'):
            run(out)
