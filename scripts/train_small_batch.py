#!/usr/bin/env python3
"""Actual formal small-batch QLoRA under the revised, training-only policy."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.auxiliary_training import bound_file
from app.five_dimensions import fingerprint
from app.local_context import chat_token_ids, TOKEN_POLICY
from app.local_model import model_path, verify_model_files
from app.small_batch_training import load_dataset, public_counts
from scripts.train_auxiliary import expanded_examples
from scripts.train_local import observed_metrics

GUARDED = ('.env', 'data/five_dimensions_report.json', 'data/latest_training_report.json',
           'data/auxiliary_calibration_report.json', 'data/latest_auxiliary_training_report.json')
CODE = ('scripts/train_small_batch.py', 'scripts/train_local.py', 'scripts/train_auxiliary.py',
        'scripts/prepare_small_batch_training.py', 'scripts/export_review_bundle.py',
        'app/small_batch_training.py', 'app/auxiliary_training.py', 'app/five_dimensions.py',
        'app/annotation_quality.py', 'app/complexity_review.py', 'app/source_policy.py',
        'app/local_context.py', 'app/local_model.py', 'scripts/prepare_training_source_revision.py',
        'scripts/build_v3.py', 'app/field_semantic.py', 'requirements-mac.lock')


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def require_unopened_test(root=ROOT):
    if (root / 'runtime/five_dimensions_v2_test_freeze.json').exists():
        raise ValueError('Independent test already opened; do not train within the same test cycle')


def prepare(args):
    require_unopened_test()
    manifest_path = (ROOT / args.small_batch_manifest).resolve()
    rows, manifest = load_dataset(ROOT, manifest_path)
    os.environ.update(LOCAL_MODEL_PATH=str(ROOT / 'models/qwen3-4b-instruct-2507-4bit'),
                      LOCAL_ADAPTER_PATH='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1')
    base = verify_model_files()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path()), local_files_only=True, trust_remote_code=False)
    examples = expanded_examples(rows, include_singletons=True)
    counts = [len(chat_token_ids(tokenizer, e['messages'], generation=False)) for e in examples]
    if not examples or any(n > args.max_seq_length for n in counts):
        raise ValueError('Every small-batch example must fit; never truncate or drop source material')
    if not 1 <= args.epochs <= 5:
        raise ValueError('Use one to five passes for this initial small-batch stage')
    minimum = ((len(examples) * args.epochs + 3) // 4) * 4
    iters = args.iters if args.iters is not None else minimum
    if iters < minimum or iters % 4 or iters > 400:
        raise ValueError('Small-batch iterations must cover all requested passes, align to four, and be at most 400')
    out = ROOT / 'artifacts/small_batch_formal' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (out / 'data').mkdir(parents=True)
    (out / 'data/train.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in examples))
    # MLX requires a validation file. This is an explicitly labelled training
    # replay probe, with no validation/test material or accuracy claim.
    (out / 'data/valid.jsonl').write_text(json.dumps(examples[0], ensure_ascii=False) + '\n')
    config = {'model': str(model_path()), 'train': True, 'data': str(out / 'data'),
        'adapter_path': str(out / 'adapters'), 'fine_tune_type': 'lora', 'num_layers': 8,
        'batch_size': 1, 'iters': iters, 'learning_rate': 1e-5,
        'max_seq_length': args.max_seq_length, 'grad_checkpoint': True,
        'grad_accumulation_steps': 4, 'mask_prompt': True, 'seed': 20261107,
        'steps_per_report': 1, 'steps_per_eval': iters, 'save_every': min(8, iters),
        'val_batches': 1, 'lora_parameters': {'rank': 8, 'dropout': 0.05, 'scale': 16.0},
        'report_to': None, 'test': False}
    import yaml
    (out / 'config.yaml').write_text(yaml.safe_dump(config))
    for relative in CODE:
        target = out / 'code' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    bound_file(ROOT, manifest['policy_file'])
    (out / 'policy.private.json').write_bytes((ROOT / manifest['policy_file']['path']).read_bytes())
    report = public_counts(rows) | {'run_id': out.name, 'stage': 'prepared', 'pilot': False,
        'formal_training_started': False, 'small_batch_training_completed': False,
        'model_path': str(model_path().relative_to(ROOT)), 'base_manifest_binding': fingerprint(base),
        'source_manifest': {'path': str(manifest_path.relative_to(ROOT)), 'sha256': sha_file(manifest_path)},
        'dataset_binding': manifest['dataset_binding'],
        'code_file_sha256': {p: sha_file(ROOT / p) for p in CODE},
        'training_file_sha256': {p: sha_file(out / p) for p in
                                ('data/train.jsonl', 'data/valid.jsonl', 'config.yaml', 'policy.private.json')},
        'guarded_file_sha256': {p: sha_file(ROOT / p) for p in GUARDED},
        'token_length_policy': TOKEN_POLICY, 'train_examples': len(examples),
        'single_evidence_training_variants': True,
        'variant_counts': dict(Counter(e['variant'] for e in examples)),
        'max_example_tokens': max(counts), 'iters': iters, 'requested_passes': args.epochs,
        'minimum_iters_for_requested_passes': minimum, 'grad_accumulation_steps': 4,
        'checkpoint_every_micro_iterations': min(8, iters),
        'validation_kind': 'training_replay_probe_not_independent_validation',
        'test_examples_in_training_directory': 0, 'actual_weight_update_verified': False,
        'training_complete': False, 'validation_accuracy_verified': False, 'test_accuracy_verified': False,
        'adapter_activated': False}
    save(out / 'run.json', report)
    print(json.dumps({'prepared_run': str(out.relative_to(ROOT)), 'training_cases': len(rows),
                      'examples': len(examples), 'max_tokens': max(counts), 'iters': iters}), flush=True)
    return out


def verify_prepared(out, report):
    require_unopened_test()
    if report['stage'] != 'prepared':
        raise ValueError('Used small-batch runs are immutable; prepare a new run')
    bound_file(ROOT, report['source_manifest'])
    load_dataset(ROOT, ROOT / report['source_manifest']['path'])
    if fingerprint(verify_model_files()) != report['base_manifest_binding']:
        raise ValueError('Base model manifest changed after preparation')
    for key, base in (('code_file_sha256', ROOT), ('guarded_file_sha256', ROOT), ('training_file_sha256', out)):
        if any(sha_file(base / p) != digest for p, digest in report[key].items()):
            raise ValueError('Prepared small-batch inputs changed; prepare a new run')


def retained_checkpoint(out, report):
    """Only a numbered checkpoint confirmed fully saved in the log is usable.

    These are weight checkpoints, not optimizer/RNG state for exact resumption.
    A partially overwritten adapters.safetensors must never stand in for one.
    """
    matches = re.findall(r'^Iter (\d+): Saved adapter weights to ',
                         (out / 'training.log').read_text(), re.MULTILINE)
    from safetensors.numpy import load_file
    from safetensors import SafetensorError
    for iteration in sorted(map(int, matches), reverse=True):
        if iteration % report['grad_accumulation_steps'] or iteration > report['iters']:
            continue
        checkpoint = out / 'adapters' / f'{iteration:07d}_adapters.safetensors'
        if not checkpoint.exists():
            continue
        try:
            tensors = load_file(str(checkpoint))
        except (OSError, SafetensorError):
            continue
        nonzero = sum(bool((v != 0).any()) for k, v in tensors.items() if k.endswith('lora_b'))
        if not nonzero:
            continue
        return {'checkpoint_micro_iteration': iteration,
                'checkpoint_optimizer_updates': iteration // report['grad_accumulation_steps'],
                'checkpoint_path': str(checkpoint.relative_to(out)),
                'checkpoint_sha256': sha_file(checkpoint), 'nonzero_lora_b_tensors': nonzero,
                'optimizer_state_saved': False, 'exact_training_resume_available': False}
    return None


def stop_child(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()


def run(out):
    report = json.loads((out / 'run.json').read_text())
    verify_prepared(out, report)
    if (out / 'stop.requested').exists():
        raise ValueError('Run already has a stop request; do not start GPU training')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    bootstrap = 'import mlx.core as mx; mx.set_memory_limit(17*1024**3); from mlx_lm.lora import main; main()'
    command = [sys.executable, '-u', '-c', bootstrap, '--config', str(out / 'config.yaml')]
    report.update(stage='running', formal_training_started=True,
                  started_at=datetime.now(timezone.utc).isoformat())
    save(out / 'run.json', report)
    process = None
    stopped = False
    try:
        with (out / 'training.log').open('w') as log:
            process = subprocess.Popen(command, env=os.environ.copy(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
                if (out / 'stop.requested').exists():
                    stopped = True
                    stop_child(process)
                    break
            returncode = process.wait()
        report.update(observed_metrics(out / 'training.log'))
        if stopped:
            report.update(stage='stopped', stop_reason='explicit_stop_request',
                          subprocess_returncode=returncode,
                          retained_weight_checkpoint=retained_checkpoint(out, report),
                          finished_at=datetime.now(timezone.utc).isoformat())
            save(out / 'run.json', report)
            save(ROOT / 'data/latest_small_batch_interrupted_report.json', report)
            print(json.dumps({'stage': 'stopped', 'training_complete': False,
                              'checkpoint_retained': bool(report['retained_weight_checkpoint'])}), flush=True)
            return
        if returncode or report['completed_micro_iterations'] != report['iters']:
            raise RuntimeError(f'Small-batch training did not finish (exit {returncode})')
        from safetensors.numpy import load_file
        adapter = out / 'adapters/adapters.safetensors'
        tensors = load_file(str(adapter))
        nonzero = sum(bool((v != 0).any()) for k, v in tensors.items() if k.endswith('lora_b'))
        if not nonzero:
            raise ValueError('No learned LoRA B parameters; cannot claim a weight update')
        if any(sha_file(ROOT / p) != digest for p, digest in report['guarded_file_sha256'].items()):
            raise ValueError('Benchmark state, historical auxiliary state or active model changed during training')
        report.update(stage='completed', small_batch_training_completed=True,
                      actual_weight_update_verified=True, optimizer_updates=report['iters']//4,
                      nonzero_lora_b_tensors=nonzero, adapter_sha256=sha_file(adapter),
                      adapter_path=str(adapter.parent.relative_to(ROOT)),
                      historical_and_active_files_unchanged=True,
                      finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        stop_child(process)
        report.update(stage='failed', error=str(exc)[:400],
                      finished_at=datetime.now(timezone.utc).isoformat())
        if (out / 'training.log').exists():
            report.update(observed_metrics(out / 'training.log'))
            report['retained_weight_checkpoint'] = retained_checkpoint(out, report)
        save(out / 'run.json', report)
        raise
    save(out / 'run.json', report)
    save(ROOT / 'data/latest_small_batch_training_report.json', report)
    print(json.dumps({'small_batch_formal_weights_updated': True,
                      'optimizer_updates': report['optimizer_updates'],
                      'training_complete': False, 'independent_accuracy_verified': False}), flush=True)


def main(args):
    if bool(args.small_batch_manifest) == bool(args.small_batch_run):
        raise ValueError('Choose exactly one small-batch manifest or prepared run')
    if args.dataset or args.pilot:
        raise ValueError('Small-batch policy cannot be combined with legacy dataset or pilot flags')
    out = (ROOT / args.small_batch_run).resolve() if args.small_batch_run else prepare(args)
    if not args.prepare_only:
        run(out)
