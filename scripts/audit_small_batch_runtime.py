#!/usr/bin/env python3
"""Check the installed tokenizer's actual training inputs without using the GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.local_context import chat_token_ids


def main(args):
    folder = (ROOT / args.run).resolve()
    if not folder.is_relative_to((ROOT / 'artifacts/small_batch_formal').resolve()):
        raise ValueError('Select a small-batch run')
    run = json.loads((folder / 'run.json').read_text())
    for name in ('data/train.jsonl', 'config.yaml'):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != run['training_file_sha256'][name]:
            raise ValueError('Frozen training inputs changed')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / run['model_path']), local_files_only=True,
                                              trust_remote_code=False)
    examples = list(map(json.loads, (folder / 'data/train.jsonl').open()))
    lengths = []; masks = []
    for example in examples:
        messages = example['messages']
        # These are the installed mlx_lm ChatDataset.process arguments.
        actual = tokenizer.apply_chat_template(messages, tools=example.get('tools'), return_dict=False)
        if actual != chat_token_ids(tokenizer, messages, generation=False):
            raise ValueError('Installed training tokenization differs from preflight')
        offset = len(tokenizer.apply_chat_template(messages[:-1], tools=example.get('tools'),
                     add_generation_prompt=messages[-1].get('role') == 'assistant', return_dict=False))
        if not 0 < offset < len(actual):
            raise ValueError('No assistant completion remains after masking the prompt')
        lengths.append(len(actual)); masks.append(len(actual) - offset)
    import yaml
    config = yaml.safe_load((folder / 'config.yaml').read_text())
    if (not lengths or len(lengths) != run['train_examples'] or max(lengths) != run['max_example_tokens'] or
        max(lengths) > config['max_seq_length'] or config['batch_size'] != 1 or
        config['grad_accumulation_steps'] != run['grad_accumulation_steps']):
        raise ValueError('Actual training lengths or sampling configuration differ')
    library = metadata.distribution('mlx-lm')
    files = ('mlx_lm/tuner/trainer.py', 'mlx_lm/tuner/datasets.py', 'mlx_lm/lora.py')
    hashes = {}
    for name in files:
        content = Path(library.locate_file(name)).read_bytes()
        target = folder / 'runtime-code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Installed training library changed after runtime capture')
        target.write_bytes(content); hashes[name] = hashlib.sha256(content).hexdigest()
    versions = {name: metadata.version(name) for name in
                ('mlx', 'mlx-lm', 'transformers', 'tokenizers', 'safetensors', 'numpy')}
    report = {'run_id': run['run_id'], 'generated_at': datetime.now(timezone.utc).isoformat(),
              'capture_stage': run['stage'], 'examples_checked': len(lengths),
              'actual_max_example_tokens': max(lengths), 'preflight_token_ids_match_installed_training_call': True,
              'truncated_examples': 0, 'minimum_supervised_completion_tokens': min(masks),
              'batch_size': 1, 'requested_micro_iterations': config['iters'],
              'complete_dataset_passes_covered_by_requested_iterations': config['iters'] // len(lengths),
              'gradient_accumulation_steps': config['grad_accumulation_steps'],
              'planned_optimizer_updates': config['iters'] // config['grad_accumulation_steps'],
              'runtime_versions': versions, 'installed_training_code_sha256': hashes,
              'independent_accuracy_verified': False}
    content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    (folder / 'runtime_audit.json').write_text(content)
    (ROOT / 'data/small_batch_runtime_audit.json').write_text(content)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    main(parser.parse_args())
