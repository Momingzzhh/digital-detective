#!/usr/bin/env python3
"""Create/revalidate a research-v2 manifest. Does not execute an experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.experiment_manifest import build_manifest, canonical, verify_resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--files', type=Path, required=True,
                        help='JSON mapping logical names to paths relative to this JSON file')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    file_map = json.loads(args.files.read_text())
    files = {name: args.files.resolve().parent / path for name, path in file_map.items()}
    current = build_manifest(config, files)
    if args.output.exists():
        verify_resume(json.loads(args.output.read_text()), config, files)
        action = 'verified_existing'
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(canonical(current).decode() + '\n')
        action = 'created'
    print(json.dumps({'action': action, 'run_id': current['run_id'],
                      'experiment_executed': False}))


if __name__ == '__main__':
    main()
