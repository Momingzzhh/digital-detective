#!/usr/bin/env python3
"""Resumable local curation, genuine training and explicit human acceptance gates."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATUS = ROOT / 'runtime/local_pipeline_status.json'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def state(phase, **values):
    value = read(STATUS) | {'phase': phase, 'pid': os.getpid(),
        'updated_at': datetime.now(timezone.utc).isoformat()} | values
    STATUS.parent.mkdir(exist_ok=True)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def step(script, *args):
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str,args)],
                   cwd=ROOT, check=True, env=os.environ.copy())


def run(args):
    (ROOT / 'runtime').mkdir(exist_ok=True)
    lock = (ROOT / 'runtime/local_pipeline.lock').open('a+')
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('A local pipeline is already running')
    state('rechecking_saved_annotations', started_at=datetime.now(timezone.utc).isoformat(),
          running=True, error=None, training_complete=False)
    try:
        step('recheck_five_dimensions.py')
        state('annotating_local_cases', batch_size=args.batch_size)
        batch_size = args.batch_size
        while True:
            attempt_start = datetime.now(timezone.utc).isoformat()
            try:
                step('annotate_five_dimensions_batch.py', '--limit', 1980, '--batch-size', batch_size,
                     '--max-tokens', 2400, '--retry-rejected', '--retry-errors')
                break
            except subprocess.CalledProcessError:
                recent = [read(p) for p in (ROOT/'data/review/five_dimensions_v1').glob('*.json')]
                memory_failure = any(r.get('generated_at','') >= attempt_start and
                    'insufficient memory' in r.get('error','').lower() for r in recent)
                if not memory_failure or batch_size == 1: raise
                batch_size //= 2
                state('retrying_after_memory_limit', batch_size=batch_size)
        step('assemble_five_dimensions.py')
        report = read(ROOT/'data/five_dimensions_report.json')
        if (ROOT/'runtime/stop_five_dimensions_annotation').exists():
            state('stopped_at_batch_boundary', running=False); return
        if not report.get('dataset_target_assembled'):
            state('dataset_shortfall', running=False, machine_annotated=report.get('machine_annotated',0),
                  reason='Insufficient source-qualified annotations; do not relax complexity to fill quotas')
            return
        from app.research import records, approved
        rows = records(); heldout = [r for r in rows if r['split'] in ('dev','test')]
        reviewed = sum(bool(approved(row)) for row in heldout)
        if reviewed != len(heldout):
            state('needs_human_source_review', running=False, heldout_cases=len(heldout),
                  heldout_human_reviewed=reviewed,
                  reason='Research personnel must attest to source and annotation quality; no automatic signatures')
            return
        state('training_local_weights')
        step('train_local.py')
        trained = read(ROOT/'data/latest_training_report.json')
        state('checking_trained_adapter_offline')
        step('smoke_offline.py', '--adapter', trained['adapter_path'],
             '--output', 'data/trained_offline_smoke_report.json')
        # Install the actual trained checkpoint for subsequent human validation.
        env_path = ROOT/'.env'
        lines = env_path.read_text().splitlines() if env_path.exists() else []
        lines = [line for line in lines if not line.startswith(('LOCAL_ADAPTER_PATH=','MODEL_BACKEND='))]
        lines += ['MODEL_BACKEND=local', 'LOCAL_ADAPTER_PATH='+trained['adapter_path']]
        temporary = env_path.with_suffix('.tmp')
        temporary.write_text('\n'.join(lines)+'\n'); temporary.chmod(0o600); temporary.replace(env_path)
        state('ready_for_human_validation', running=False,
              reason='Human interactive validation, then frozen final test, are still required')
    except Exception as exc:
        state('failed', running=False, error=str(exc)[:500])
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def launch(args):
    (ROOT/'runtime').mkdir(exist_ok=True)
    probe = (ROOT/'runtime/local_pipeline.lock').open('a+')
    try: fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Already running; use --status')
    fcntl.flock(probe.fileno(), fcntl.LOCK_UN); probe.close()
    (ROOT/'runtime/stop_five_dimensions_annotation').unlink(missing_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run', '--batch-size',str(args.batch_size)]
    if sys.platform == 'darwin' and Path('/usr/bin/caffeinate').exists():
        command = ['/usr/bin/caffeinate', '-i', *command]
    with (ROOT/'runtime/local_pipeline.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'launched_pid':process.pid, 'status':'runtime/local_pipeline_status.json',
                      'log':'runtime/local_pipeline.log'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--launch',action='store_true')
    mode.add_argument('--status',action='store_true')
    mode.add_argument('--stop',action='store_true',help='Stop at the next annotation batch boundary')
    parser.add_argument('--batch-size',type=int,choices=[1,2,4,8],default=4)
    args = parser.parse_args()
    if args.run: run(args)
    elif args.launch: launch(args)
    elif args.stop:
        (ROOT/'runtime').mkdir(exist_ok=True)
        (ROOT/'runtime/stop_five_dimensions_annotation').touch()
        print('Stop requested at the next annotation batch boundary; training already in progress is not interrupted.')
    else:
        value = read(STATUS)
        if value.get('running'):
            try: os.kill(value['pid'],0)
            except ProcessLookupError: value.update(running=False,phase='process_exited_unexpectedly')
            except PermissionError: value['process_check']='restricted'
        value['dataset'] = read(ROOT/'data/five_dimensions_report.json')
        print(json.dumps(value,ensure_ascii=False,indent=2))
