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
import time

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
                   cwd=ROOT, check=True, env=os.environ.copy(),
                   timeout=1800 if script == 'annotate_five_dimensions_batch.py' else None)


def annotate_until_ready(batch_size, chunk_cases, annotation_model='curator'):
    """Release GPU/grammar allocations between bounded, checkpointed workers."""
    chunks = 0
    while True:
        report = read(ROOT/'data/five_dimensions_report.json')
        if report.get('dataset_target_assembled'): return
        if (ROOT/'runtime/stop_five_dimensions_annotation').exists(): return
        chunks += 1
        state('annotating_local_cases',batch_size=batch_size,annotation_chunk=chunks,
              machine_annotated=report.get('machine_annotated',0),reason='Local annotation, source checks and semantic screening; no external model API')
        result_path = ROOT/'runtime/five_dimensions_annotation_result.json'
        result_path.unlink(missing_ok=True)
        attempt_start = datetime.now(timezone.utc).isoformat()
        try:
            step('annotate_five_dimensions_batch.py', '--limit', chunk_cases, '--batch-size', batch_size,
                 '--max-tokens', 4000, '--retry-rejected', '--retry-errors','--repair-rejected','--model-profile',annotation_model)
        except subprocess.TimeoutExpired:
            if batch_size == 1: raise
            batch_size //= 2
            state('restarting_timed_out_worker',batch_size=batch_size)
            continue
        except subprocess.CalledProcessError:
            failure = read(ROOT/'runtime/five_dimensions_annotation_failure.json')
            memory_failure = (failure.get('failed_at','') >= attempt_start and
                              'insufficient memory' in failure.get('error','').lower())
            if not memory_failure or batch_size == 1: raise
            batch_size //= 2
            state('retrying_after_memory_limit',batch_size=batch_size)
            continue
        result = read(result_path)
        if not result.get('finished'): raise RuntimeError('Worker exited without a completion checkpoint')
        if not result['processed']: return  # Pool and bounded format retries exhausted.


def run(args):
    (ROOT / 'runtime').mkdir(exist_ok=True)
    lock = (ROOT / 'runtime/local_pipeline.lock').open('a+')
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('A local pipeline is already running')
    state('rechecking_saved_annotations', started_at=datetime.now(timezone.utc).isoformat(),
          running=True, error=None, training_complete=False,batch_size=args.batch_size,
          annotation_model=getattr(args,'annotation_model','curator'),annotation_chunk=0,
          machine_annotated=read(ROOT/'data/five_dimensions_report.json').get('machine_annotated',0))
    try:
        curator=ROOT/'models/qwen3-14b-mlx-4bit/local_manifest.json'
        if getattr(args,'annotation_model','curator')=='curator' and not curator.exists():
            if not getattr(args,'wait_for_curator',False):
                raise RuntimeError('Download the pinned local curator first: scripts/download_local_model.py --profile curator')
            deadline=time.monotonic()+4*60*60
            state('waiting_for_curator_download',reason='Only public weights are downloading; no case data or API inference is sent externally')
            while not curator.exists():
                if (ROOT/'runtime/stop_five_dimensions_annotation').exists():
                    state('stopped_while_waiting_for_download',running=False);return
                if time.monotonic()>=deadline:raise RuntimeError('Curator download not ready after four active hours; resume download before relaunch')
                time.sleep(5)
        step('recheck_five_dimensions.py')
        annotate_until_ready(args.batch_size,args.chunk_cases,getattr(args,'annotation_model','curator'))
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
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run', '--batch-size',str(args.batch_size),
               '--chunk-cases',str(args.chunk_cases),'--annotation-model',args.annotation_model]
    if args.wait_for_curator:command.append('--wait-for-curator')
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
    parser.add_argument('--batch-size',type=int,choices=[1,2,4,8],default=1)
    parser.add_argument('--chunk-cases',type=int,choices=[2,4,8,16],default=2)
    parser.add_argument('--annotation-model',choices=['agent','curator'],default='curator')
    parser.add_argument('--wait-for-curator',action='store_true',help='Wait for an already-running curator download, then curate; never downloads or calls a model API')
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
        value['worker_progress'] = read(ROOT/'runtime/five_dimensions_annotation_progress.json')
        value['worker_progress_stale'] = value['worker_progress'].get('updated_at','') < value.get('started_at','')
        hub=read(ROOT/'models/qwen3-14b-mlx-4bit/hub-files.json')
        if hub:
            directory=ROOT/'models/qwen3-14b-mlx-4bit'
            files=[f for f in hub['siblings'] if not f['rfilename'].startswith('.')]
            received=0
            for file in files:
                target=directory/file['rfilename'];partial=target.with_name(target.name+'.partial')
                present=target if target.exists() else partial
                received+=min(file['size'],present.stat().st_size if present.exists() else 0)
            value['curator_download']={'received_file_bytes':received,'expected_file_bytes':sum(f['size'] for f in files),
                'hash_verified_manifest_exists':(directory/'local_manifest.json').exists()}
        value['power_note'] = '本地GPU计算会在合盖休眠时暂停；持续整理时请保持开盖并接通电源。'
        print(json.dumps(value,ensure_ascii=False,indent=2))
