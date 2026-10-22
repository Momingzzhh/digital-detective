from argparse import Namespace
import json
from scripts import run_local_pipeline as pipeline
from app import research


def test_full_machine_bundle_cannot_trigger_training_without_real_human_reviews(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'STATUS',tmp_path/'runtime/status.json')
    (tmp_path/'data').mkdir()
    (tmp_path/'data/five_dimensions_report.json').write_text(json.dumps({'dataset_target_assembled':True}))
    rows=[{'split':'train' if i<420 else ('dev' if i<510 else 'test')} for i in range(600)]
    monkeypatch.setattr(research,'records',lambda:rows)
    monkeypatch.setattr(research,'approved',lambda row:False)
    calls=[];monkeypatch.setattr(pipeline,'step',lambda script,*args:calls.append(script))
    pipeline.run(Namespace(batch_size=2,chunk_cases=4,annotation_model='agent'))
    assert 'train_local.py' not in calls
    status=json.loads(pipeline.STATUS.read_text())
    assert status['phase']=='needs_human_source_review' and status['heldout_cases']==180
    assert status['heldout_human_reviewed']==0 and not status['training_complete']


def test_annotation_restarts_workers_and_stops_when_pool_is_exhausted(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'STATUS',tmp_path/'runtime/status.json')
    (tmp_path/'runtime').mkdir();(tmp_path/'data').mkdir()
    (tmp_path/'data/five_dimensions_report.json').write_text(json.dumps({'dataset_target_assembled':False}))
    calls=[]
    def worker(script,*args):
        calls.append((script,args))
        (tmp_path/'runtime/five_dimensions_annotation_result.json').write_text(json.dumps({
            'finished':True,'processed':4 if len(calls)==1 else 0}))
    monkeypatch.setattr(pipeline,'step',worker)
    pipeline.annotate_until_ready(2,4)
    assert len(calls)==2
    assert all(args[args.index('--limit')+1]==4 for _,args in calls)


def test_memory_failure_reduces_batch_without_faking_case_outputs(tmp_path,monkeypatch):
    import subprocess
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'STATUS',tmp_path/'runtime/status.json')
    (tmp_path/'runtime').mkdir();(tmp_path/'data').mkdir()
    (tmp_path/'data/five_dimensions_report.json').write_text('{}')
    sizes=[]
    def worker(script,*args):
        sizes.append(args[args.index('--batch-size')+1])
        if len(sizes)==1:
            (tmp_path/'runtime/five_dimensions_annotation_failure.json').write_text(json.dumps({
                'failed_at':'9999-01-01','error':'Metal Insufficient Memory'}))
            raise subprocess.CalledProcessError(1,['worker'])
        (tmp_path/'runtime/five_dimensions_annotation_result.json').write_text(json.dumps({'finished':True,'processed':0}))
    monkeypatch.setattr(pipeline,'step',worker)
    pipeline.annotate_until_ready(2,4)
    assert sizes==[2,1]


def test_waiting_for_download_can_stop_without_gpu_or_case_work(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'STATUS',tmp_path/'runtime/status.json')
    (tmp_path/'runtime').mkdir();(tmp_path/'data').mkdir()
    (tmp_path/'data/five_dimensions_report.json').write_text('{"machine_annotated":0}')
    pipeline.STATUS.write_text('{"machine_annotated":53}')
    def stop(_):(tmp_path/'runtime/stop_five_dimensions_annotation').touch()
    monkeypatch.setattr(pipeline.time,'sleep',stop)
    monkeypatch.setattr(pipeline,'step',lambda *a:(_ for _ in ()).throw(AssertionError('No case work while download is pending')))
    pipeline.run(Namespace(batch_size=1,chunk_cases=2,annotation_model='curator',wait_for_curator=True))
    out=json.loads(pipeline.STATUS.read_text())
    assert out['phase']=='stopped_while_waiting_for_download' and out['running'] is False
    assert out['machine_annotated']==0
