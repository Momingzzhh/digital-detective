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
    pipeline.run(Namespace(batch_size=4))
    assert 'train_local.py' not in calls
    status=json.loads(pipeline.STATUS.read_text())
    assert status['phase']=='needs_human_source_review' and status['heldout_cases']==180
    assert status['heldout_human_reviewed']==0 and not status['training_complete']
