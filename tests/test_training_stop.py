"""Synthetic checkpoints and child supervision; no GPU or real case data."""
import json
import subprocess
import numpy as np
from safetensors.numpy import save_file
from scripts import train_small_batch as worker


def checkpoint(folder, iteration):
    p = folder / 'adapters' / f'{iteration:07d}_adapters.safetensors'
    p.parent.mkdir(exist_ok=True)
    save_file({'synthetic.lora_b': np.ones((2, 2), dtype=np.float32)}, str(p))
    return p


def test_only_logged_complete_aligned_weights_are_retained(tmp_path):
    checkpoint(tmp_path, 8)
    checkpoint(tmp_path, 16)  # Not confirmed saved by the log.
    (tmp_path / 'training.log').write_text('Iter 8: Saved adapter weights to synthetic.\nIter 17: Train loss 1.0\n')
    report = {'grad_accumulation_steps': 4, 'iters': 24}
    saved = worker.retained_checkpoint(tmp_path, report)
    assert saved['checkpoint_micro_iteration'] == 8
    assert saved['checkpoint_optimizer_updates'] == 2
    assert saved['nonzero_lora_b_tensors'] == 1
    assert saved['exact_training_resume_available'] is False
    assert saved['optimizer_state_saved'] is False
    (tmp_path / 'adapters/0000008_adapters.safetensors').write_bytes(b'partial')
    assert worker.retained_checkpoint(tmp_path, report) is None


def test_stop_saves_partial_status_and_leaves_completed_report_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'verify_prepared', lambda *a: None)
    folder = tmp_path / 'run'; folder.mkdir()
    (folder / 'run.json').write_text(json.dumps({'stage': 'prepared', 'iters': 24,
        'grad_accumulation_steps': 4, 'small_batch_training_completed': False,
        'actual_weight_update_verified': False, 'training_complete': False}))
    (tmp_path / 'data').mkdir()
    latest = tmp_path / 'data/latest_small_batch_training_report.json'
    latest.write_text('previous complete report')

    class Child:
        ended = False
        def __init__(self, *a, **kw): self.stdout = self.lines()
        def lines(self):
            yield 'Iter 8: Train loss 1.0\n'
            checkpoint(folder, 8)
            yield 'Iter 8: Saved adapter weights to synthetic.\n'
            (folder / 'stop.requested').touch()
            yield 'Iter 9: Train loss 0.9\n'
        def poll(self): return -15 if self.ended else None
        def terminate(self): self.ended = True
        def wait(self, timeout=None): assert self.ended; return -15

    monkeypatch.setattr(worker.subprocess, 'Popen', Child)
    worker.run(folder)
    report = json.loads((folder / 'run.json').read_text())
    assert report['stage'] == 'stopped'
    assert report['completed_micro_iterations'] == 9
    assert report['retained_weight_checkpoint']['checkpoint_micro_iteration'] == 8
    assert not report['small_batch_training_completed']
    assert not report['actual_weight_update_verified']
    assert latest.read_text() == 'previous complete report'
    assert json.loads((tmp_path / 'data/latest_small_batch_interrupted_report.json').read_text()) == report


def test_stop_waits_for_exit_and_kills_unresponsive_child():
    calls = []
    class Child:
        def poll(self): return None
        def terminate(self): calls.append('terminate')
        def kill(self): calls.append('kill')
        def wait(self, timeout=None):
            calls.append(('wait', timeout))
            if timeout: raise subprocess.TimeoutExpired('synthetic', timeout)
            return -9
    worker.stop_child(Child())
    assert calls == ['terminate', ('wait', 15), 'kill', ('wait', None)]
