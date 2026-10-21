import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from app.local_resource import model_resource


def test_gpu_reservation_excludes_other_processes_and_allows_worker_threads(tmp_path, monkeypatch):
    monkeypatch.setenv('RUNTIME_DIR', str(tmp_path))
    code = ('from app.local_resource import model_resource\n'
            'try:\n'
            '    with model_resource("test-child"): print("acquired")\n'
            'except ValueError: print("busy")\n')
    def nested():
        with model_resource('worker-thread'): return True
    with model_resource('test-parent'):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(nested).result(timeout=3)
        child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=os.environ.copy(), timeout=5)
        assert child.returncode == 0 and child.stdout.strip() == 'busy'
    child = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=os.environ.copy(), timeout=5)
    assert child.returncode == 0 and child.stdout.strip() == 'acquired'
