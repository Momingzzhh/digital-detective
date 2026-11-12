import json
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('entry', ['run.sh', 'run_offline.sh'])
def test_entrypoints_never_install_and_force_offline(entry, tmp_path):
    for script in ('run.sh', 'run_offline.sh'):
        shutil.copyfile(ROOT / script, tmp_path / script)
    missing = subprocess.run(['sh', entry], cwd=tmp_path, capture_output=True, text=True)
    assert missing.returncode == 1
    assert '不会自动安装' in missing.stderr
    assert not (tmp_path / '.venv').exists()
    executable = tmp_path / '.venv/bin/python'
    executable.parent.mkdir(parents=True)
    executable.write_text('''#!/bin/sh
test "$MODEL_BACKEND" = local || exit 21
test "$HF_HUB_OFFLINE" = 1 || exit 22
test "$TRANSFORMERS_OFFLINE" = 1 || exit 23
printf '%s\\n' "$*" >> calls.txt
''')
    executable.chmod(0o755)
    completed = subprocess.run(['sh', entry], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0
    assert (tmp_path / 'calls.txt').read_text().splitlines() == [
        'scripts/check_local_setup.py',
        '-m uvicorn app.main:app --host 127.0.0.1 --port 8765']
    assert not (tmp_path / '.env').exists()
