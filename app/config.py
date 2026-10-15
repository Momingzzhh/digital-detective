import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def settings():
    values = {}
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1)
                values[k.strip()] = v.strip().strip('\"\'')
    values.update(os.environ)
    return values


def runtime():
    path = Path(settings().get('RUNTIME_DIR', str(ROOT / 'runtime')))
    path.mkdir(parents=True, exist_ok=True)
    return path
