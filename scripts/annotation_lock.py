"""Prevent concurrent annotation processes from racing on checkpoint files."""
import fcntl
import atexit
import os
from pathlib import Path
from contextlib import ExitStack
from app.local_resource import model_resource

_held = None
_resources = ExitStack()
atexit.register(_resources.close)


def acquire_annotation_lock():
    global _held
    directory = Path(__file__).resolve().parents[1] / 'runtime'
    directory.mkdir(exist_ok=True)
    held = (directory / 'five_dimensions_annotation.lock').open('a+')
    try: fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        held.close()
        raise RuntimeError('Another annotation process is active; continue its checkpoint rather than running two writers')
    held.seek(0); held.truncate(); held.write(str(os.getpid())); held.flush()
    _held = held
    _resources.enter_context(model_resource('annotation'))
