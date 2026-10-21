"""One Metal task at a time across the local web, annotation and training jobs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
import threading
from .config import runtime

_mutex = threading.RLock()
_held = None
_references = 0


@contextmanager
def model_resource(operation):
    global _held, _references
    with _mutex:
        # A reservation belongs to the process, including asyncio worker threads.
        # Actual MLX generation is separately serialized by local_model._lock.
        if _held is None:
            handle = (runtime() / 'local_model.lock').open('a+')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise ValueError('本地模型正用于标注、训练或其他分析，请待当前任务结束后重试。')
            handle.seek(0); handle.truncate()
            json.dump({'pid': os.getpid(), 'operation': operation,
                       'started_at': datetime.now(timezone.utc).isoformat()}, handle)
            handle.flush()
            _held = handle
        _references += 1
    try:
        yield
    finally:
        with _mutex:
            _references -= 1
            if _references == 0:
                fcntl.flock(_held.fileno(), fcntl.LOCK_UN)
                _held.close(); _held = None
