#!/usr/bin/env python3
"""Run software tests without real model imports or outbound socket connections."""
import importlib.abc
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DASHSCOPE_API_KEY='')


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm'}:
            raise ModuleNotFoundError('CPU test runner prohibits real model imports: ' + fullname)


def no_connect(*args, **kwargs):
    raise RuntimeError('CPU software tests must not open network connections')


if __name__ == '__main__':
    sys.meta_path.insert(0, NoModels())
    socket.socket.connect = no_connect
    socket.socket.connect_ex = no_connect
    socket.create_connection = no_connect
    import pytest
    raise SystemExit(pytest.main(['-q', *sys.argv[1:]]))
