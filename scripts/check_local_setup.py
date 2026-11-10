#!/usr/bin/env python3
"""Check the local app's prerequisites without downloading files or loading MLX."""
import argparse
import importlib.util
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(verify=False):
    from app.local_model import available, adapter_status, verify_model_files
    errors = []
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        errors.append('当前本地推理要求 Apple Silicon Mac。')
    missing = [name for name in ('fastapi', 'uvicorn', 'mlx', 'mlx_lm', 'outlines', 'docx', 'pypdf')
               if importlib.util.find_spec(name) is None]
    if missing:
        errors.append('缺少依赖：' + ', '.join(missing) + '；请按 README 完成一次联网准备。')
    if not available():
        errors.append('本地基座文件不完整；请先运行 scripts/download_local_model.py。')
    elif verify:
        try: verify_model_files()
        except (OSError, ValueError) as exc: errors.append(str(exc))
    adapter = adapter_status()
    if not adapter['ready']:
        errors.append(adapter['error'])
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    name = f"{adapter.get('training_cases')} 案微调 / {adapter.get('run_id')}" if adapter.get('training_verified') else adapter['mode']
    print(f'本机环境检查通过；当前模型：{name}。')
    print('打开 http://127.0.0.1:8765；首次使用请注册本机账户。按 Ctrl+C 停止服务。')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Also hash every base-model file')
    raise SystemExit(main(parser.parse_args().verify))
