"""Inspect local adapter readiness without importing MLX or loading the GPU."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_adapter(path, expected_sha256='', expected_config_sha256=''):
    if not path:
        return {'ready': True, 'mode': 'base', 'training_verified': False}
    directory = Path(path)
    try:
        weights = directory / 'adapters.safetensors'
        config = directory / 'adapter_config.json'
        if not weights.is_file() or not config.is_file():
            raise ValueError('微调适配器文件不完整，请重新激活已完成的训练运行。')
        digest, config_digest = file_sha256(weights), file_sha256(config)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError('微调权重与已激活版本不一致，请重新核验并激活。')
        if expected_config_sha256 and config_digest != expected_config_sha256:
            raise ValueError('微调配置与已激活版本不一致，请重新核验并激活。')
        if not isinstance(json.loads(config.read_text()), dict):
            raise ValueError('微调配置格式无效。')
        result = {'ready': True, 'mode': 'adapter', 'adapter_sha256': digest,
                  'config_sha256': config_digest, 'training_verified': False}
        run_path = directory.parent / 'run.json'
        if run_path.is_file():
            run = json.loads(run_path.read_text())
            if (run.get('stage') == 'completed' and run.get('actual_weight_update_verified')
                    and run.get('adapter_sha256') == digest):
                result.update(training_verified=True, run_id=run['run_id'],
                              training_cases=run.get('training_cases'), train_examples=run.get('train_examples'),
                              optimizer_updates=run.get('optimizer_updates'))
        return result
    except (OSError, ValueError, KeyError, TypeError):
        # Do not reveal local paths through the authenticated status endpoint.
        return {'ready': False, 'mode': 'adapter', 'training_verified': False,
                'error': '微调权重或配置缺失、变更或无效，请使用激活脚本重新核验。'}


def require_adapter(path, expected_sha256='', expected_config_sha256=''):
    result = inspect_adapter(path, expected_sha256, expected_config_sha256)
    if not result['ready']:
        raise ValueError(result['error'])
    return result
