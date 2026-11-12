"""Versioned experiment identities and fail-closed cache access (no model imports).

Callers must bind every input file and pass the effective configuration. This is
the research-v2 foundation; legacy evaluate.py is not migrated by this module.
"""
import hashlib
import json
import math
from pathlib import Path

VERSION = 'research-experiment-v1'
REQUIRED = frozenset({'data', 'environment', 'actions', 'model', 'adapter', 'prompts',
                      'decoding', 'scorer', 'code', 'seeds', 'runtime'})


def canonical(value):
    """Strict JSON: reject lossy keys, NaN, tuples, and unsupported objects."""
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(k) is str for k in item):
            for child in item.values():
                check(child)
            return
        raise ValueError('Fingerprint inputs must be finite, string-keyed JSON values')
    check(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_manifest(config, files):
    """files maps stable logical names to readable files; paths are not identity.

    Relocation preserves identity; byte changes do not. Bind dataset manifests
    AND their referenced files, model shards, tokenizer, adapter, prompts, code,
    lockfiles and scorer. Hashing a manifest alone does not verify its children.
    """
    canonical(config)
    if type(config) is not dict or not REQUIRED <= config.keys():
        raise ValueError('Missing required experiment configuration sections')
    if any(type(config[k]) is not dict or not config[k] for k in REQUIRED):
        raise ValueError('Each required section must be an explicit nonempty object')
    if not config['seeds'] or any(type(v) is not int or v < 0 for v in config['seeds'].values()):
        raise ValueError('Seeds must be explicit nonnegative integers')
    if not files or any(type(k) is not str or not k for k in files):
        raise ValueError('At least one named input file is required')
    bindings = {name: {'sha256': file_digest(path), 'bytes': Path(path).stat().st_size}
                for name, path in sorted(files.items())}
    payload = {'version': VERSION, 'config': json.loads(canonical(config)), 'files': bindings}
    return {**payload, 'run_id': VERSION + '-' + digest(payload)}


def validate_manifest(manifest):
    if type(manifest) is not dict or set(manifest) != {'version', 'config', 'files', 'run_id'}:
        raise ValueError('Malformed experiment manifest')
    if manifest['version'] != VERSION:
        raise ValueError('Unsupported experiment manifest version')
    payload = {k: manifest[k] for k in ('version', 'config', 'files')}
    if manifest['run_id'] != VERSION + '-' + digest(payload):
        raise ValueError('Experiment manifest digest mismatch')


def verify_resume(saved, config, files):
    """Rehash current inputs, not merely the saved JSON, before any cache access."""
    validate_manifest(saved)
    current = build_manifest(config, files)
    if current != saved:
        raise ValueError('Experiment inputs changed; create a new run, do not reuse cache')
    return current


def _record_path(directory, run_id, item_id):
    # Hash arbitrary case/episode IDs so neither can escape the cache directory.
    return Path(directory) / run_id / (digest(item_id) + '.json')


def cache_write(directory, saved, config, files, item_id, result):
    verify_resume(saved, config, files)
    if not isinstance(item_id, str) or not item_id:
        raise ValueError('A nonempty case/episode ID is required')
    payload = {'run_id': saved['run_id'], 'item_id': item_id, 'result': result}
    record = {**payload, 'payload_sha256': digest(payload)}
    path = _record_path(directory, saved['run_id'], item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never overwrites a completed result. An interrupted
    # partial record is detected on read and requires explicit recovery.
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(record).decode('utf-8') + '\n')
    return path


def cache_read(directory, saved, config, files, item_id):
    verify_resume(saved, config, files)
    path = _record_path(directory, saved['run_id'], item_id)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    payload = {k: record[k] for k in ('run_id', 'item_id', 'result')}
    if (record.get('payload_sha256') != digest(payload)
            or payload['run_id'] != saved['run_id'] or payload['item_id'] != item_id):
        raise ValueError('Cache provenance or payload mismatch')
    return record['result']
