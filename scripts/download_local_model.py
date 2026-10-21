#!/usr/bin/env python3
"""Download pinned public weights, verify Git blobs and LFS SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPO = 'mlx-community/Qwen3-4B-Instruct-2507-4bit'
REVISION = '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def verified(path, meta):
    if not path.exists() or path.stat().st_size != meta['size']: return False
    h = hashlib.sha256() if 'lfs' in meta else hashlib.sha1()
    if 'lfs' not in meta: h.update(f'blob {meta["size"]}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest() == (meta['lfs']['sha256'] if 'lfs' in meta else meta['blobId'])


def download(url, path):
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--retry-delay', '2',
        '--connect-timeout', '20', '--max-time', '7200', '--continue-at', '-',
        '--silent', '--show-error', url, '--output', str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', choices=['https://huggingface.co', 'https://hf-mirror.com'], default='https://hf-mirror.com')
    args = parser.parse_args()
    directory = ROOT / 'models/qwen3-4b-instruct-2507-4bit'
    directory.mkdir(parents=True, exist_ok=True)
    info_path = directory / 'hub-files.json'
    if not info_path.exists():
        download(f'{args.endpoint}/api/models/{REPO}/revision/{REVISION}?blobs=true', info_path)
    info = json.loads(info_path.read_text())
    if info['sha'] != REVISION: raise ValueError('Upstream revision changed')
    files = [f for f in info['siblings'] if not f['rfilename'].startswith('.')]
    for meta in files:
        name = meta['rfilename']
        if Path(name).name != name or Path(name).suffix not in ('.json', '.safetensors', '.txt', '.md', '.jinja'):
            raise ValueError('Unexpected model file')
        target = directory / name
        if not verified(target, meta):
            temporary = target.with_name(target.name + '.partial')
            print(json.dumps({'downloading': name, 'bytes': meta['size']}), flush=True)
            download(f'{args.endpoint}/{REPO}/resolve/{REVISION}/{quote(name)}', temporary)
            if not verified(temporary, meta): raise ValueError('Integrity failure: ' + name)
            temporary.replace(target)
        print(json.dumps({'verified': name, 'bytes': meta['size']}), flush=True)
    manifest = {'repository': REPO, 'revision': REVISION, 'download_endpoint': args.endpoint,
        'base_model': 'Qwen/Qwen3-4B-Instruct-2507', 'license': 'Apache-2.0',
        'files': {f['rfilename']: {'bytes': f['size'], 'sha256': sha256(directory / f['rfilename'])} for f in files}}
    (directory / 'local_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'data/local_model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'ready': str(directory)}), flush=True)


if __name__ == '__main__': main()
