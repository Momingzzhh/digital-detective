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
PROFILES = {
    'agent':(REPO,REVISION,'qwen3-4b-instruct-2507-4bit','Qwen/Qwen3-4B-Instruct-2507','local_model_manifest.json'),
    'curator':('Qwen/Qwen3-14B-MLX-4bit','ba63a5141812f9870287df53341123a71ba41433',
               'qwen3-14b-mlx-4bit','Qwen/Qwen3-14B','local_curator_manifest.json'),
}


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
    parser.add_argument('--profile',choices=tuple(PROFILES),default='agent')
    args = parser.parse_args()
    repo,revision,folder,base,manifest_file = PROFILES[args.profile]
    directory = ROOT / 'models' / folder
    directory.mkdir(parents=True, exist_ok=True)
    info_path = directory / 'hub-files.json'
    if not info_path.exists():
        download(f'{args.endpoint}/api/models/{repo}/revision/{revision}?blobs=true', info_path)
    info = json.loads(info_path.read_text())
    if info['sha'] != revision or info['id'] != repo: raise ValueError('Upstream revision or repository changed')
    files = [f for f in info['siblings'] if not f['rfilename'].startswith('.')]
    for meta in files:
        name = meta['rfilename']
        if Path(name).name != name or (name != 'LICENSE' and Path(name).suffix not in ('.json', '.safetensors', '.txt', '.md', '.jinja')):
            raise ValueError('Unexpected model file')
        target = directory / name
        if not verified(target, meta):
            temporary = target.with_name(target.name + '.partial')
            print(json.dumps({'downloading': name, 'bytes': meta['size']}), flush=True)
            download(f'{args.endpoint}/{repo}/resolve/{revision}/{quote(name)}', temporary)
            if not verified(temporary, meta): raise ValueError('Integrity failure: ' + name)
            temporary.replace(target)
        print(json.dumps({'verified': name, 'bytes': meta['size']}), flush=True)
    manifest = {'repository': repo, 'revision': revision, 'download_endpoint': args.endpoint,
        'base_model': base, 'license': 'Apache-2.0',
        'files': {f['rfilename']: {'bytes': f['size'], 'sha256': sha256(directory / f['rfilename'])} for f in files}}
    ready=directory/'local_manifest.json'
    temporary=ready.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(ready)
    (ROOT / 'data' / manifest_file).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'ready': str(directory)}), flush=True)


if __name__ == '__main__': main()
