#!/usr/bin/env python3
"""Download checksum-pinned per-trial English transcripts, not generated cases."""
import argparse
import json
from pathlib import Path
import httpx
from fetch_lecardv2 import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts',default='0,1,2,3,4,5')
    args=parser.parse_args();parts=[int(x) for x in args.parts.split(',')]
    if len(set(parts))!=len(parts) or any(x not in range(6) for x in parts):parser.error('Invalid parts')
    manifest=json.loads((ROOT/'data/oldbailey_download_manifest.json').read_text())
    out=ROOT/'data/raw/oldbailey';out.mkdir(parents=True,exist_ok=True)
    with httpx.Client(timeout=180,follow_redirects=True) as client:
        for number in parts:
            key=f'trials/train-{number:05}-of-00006.parquet'
            path=out/Path(key).name;expected=manifest['files'][key]
            if path.exists():
                if path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
                    raise ValueError('Existing checksum mismatch')
                print(path.name,'verified existing',flush=True);continue
            tmp=path.with_suffix('.download')
            url=f"https://hf-mirror.com/datasets/biglam/old_bailey_proceedings/resolve/{manifest['revision']}/{key}"
            with client.stream('GET',url) as response:
                response.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in response.iter_bytes(1<<20):f.write(chunk)
            if tmp.stat().st_size!=expected['bytes'] or digest(tmp)!=expected['sha256']:
                raise ValueError('Downloaded checksum mismatch')
            tmp.replace(path);print(path.name,'verified downloaded',flush=True)


if __name__=='__main__':main()
