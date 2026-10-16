#!/usr/bin/env python3
"""Fetch only selected checksum-pinned shards, not the entire 125 GB corpus."""
import argparse
import json
from pathlib import Path
import httpx
from fetch_lecardv2 import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards',default='000,250,500,750')
    parser.add_argument('--max-download-gb',type=float,default=2)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'data/mnbvc_download_manifest.json').read_text())
    numbers=[int(x) for x in args.shards.split(',')]
    if len(numbers)!=len(set(numbers)) or any(n<0 or n>993 for n in numbers):parser.error('Invalid shards')
    files=[(f'data/{n:03}.jsonl.gz',f'mnbvc-{n:03}.jsonl.gz') for n in numbers]
    size=sum(manifest['files'][key]['bytes'] for key,_ in files)
    if args.max_download_gb<=0 or size>args.max_download_gb*1_000_000_000:parser.error('Selected shards exceed download cap')
    directory=ROOT/'data/raw';directory.mkdir(parents=True,exist_ok=True)
    with httpx.Client(follow_redirects=True,timeout=180) as client:
        for key,name in files:
            path=directory/name;expected=manifest['files'][key]['sha256']
            if path.exists():
                if digest(path)!=expected:raise ValueError('Existing shard checksum mismatch: '+name)
                print(name,'verified existing',flush=True);continue
            tmp=path.with_suffix('.download')
            url=f"https://hf-mirror.com/datasets/wormtooth/MNBVC-judgment/resolve/{manifest['revision']}/{key}"
            with client.stream('GET',url) as response:
                response.raise_for_status()
                with tmp.open('wb') as out:
                    for chunk in response.iter_bytes(1<<20):out.write(chunk)
            if digest(tmp)!=expected:raise ValueError('Downloaded checksum mismatch: '+name)
            tmp.replace(path);print(name,'verified downloaded',flush=True)


if __name__=='__main__':main()
