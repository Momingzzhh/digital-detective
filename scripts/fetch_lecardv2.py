#!/usr/bin/env python3
"""Fetch pinned full-text LeCaRDv2 triplet files with SHA-256 verification."""
import hashlib
import argparse
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
REV='84ebcf30a342b044e4c99c3510e34bb8830356da'
BASE=f'https://hf-mirror.com/datasets/sentence-transformers/lecard-v2/resolve/{REV}/triplet-all/'
FILES={
 'train-00000-of-00002.parquet':('lecardv2-triplet-all-0.parquet','c9cb05e8c53ab58144409695e150c67c79d4778f7dd003b00cc723fc9db7b59f'),
 'train-00001-of-00002.parquet':('lecardv2-triplet-all-1.parquet','69ec6e3dcf4093f9fce0869b9c194897ae2fc7d1c576b635a1832e4e2e669941'),
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        while chunk:=source.read(1<<20):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-mteb',action='store_true')
    args=parser.parse_args()
    downloads=[(BASE+remote,name,expected) for remote,(name,expected) in FILES.items()]
    if args.include_mteb:
        downloads.append(('https://hf-mirror.com/datasets/mteb/LeCaRDv2/resolve/566865fc50a566db1635307496e46c5a1a118cb2/corpus.jsonl',
                          'lecardv2-mteb-corpus.jsonl','f112eb82ff3b7cbc1e97e44d96ae9e520746df3858b77a0028115f1436454c0c'))
    directory=ROOT/'data/raw';directory.mkdir(parents=True,exist_ok=True)
    with httpx.Client(timeout=120,follow_redirects=True) as client:
        for url,name,expected in downloads:
            path=directory/name
            if path.exists():
                if digest(path)!=expected:raise ValueError('Existing archive checksum mismatch: '+name)
                print(name,'verified existing');continue
            tmp=directory/(name+'.download')
            with client.stream('GET',url) as response:
                response.raise_for_status()
                with tmp.open('wb') as output:
                    for chunk in response.iter_bytes(1<<20):output.write(chunk)
            if digest(tmp)!=expected:
                raise ValueError('Downloaded archive checksum mismatch: '+name)
            tmp.replace(path)
            print(name,'verified downloaded')


if __name__=='__main__':main()
