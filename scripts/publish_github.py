#!/usr/bin/env python3
"""Publish one exact local commit through GitHub's Git Data API when Git TLS fails.

Credentials stay inside the official gh CLI/keychain. Ref updates never force.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Momingzzhh/digital-detective'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def allowed_public_file(name, content):
    if (name.startswith(('.env','data/raw/','data/processed/','data/review/','models/','artifacts/','runtime/'))
        and name != '.env.example'):
        raise ValueError('Private path cannot be published: '+name)
    for pattern in (rb'sk-[A-Za-z0-9]{24,}', rb'gh[pousr]_[A-Za-z0-9]{25,}', rb'-----BEGIN [A-Z ]*PRIVATE KEY-----'):
        if re.search(pattern,content): raise ValueError('Possible credential in '+name)
    content.decode('utf-8')  # This source-only publisher does not upload weights/binaries.


def identity(line):
    match = re.fullmatch(r'(.+) <([^>]+)> (\d+) ([+-]\d{4})',line)
    if not match: raise ValueError('Unsupported commit identity')
    name,email,seconds,offset = match.groups()
    minutes = (int(offset[1:3])*60+int(offset[3:])) * (-1 if offset[0]=='-' else 1)
    date = datetime.fromtimestamp(int(seconds), timezone(timedelta(minutes=minutes))).isoformat(timespec='seconds')
    return {'name':name,'email':email,'date':date}


def api(gh, path, method='GET', body=None):
    command = [gh,'api','--hostname','github.com','repos/'+REPO+'/'+path,'--method',method]
    (ROOT/'runtime').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=ROOT/'runtime',suffix='.publish.json') as source:
        if body is not None:
            json.dump(body,source,ensure_ascii=False); source.flush()
            command += ['--input',source.name]
        result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
    if result.returncode: raise RuntimeError(f'GitHub {method} {path} failed; branch was not force-updated')
    return json.loads(result.stdout)


def main(args):
    local = git('rev-parse','HEAD').decode().strip()
    ref = api(args.gh,'git/ref/heads/main')['object']['sha']
    if ref == local:
        print(json.dumps({'already_published':True,'commit':local})); return
    raw = git('cat-file','commit',local).decode()
    headers,message = raw.split('\n\n',1)
    metadata = {}
    for line in headers.splitlines():
        key,value = line.split(' ',1)
        if key not in ('tree','parent','author','committer') or key in metadata:
            raise ValueError('Only one-parent unsigned commits are supported; use ordinary git for this commit')
        metadata[key] = value
    if metadata.get('parent') != ref:
        raise ValueError('Remote main is not the direct parent; reconcile local/remote changes before publishing')
    names = [n.decode() for n in git('diff-tree','--no-commit-id','--name-only','-r','-z',ref,local).split(b'\0') if n]
    tree = []
    for name in names:
        entry = git('ls-tree',local,'--',name).decode().strip()
        if not entry:
            tree.append({'path':name,'mode':'100644','type':'blob','sha':None}); continue
        mode,kind,blob = entry.split('\t',1)[0].split()
        if kind != 'blob' or mode not in ('100644','100755'): raise ValueError('Unsupported tree entry')
        content = git('cat-file','blob',blob)
        allowed_public_file(name,content)
        tree.append({'path':name,'mode':mode,'type':'blob','content':content.decode()})
    if not args.publish:
        print(json.dumps({'dry_run':True,'commit':local,'files':len(tree),'remote_parent_verified':True})); return
    parent = api(args.gh,'git/commits/'+ref)
    uploaded_tree = api(args.gh,'git/trees','POST',{'base_tree':parent['tree']['sha'],'tree':tree})
    if uploaded_tree['sha'] != metadata['tree']: raise ValueError('Remote tree differs from local tree; ref unchanged')
    uploaded = api(args.gh,'git/commits','POST',{'tree':metadata['tree'],'parents':[ref], 'message':message,
        'author':identity(metadata['author']),'committer':identity(metadata['committer'])})
    if uploaded['sha'] != local: raise ValueError('Commit identity differs; ref unchanged. Use UTC Git author/committer dates.')
    api(args.gh,'git/refs/heads/main','PATCH',{'sha':local,'force':False})
    verified = api(args.gh,'git/ref/heads/main')['object']['sha']
    if verified != local: raise ValueError('Published ref changed concurrently; inspect remote main')
    git('update-ref','refs/remotes/origin/main',local)
    print(json.dumps({'published_commit':local,'files':len(tree),'tree_verified':True,
                      'repository':'https://github.com/'+REPO}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gh',default='gh')
    parser.add_argument('--publish',action='store_true')
    main(parser.parse_args())
