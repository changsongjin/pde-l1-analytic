"""Prepare and publish this exact research checkpoint, without rerunning science.

Git keeps sources and small JSON; a verified Release ZIP keeps every selected
raw array, matrix and report. Git's credential helper supplies authentication in
memory. Credentials are never saved or printed.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = 'changsongjin/pde-l1-analytic'
BRANCH = 'codex/research-checkpoint-20261106'
TAG = 'research-checkpoint-20261107-batch3'
ARTIFACT = ROOT/'_github_upload'/'pde-research-checkpoint-20261107.zip'
MANIFEST = ROOT/'GITHUB_CHECKPOINT_MANIFEST.json'
RESULT = ROOT/'github_checkpoint_result.json'
DIRECTORIES = ['stage0_data', 'stage2_data', 'stage2_matrices', 'stage2a_progress',
    'stage2b_data', 'stage2b_matrices', 'stage2b_progress', 'stage2c_matrices',
    'stage2c_progress', 'stage3_data', 'stage3_campaign', 'stage3_review_snapshots']
EXTRA = ['.gitattributes', '.gitignore', 'AGENTS.md', 'README.md', 'HANDOFF.md',
    'EXECUTION_STATUS.md', 'RESEARCH_PLAN_DRAFT.md', 'UPLOAD_CHECKPOINT.md',
    'requirements-research.txt', 'requirements-experiment.txt', 'github_checkpoint.py']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def prepare():
    if RESULT.exists() or ARTIFACT.exists() or MANIFEST.exists():
        raise FileExistsError('Keep the existing checkpoint; publish it or create a new version.')
    protocol = json.loads((ROOT/'stage3_protocol_result.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/n) == h for n,h in protocol['source_sha256'].items())
    paths = set(ROOT.glob('stage*.py')) | set(ROOT.glob('stage*.json'))
    historical = json.loads((ROOT/'stage0_gate_result.json').read_text(encoding='utf-8'))['historical_script_sha256']
    paths.update(ROOT/n for n in set(historical) | set(protocol['source_sha256']))
    paths.update(ROOT/n for n in EXTRA)
    for name in DIRECTORIES:
        paths.update(p for p in (ROOT/name).rglob('*') if p.suffix in ('.json', '.npz'))
    assert all(p.is_file() and p.resolve().is_relative_to(ROOT) for p in paths)
    files = {p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size, 'sha256':sha(p)} for p in sorted(paths)}
    git_files = [n for n,r in files.items() if not n.endswith('.npz') and r['bytes'] <= 2_000_000]
    manifest = {'scope':'Complete selected research checkpoint; excludes unrelated work and credentials',
        'repository':REPO, 'branch':BRANCH, 'release_tag':TAG,
        'file_count':len(files), 'total_bytes':sum(r['bytes'] for r in files.values()),
        'files':files, 'git_files':git_files + [MANIFEST.name, RESULT.name]}
    write(MANIFEST, manifest)
    ARTIFACT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARTIFACT, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in files:
            archive.write(ROOT/name, name, compress_type=zipfile.ZIP_STORED if name.endswith('.npz') else zipfile.ZIP_DEFLATED)
        archive.write(MANIFEST, MANIFEST.name)
    with zipfile.ZipFile(ARTIFACT) as archive:
        for name,record in files.items():
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest() == record['sha256'], name
            assert sha(ROOT/name) == record['sha256'], 'Changed during packaging: '+name
    result = {'all_checks_passed':True, 'scope':'Packaging and byte integrity only; not publication acceptance',
        'archive_file':ARTIFACT.relative_to(ROOT).as_posix(), 'archive_bytes':ARTIFACT.stat().st_size,
        'archive_sha256':sha(ARTIFACT), 'file_count':len(files), 'git_file_count':len(manifest['git_files']),
        'manifest_sha256':sha(MANIFEST), 'release_tag':TAG, 'repository':REPO,
        'frozen_protocol_sha256':sha(ROOT/'stage3_protocol_result.json')}
    write(RESULT, result)
    print(json.dumps(result), flush=True)


def publish():
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    assert sha(ARTIFACT) == result['archive_sha256']
    remote = subprocess.check_output(['git','remote','get-url','origin'], text=True).strip()
    assert remote == 'https://github.com/'+REPO+'.git'
    head = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    credentials = subprocess.run(['git','credential','fill'], input='protocol=https\nhost=github.com\n\n',
        text=True, capture_output=True, check=True)
    values = dict(line.split('=',1) for line in credentials.stdout.splitlines() if '=' in line)
    headers = {'Authorization':'Bearer '+values['password'], 'Accept':'application/vnd.github+json',
        'User-Agent':'PDE-research-checkpoint', 'X-GitHub-Api-Version':'2022-11-28'}
    base = 'https://api.github.com/repos/'+REPO

    def api(path, data=None, method=None):
        payload = json.dumps(data,ensure_ascii=False).encode('utf-8') if data is not None else None
        request = urllib.request.Request(base+path, data=payload,
            headers={**headers,'Content-Type':'application/json'}, method=method)
        with urllib.request.urlopen(request,timeout=60) as response:
            return json.load(response)

    assert api('/git/ref/heads/'+BRANCH)['object']['sha'] == head, 'Push branch before publishing assets.'
    body = ('Research checkpoint: 25/606 datasets and 150 complete method paths; research is unfinished.\n\n'
        'Code and small reports: `'+BRANCH+'` at `'+head+'`.\n'
        'The ZIP contains byte-exact raw fields, cached matrices, full paths, failure evidence and reports.\n'
        'Restore instructions: `UPLOAD_CHECKPOINT.md`. No complete 20-seed noise comparison or publication claim.\n\n'
        'Archive SHA256: `'+result['archive_sha256']+'`\n'
        'Frozen protocol SHA256: `'+result['frozen_protocol_sha256']+'`')
    try:
        release = api('/releases/tags/'+TAG)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        release = next((r for r in api('/releases?per_page=100') if r['tag_name']==TAG),None)
        if release is None:
            release = api('/releases', {'tag_name':TAG,'target_commitish':head,
                'name':'PDE research checkpoint — 2026-11-07 batch 3','body':body,'draft':True,'prerelease':True})
    assert release['target_commitish'] in (head, BRANCH), 'Existing release points elsewhere.'
    assets = api('/releases/'+str(release['id'])+'/assets')
    asset = next((a for a in assets if a['name']==ARTIFACT.name),None)
    if asset is None:
        url = release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(ARTIFACT.name)
        assert urllib.parse.urlsplit(url).hostname == 'uploads.github.com'
        print(json.dumps({'uploading_bytes':ARTIFACT.stat().st_size}),flush=True)
        with ARTIFACT.open('rb') as stream:
            request = urllib.request.Request(url, data=stream, headers={**headers,
                'Content-Type':'application/zip','Content-Length':str(ARTIFACT.stat().st_size)}, method='POST')
            with urllib.request.urlopen(request,timeout=1200) as response:
                asset = json.load(response)
    assert asset['size'] == result['archive_bytes']
    assert asset.get('digest') == 'sha256:'+result['archive_sha256'], 'Remote asset digest differs or is unavailable.'
    release = api('/releases/'+str(release['id']), {'draft':False,'prerelease':True,'body':body}, method='PATCH')
    receipt = {'pushed_commit':head, 'release_url':release['html_url'],
        'asset_url':asset['browser_download_url'], 'asset_bytes':asset['size'], 'asset_digest':asset['digest'],
        'remote_branch_verified':True, 'remote_archive_sha256_verified':True}
    write(ARTIFACT.parent/'upload_receipt_result.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    {'prepare':prepare, 'publish':publish}[sys.argv[1]]()
