"""Retain immutable index snapshots around the unchanged numerical audit.

The initial review bound a hash of the resumable index; this wrapper archives
that exact state and makes later reviews refer directly to immutable copies.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import stage3_checkpoint_review as OLD
import stage3_campaign as C

ROOT = Path(__file__).resolve().parent


def archive_first_index():
    review_file = ROOT/'stage3_checkpoint_review_20261106T131219Z_result.json'
    review = C.read(review_file)
    current = C.read(C.INDEX)
    old = deepcopy(current)
    old['batches'] = old['batches'][:1]
    old['datasets'] = {}
    for detail in review['checked_datasets']:
        identity = detail['dataset']
        tag = C.dataset_tag(identity['case'], identity['sigma'], identity['noise_seed'])
        row = deepcopy(current['datasets'][tag])
        row['methods'] = dict(list(row['methods'].items())[:detail['checked_methods']])
        row['status'] = detail['status']
        if detail['status'] != 'complete':
            row['result'] = None
        old['datasets'][tag] = row
    old['execution_complete'] = old['all_checks_passed'] = False
    expected = review['index']['sha256']
    payload = json.dumps(old, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    variants = [payload.encode('utf-8'), payload.replace('\n', '\r\n').encode('utf-8')]
    restored = next((data for data in variants if hashlib.sha256(data).hexdigest() == expected), None)
    if restored is None:
        raise ValueError('Cannot reproduce the first review index exactly; never label an approximate reconstruction as the original.')
    path = ROOT/'stage3_review_snapshots'/f'index_{expected}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != restored:
            raise ValueError('Immutable index archive changed.')
    else:
        with path.open('xb') as stream:
            stream.write(restored)
    return {'review':C.reference(review_file), 'exact_archived_index':C.reference(path),
            'sha256_matches_original_review':C.digest(path) == expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    first = archive_first_index()
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report = {'stage':'3_checkpoint_review_with_immutable_index', 'source':C.reference(Path(__file__)),
        'first_review_index_archive':first, 'no_campaign_artifact_modified':True}
    if args.archive_only:
        report.update(all_checks_passed=first['sha256_matches_original_review'], scope='Exact-byte archival only; no numerical checks rerun.')
    else:
        index = C.INDEX
        current_bytes = index.read_bytes()
        if C.read(index)['batches'][-1]['status'] == 'running':
            raise RuntimeError('Stop campaign before numerical audit.')
        snapshot = ROOT/'stage3_review_snapshots'/f'index_{now}.json'
        with snapshot.open('xb') as stream:
            stream.write(current_bytes)
        before = set(ROOT.glob('stage3_checkpoint_review_*_result.json'))
        try:
            C.INDEX = snapshot
            OLD.main()
        finally:
            C.INDEX = index
        created = set(ROOT.glob('stage3_checkpoint_review_*_result.json')) - before
        if len(created) != 1 or index.read_bytes() != current_bytes:
            raise ValueError('Review output not unique or actual index changed during audit.')
        review_file = created.pop()
        verified = C.read(review_file)
        report.update(all_checks_passed=verified['all_checks_passed'], review=C.reference(review_file),
            immutable_index=C.reference(snapshot), completed_datasets=verified['completed_datasets'],
            complete_method_paths=verified['complete_method_paths'], certified_final_points=verified['certified_final_points'],
            scope='Unchanged v1 numerical and data checks with immutable input-index provenance.')
    path = ROOT/f'stage3_checkpoint_review_v2_{now}_result.json'
    C.atomic_json(path, report)
    print(json.dumps({'file':path.name, **report}))


if __name__ == '__main__':
    main()
