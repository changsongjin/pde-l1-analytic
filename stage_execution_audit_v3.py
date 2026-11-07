"""Formal first-batch preservation/serialization checkpoint, not acceptance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import stage3_campaign as C
import stage3_data_v2 as DATA
from stage_execution_audit_v2 import read

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'stage_execution_audit_v3_result.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve previous audit and use a new version.')
    previous = read(ROOT/'stage_execution_audit_v2_result.json')
    checks, changed, scripts, results, errors = {}, [], {}, {}, []
    for label, mapping in [('previous_scripts', previous['new_scripts']),
                           ('previous_results', {n:r['sha256'] for n,r in previous['result_manifests'].items()})]:
        bad = [n for n,h in mapping.items() if not (ROOT/n).exists() or C.digest(ROOT/n) != h]
        checks[label+'_unchanged'] = not bad
        changed.extend(bad)
    protocol = DATA.load_frozen_protocol()
    C.protocol_check(protocol)
    checks['frozen_sources_current'] = True
    statistics = read(ROOT/'stage3_statistics_result.json')
    checks['statistics_synthetic_checks_and_source'] = (statistics['all_checks_passed'] is True
        and statistics['source_sha256'] == C.digest(ROOT/'stage3_statistics.py')
        and statistics['protocol_sha256'] == C.digest(DATA.PROTOCOL_PATH))
    index = read(C.INDEX)
    checks['campaign_stopped_and_saved'] = index['batches'][-1]['status'] != 'running'
    reviews = [p for p in ROOT.glob('stage3_checkpoint_review_v2_*_result.json') if 'review' in read(p)]
    if not reviews:
        raise ValueError('Run immutable-index numerical checkpoint review first.')
    review_ref = C.reference(sorted(reviews)[-1])
    wrapper = read(ROOT/review_ref['file'])
    review = read(C.check_reference(wrapper['review']))
    snapshot = C.check_reference(wrapper['immutable_index'])
    checks['review_matches_current_index'] = C.digest(snapshot) == C.digest(C.INDEX)
    checks['checkpoint_numerical_review_passed'] = review['all_checks_passed'] is True
    for path in sorted(ROOT.glob('stage*.py')):
        try:
            compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
            scripts[path.name] = C.digest(path)
        except Exception as exc:
            errors.append({'file':path.name, 'error':str(exc)})
    paths = list(ROOT.glob('stage*.json')) + list((ROOT/'stage3_campaign').rglob('*.json'))
    paths += list((ROOT/'stage3_data').rglob('*.json')) + list((ROOT/'stage3_review_snapshots').glob('*.json'))
    for path in sorted(set(paths)):
        if path == OUTPUT:
            continue
        try:
            data = read(path)
            results[C.relative(path)] = {'sha256':C.digest(path), 'size_bytes':path.stat().st_size,
                'reported_all_checks_passed':data.get('all_checks_passed')}
        except Exception as exc:
            errors.append({'file':C.relative(path), 'error':str(exc)})
    checks['syntax_and_finite_JSON'] = not errors
    checks['saved_within_user_two_hour_batch'] = datetime.now(timezone.utc) < datetime(2026,11,6,14,51,tzinfo=timezone.utc)
    methods = [m for row in index['datasets'].values() for m in row['methods'].values()]
    awaiting_ingest = []
    for value in methods:
        record = read(C.check_reference(value['result']))
        method_dir = C.check_reference(record['job']).parent
        requests = sorted((method_dir/'continuations').glob('*.json'))
        if requests and read(requests[-1])['path_source'] != record['path_source']:
            awaiting_ingest.append(C.reference(requests[-1]))
    counts = {'planned_datasets':606, 'planned_method_paths':3636,
        'started_datasets':len(index['datasets']), 'complete_datasets':sum(r.get('status')=='complete' for r in index['datasets'].values()),
        'recorded_method_paths':len(methods), 'certified_complete_method_paths':sum(m['numerically_complete'] for m in methods),
        'numerically_pending_method_paths':sum(not m['numerically_complete'] for m in methods),
        'numerical_requests_awaiting_ingest':len(awaiting_ingest),
        'certified_final_points_rechecked':review['certified_final_points']}
    report = {'stage':'third_batch_integrity_checkpoint', 'all_checks_passed':all(checks.values()),
        'checks':checks, 'changed_previous_files':changed, 'errors':errors,
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'batch_started_utc':'2026-11-06T12:52:22+00:00', 'conservative_batch_deadline_utc':'2026-11-06T14:51:00+00:00',
        'numerical_checkpoint_review':review_ref, 'counts':counts,
        'numerical_requests_awaiting_ingest':awaiting_ingest,
        'source_scripts':scripts, 'result_manifests':results,
        'formal_grid_complete':index['execution_complete'],
        'scope':'Historical/source preservation, syntax, finite JSON and saved certificate/evaluation review. Not completion of all formal seeds, method superiority or publication acceptance.',
        'publication_ready':False, 'new_checkpoint_uploaded_to_GitHub':False}
    C.atomic_json(OUTPUT, report)
    print(json.dumps({'all_checks_passed':report['all_checks_passed'], 'counts':counts, 'errors':errors}))
    if not report['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
