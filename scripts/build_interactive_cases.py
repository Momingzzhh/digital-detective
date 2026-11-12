#!/usr/bin/env python3
"""Train-only, fail-closed research v2 curation. Never deserialize heldout text."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.experiment_manifest import digest, file_digest
from app.sealed_metadata import project
from app.environment_models import (PrivateCase, PrivateEvidence, SourceSpan, ReleaseRule,
    SupportedClaim, SentenceReference, Observation, InvestigationAction, ActionResult, StateSupport)
from app.models import InteractiveConclusion
from app.interactive_curation import PUBLIC_BRIEFING, PUBLIC_KINDS, reachability
from app.source_policy import require_clean_evidence
from scripts.export_review_bundle import verify

TRAIN = ROOT / 'data/processed/small_batch_formal/9f52b61a0e3c27f5/data.private.jsonl'
EXPECTED_TRAIN = '7b55bc255ba6188d81ea231492e0456adc4a1d9493f13b42ca596151d2006237'
EXPECTED_AUDIT = 'acf48c208baea4951fc87da969ceadc90e3c4c94649c07965ce0aaaa25389295'
EXPECTED_BASELINE = 'e7015bd974ea04e0d00566554918f6f69daf00b60b8fc9cb13e58e815e8178f9'
OUT = ROOT / 'data/processed/research_v2'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    temp = path.with_suffix('.tmp'); temp.write_text(text); temp.replace(path)


def registry():
    audit_path = ROOT / 'data/processed/case_group_audit.json'
    baseline_path = ROOT / 'artifacts/research_v2/baseline_manifest.initial.json'
    if file_digest(audit_path) != EXPECTED_AUDIT or file_digest(baseline_path) != EXPECTED_BASELINE:
        raise ValueError('Pinned group audit or historical baseline changed')
    audit = json.loads(audit_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    historical = baseline['historical_training_groups']
    historical_ids = {cid for v in historical.values() for cid in v['case_ids']}
    records = {}; group_splits = defaultdict(set); history_groups = set()
    for name, expected in audit['source_files_sha256'].items():
        path = ROOT / 'data/processed' / name
        if file_digest(path) != expected: raise ValueError('Frozen group audit stale: '+name)
        with path.open() as stream:
            for line in stream:
                meta = project(line)
                cid = meta['case_id']; linked = audit['cases'][cid]
                group = linked['linked_group_id']
                if cid in records: raise ValueError('Duplicate source case')
                records[cid] = meta | linked
                group_splits[group].add(meta['split'])
                if cid in historical_ids or meta['group_id'] in historical: history_groups.add(group)
    if set(records) != set(audit['cases']): raise ValueError('Audit coverage changed')
    for row in records.values():
        group = row['linked_group_id']
        row['historical_training_exposure'] = group in history_groups
        row['excluded_from_heldout'] = group in history_groups or row['cross_split_conflict']
        if len(group_splits[group]) > 1 and not row['cross_split_conflict']:
            raise ValueError('Unquarantined split conflict')
    report = {'version': 'metadata-group-registry-v1',
        'method': 'Reuse exact-byte-bound case-number/full/fact-hash transitive audit; project identity fields only',
        'audit_sha256': file_digest(audit_path), 'baseline_sha256': file_digest(baseline_path),
        'source_files_sha256': audit['source_files_sha256'], 'counts': audit['counts'],
        'historical_exposed_linked_groups': len(history_groups),
        'historical_case_ids_not_in_modern_registry': len(historical_ids-records.keys()),
        'metadata_split_counts': dict(Counter(r['split'] for r in records.values())),
        'heldout_documents_excluded': sum(r['split'] != 'train' and r['excluded_from_heldout'] for r in records.values()),
        'heldout_text_deserialized': False,
        'limitations': ['Historical registry conservatively includes candidates, not proof of actual optimizer exposure.',
            'Frozen linkage can miss paraphrases/unmentioned related cases or overmerge prior convictions.',
            'No heldout semantic re-audit performed; no independence certification.']}
    external = {group: value | {'excluded_from_heldout': True,
        'reason': 'historical_training_identity_outside_modern_audit'}
        for group, value in historical.items() if set(value['case_ids'])-records.keys()}
    report['external_historical_group_count'] = len(external)
    report['external_historical_source_counts'] = dict(Counter(
        cid.split('-')[0] for cid in historical_ids-records.keys()))
    report['limitations'].append('External historical identities remain excluded; no cross-domain semantic linkage inferred.')
    save(OUT/'group_registry.private.json', report | {'cases': records, 'external_historical_groups': external})
    save(ROOT/'docs/research_v2/group_registry_report.json', report)
    return records


def kind_for(text):
    if '辨认' in text: return '辨认记录'
    if '证言' in text: return '证言'
    if '陈述' in text or '供述' in text or '辩解' in text: return '陈述'
    if '鉴定' in text: return '鉴定记录'
    if '勘验' in text or '现场' in text: return '现场记录'
    if '视' in text or '录音' in text: return '视听记录'
    return '书面记录'


def build(row, group):
    if row['split'] != 'train' or group['split'] != 'train' or group['cross_split_conflict']:
        raise ValueError('Not an unconflicted training case')
    verify(row, {'raw_record': row['raw_record']})
    require_clean_evidence(row['evidence'])
    ev = {e['id']: e for e in row['evidence']}
    spans = {s['id']: s for s in row['sentence_spans']}
    if len(spans) != len(row['sentence_spans']): raise ValueError('Duplicate sentence')
    for s in spans.values():
        if s['evidence_id'] not in ev or ev[s['evidence_id']]['text'][s['start']:s['end']] != s['text']:
            raise ValueError('Sentence span changed')
    # Exact text retained; only the already-audited neutral role substitutions.
    evidence = [PrivateEvidence(id=e['id'], text=e['text'], kind=kind_for(e['kind']),
        source=SourceSpan(start=e['start'], end=e['end'], quote=e['quote'])) for e in row['evidence']]
    grouped = defaultdict(list)
    for e in evidence: grouped[e.kind].append(e.id)
    claims = []
    dims = row['five_dimensions']
    for category, items in [('event', dims['events']), ('actor', dims['actor_assessments'])]:
        for item in items:
            refs = item['sentence_ids']
            if not refs or set(refs)-spans.keys(): raise ValueError('Unbound annotation')
            ids = sorted({spans[s]['evidence_id'] for s in refs})
            claim = item['claim'] if category == 'event' else item['name']+'：'+item['role']
            claims.append(SupportedClaim(id=f'{category}-{len(claims)+1}', category=category,
                claim=claim, support_sets=[ids], support_citations=[[SentenceReference(
                    evidence_id=spans[s]['evidence_id'], start=spans[s]['start'], end=spans[s]['end'],
                    text=spans[s]['text']) for s in refs]]))
    claims.append(SupportedClaim(id='legal-unknown', category='legal_element', claim='犯罪构成及法律结论',
        support_sets=[], support_citations=[], quality='unknown', unknown_reason='尚无逐要件专家审查；裁判结果不得直接作为部分观察金标。'))
    return PrivateCase(case_id=row['case_id'], group_id=group['linked_group_id'], split='train',
        source_sha256=row['source']['sha256_fulltext'], source_fact_sha256=row['source_fact_sha256'],
        source_row_sha256=digest(row), briefing=PUBLIC_BRIEFING, initial_evidence_ids=[], evidence=evidence,
        release_rules=[ReleaseRule(kind=kind, evidence_ids=ids) for kind, ids in sorted(grouped.items())],
        claims=claims)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', type=Path, help='Private assistant source-bound revision; never a human receipt')
    args = parser.parse_args()
    if not args.revision and (OUT/'current.json').exists():
        current = json.loads((OUT/'current.json').read_text())
        current_manifest = json.loads((OUT/current['directory']/'manifest.private.json').read_text())
        if current_manifest.get('assistant_revision'):
            raise ValueError('Current bundle has a source revision; supply its --revision path, do not silently revert it')
    groups = registry()
    if file_digest(TRAIN) != EXPECTED_TRAIN: raise ValueError('Pinned seven-case dataset changed')
    rows = []
    for line in TRAIN.open():
        if project(line)['split'] != 'train': raise ValueError('Heldout row in pinned train file')
        rows.append(json.loads(line))
    if len(rows) != 7: raise ValueError('Expected exactly seven initial training cases')
    cases = []; held = []; source_rows = {}
    for row in rows:
        try: case = build(row, groups[row['case_id']])
        except (ValueError, KeyError) as exc:
            held.append({'case_id': row['case_id'], 'reason': str(exc)}); continue
        cases.append(case); source_rows[row['case_id']] = row
    revision_info = None
    if args.revision:
        revision = json.loads(args.revision.read_text())
        if revision.get('author_type') != 'assistant_source_reader' or revision.get('human_approval') is not False:
            raise ValueError('Revision must explicitly be nonhuman and unapproved')
        matches = [i for i,c in enumerate(cases) if c.case_id == revision['case_id']]
        if len(matches) != 1: raise ValueError('Revision is outside this fixed training cohort')
        i = matches[0]; original = cases[i]
        if digest(original.model_dump()) != revision['parent_case_binding']: raise ValueError('Revision parent changed')
        cases[i] = PrivateCase.model_validate(original.model_dump() | {'claims': revision['claims']})
        revision_info = {'sha256': file_digest(args.revision), 'parent_case_binding': revision['parent_case_binding'],
                         'author_type': revision['author_type'], 'human_approval': False}
    values = [c.model_dump() for c in cases]
    binding = digest({'source_file': EXPECTED_TRAIN, 'cases': values})
    version = OUT / binding[:16]
    save(version/'cases.private.json', values)
    save(version/'sources.private.json', source_rows)
    save(version/'holds.private.json', held)
    save(version/'solvability.private.json', {c.case_id: reachability(c) for c in cases})
    save(version/'evaluation_private/state_support_rules.json',
         {c.case_id: [claim.model_dump() for claim in c.claims] for c in cases})
    manifest = {'version': 'train-curation-manifest-v1', 'binding': binding, 'source_sha256': EXPECTED_TRAIN,
        'case_count': len(cases), 'group_count': len({c.group_id for c in cases}), 'held_count': len(held),
        'selection': 'Existing seven train cases fixed before model generation; no answer/accuracy selection',
        'public_action_kinds': list(PUBLIC_KINDS),
        'semantic_audit': 'pending', 'human_review': 'pending', 'formal_benchmark_eligible': False,
        'dev': 'deferred_user_train_only', 'test': 'sealed_no_text_read',
        'expansion': 'blocked_until_first_batch_audited',
        'assistant_revision': revision_info,
        'known_limitations': ['Events inherited from legacy annotations may be compound; atomization pending.',
            'Single conjunctive support set inherited per claim; alternatives/counterevidence need semantic review.',
            'Generic type-only requests are a pilot; concrete person/time availability is not invented.',
            'Reachability checks are structural, not semantic solvability or formal scoring.']}
    save(version/'manifest.private.json', manifest)
    if args.revision: save(version/'assistant_revision.private.json', revision)
    save(OUT/'current.json', {'directory': binding[:16], 'binding': binding})
    save(ROOT/'docs/research_v2/train_curation_report.json', manifest)
    schema_dir = ROOT/'schemas/research_v2'
    for model in (PrivateCase, Observation, InvestigationAction, ActionResult, StateSupport, InteractiveConclusion):
        save(schema_dir/(model.__name__+'.json'), model.model_json_schema())
    print(json.dumps({'binding': binding, 'cases': len(cases), 'holds': len(held)}))


if __name__ == '__main__': main()
