#!/usr/bin/env python3
"""Revise explicit training source ranges or types, preserving the frozen pool."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.complexity_review import complexity_exclusion, source_review_hold
from app.five_dimensions import fingerprint, validate_splits
from app.source_policy import require_clean_evidence, court_boundary
from scripts.build_v3 import START, ITEM, LEAK, evidence_kind, sentences
from scripts.export_review_bundle import verify

PROTOCOL = 'assistant-training-source-supplement-v1'
SOURCE_KEYS = ('case_id', 'group_id', 'split', 'source', 'evidence', 'sentence_spans', 'source_fact_sha256')


def revise(row, plan):
    if row['split'] != 'train' or row['source_binding'] != plan['parent_source_binding']:
        raise ValueError('Only bound original training sources may be supplemented')
    if row['source_binding'] != fingerprint({k: row[k] for k in SOURCE_KEYS}):
        raise ValueError('Original source binding differs')
    if complexity_exclusion(row) or source_review_hold(row):
        raise ValueError('Use the human-feedback revision workflow for a held source')
    verify(row, {'raw_record': row['raw_record']})
    out = copy.deepcopy(row)
    fact = row['raw_record']['ajjbqk']
    marker = START.search(fact)
    if not marker:
        raise ValueError('No explicit original evidence section')
    starts = [marker.end()+m.start() for m in ITEM.finditer(fact[marker.end():])]
    changes = []
    trimmed = set()
    for action in plan.get('trim_judicial_tails', []):
        matches = [e for e in out['evidence'] if e['id'] == action['evidence_id']]
        if len(matches) != 1 or action['evidence_id'] in trimmed:
            raise ValueError('Judicial-tail trim must identify one distinct existing item')
        e = matches[0]
        boundary = court_boundary(e['quote'])
        end = action['end']
        if (not boundary or e['end'] != action['previous_end'] or
            end != e['start'] + boundary['offset'] or not e['start'] < end < e['end'] or
            fact[e['start']:e['end']] != e['quote'] or not action.get('reason', '').strip()):
            raise ValueError('Only the exact first recognized judicial boundary may shorten a source item')
        quote = fact[e['start']:end]
        if len(quote) < 40:
            raise ValueError('Judicial-tail trim must retain a substantive original source prefix')
        before = fingerprint(e)
        removed = fact[end:e['end']]
        previous_end = e['end']
        text = quote.replace('被告人','当事人').replace('犯罪嫌疑人','当事人')
        e.update(end=end, quote=quote, text=text,
                 transforms=[] if text == quote else [{'type':'neutral_role_label',
                     'from':['被告人','犯罪嫌疑人'],'to':'当事人'}])
        offset = row['source'].get('fact_offset_in_fulltext')
        if offset is not None:
            e.update(fulltext_start=offset + e['start'], fulltext_end=offset + end)
        trimmed.add(e['id'])
        changes.append({'operation':'trim_judicial_tail','evidence_id':e['id'],
                        'previous_end':previous_end,'end':end,'boundary_rule':boundary['rule'],
                        'removed_characters':len(removed), 'removed_tail_sha256':hashlib.sha256(removed.encode()).hexdigest(),
                        'before_binding':before,'after_binding':fingerprint(e),'reason':action['reason']})
    require_clean_evidence(out['evidence'])
    changed_types = set()
    for action in plan.get('reclassify_items', []):
        matches = [e for e in out['evidence'] if e['id'] == action['evidence_id']]
        if len(matches) != 1 or action['evidence_id'] in changed_types:
            raise ValueError('Type correction must identify one distinct existing item')
        e = matches[0]
        if (e['kind'] != action['previous_kind'] or action['kind'] == e['kind'] or
            evidence_kind(e['quote']) != action['kind'] or not action.get('reason', '').strip()):
            raise ValueError('Type correction must match the original heading and previous type')
        previous = e['kind']
        e['kind'] = action['kind']
        changed_types.add(e['id'])
        changes.append({'operation':'reclassify_existing','evidence_id':e['id'],
                        'previous_kind':previous,'kind':e['kind'],
                        'source_quote_sha256':hashlib.sha256(e['quote'].encode()).hexdigest(),
                        'reason':action['reason']})
    for action in plan['append_ranges']:
        start, end = action['start'], action['end']
        if start not in starts or not start < end <= len(fact):
            raise ValueError('Supplement must start at an explicit evidence item')
        next_start = next((s for s in starts if s > start), len(fact))
        if end > next_start:
            raise ValueError('A supplement cannot cross source item boundaries')
        quote = fact[start:end]
        if not 40 <= len(quote) <= 10000 or evidence_kind(quote) != action['kind']:
            raise ValueError('Supplement needs a recognized concrete source heading')
        if LEAK.search(quote) or any(charge in quote for charge in row['hidden_gold']['charges']):
            raise ValueError('Target charge or judicial classification cannot be supplemented as evidence')
        if any(max(start,e['start']) < min(end,e['end']) for e in out['evidence']):
            raise ValueError('Supplement repeats or overlaps existing evidence')
        text = quote.replace('被告人','当事人').replace('犯罪嫌疑人','当事人')
        e = {'id': f'E{len(out["evidence"])+1}', 'kind': action['kind'], 'quote': quote,
             'text': text, 'start': start, 'end': end, 'source_field': 'ajjbqk',
             'source_context_before': fact[max(0,start-180):start],
             'transforms': [] if text == quote else [{'type':'neutral_role_label',
                 'from':['被告人','犯罪嫌疑人'],'to':'当事人'}]}
        offset = row['source'].get('fact_offset_in_fulltext')
        if offset is not None:
            e.update(fulltext_start=offset+start, fulltext_end=offset+end)
        require_clean_evidence([e])
        out['evidence'].append(e)
        changes.append({'evidence_id':e['id'],'start':start,'end':end,
                        'source_quote_sha256':hashlib.sha256(quote.encode()).hexdigest(),
                        'reason':action['reason']})
    for action in plan.get('extend_ranges', []):
        matches = [e for e in out['evidence'] if e['id'] == action['evidence_id']]
        if len(matches) != 1:
            raise ValueError('Extension must identify one existing source item')
        e = matches[0]
        start, old_end, end = e['start'], e['end'], action['end']
        if start not in starts or not old_end < end <= len(fact):
            raise ValueError('Extension must preserve the original start and grow its end')
        if fact[start:old_end] != e['quote']:
            raise ValueError('Extension cannot rewrite the original source prefix')
        crossed = [s for s in starts if old_end <= s < end]
        if crossed != action['nested_item_starts']:
            raise ValueError('Every crossed nested number must be explicitly declared')
        for nested in crossed:
            following = next((s for s in starts if s > nested), len(fact))
            if evidence_kind(fact[nested:following]) is not None:
                raise ValueError('Extension cannot swallow another recognized source document')
        quote = fact[start:end]
        if (len(quote) > 10000 or evidence_kind(quote) != e['kind'] or LEAK.search(quote)
            or any(charge in quote for charge in row['hidden_gold']['charges'])):
            raise ValueError('Extended range needs the same clean source heading and no answer label')
        if any(other['id'] != e['id'] and max(start,other['start']) < min(end,other['end'])
               for other in out['evidence']):
            raise ValueError('Extension overlaps another existing source item')
        text = quote.replace('被告人','当事人').replace('犯罪嫌疑人','当事人')
        require_clean_evidence([{'text':text}])
        before = fingerprint(e)
        e.update(end=end, quote=quote, text=text,
                 transforms=[] if text == quote else [{'type':'neutral_role_label',
                     'from':['被告人','犯罪嫌疑人'],'to':'当事人'}])
        offset = row['source'].get('fact_offset_in_fulltext')
        if offset is not None:
            e.update(fulltext_start=offset+start, fulltext_end=offset+end)
        changes.append({'operation':'extend_existing','evidence_id':e['id'],
                        'start':start,'previous_end':old_end,'end':end,
                        'nested_item_starts':crossed,'before_binding':before,'after_binding':fingerprint(e),
                        'source_quote_sha256':hashlib.sha256(quote.encode()).hexdigest(),
                        'reason':action['reason']})
    out['sentence_spans'] = sentences(out['evidence'])
    out['source_binding'] = fingerprint({k:out[k] for k in SOURCE_KEYS})
    out.update(pipeline_version=PROTOCOL, status='training_source_revision_unverified',
               human_reviewed=False, formal_approval=False,
               training_source_revision={'protocol':PROTOCOL,'author':'assistant_source_reader',
                   'parent_source_binding':row['source_binding'],'plan_binding':fingerprint(plan),
                   'changes':changes,'qualified_cases_added':0})
    out.pop('five_dimensions', None)
    verify(out, {'raw_record':out['raw_record']})
    require_clean_evidence(out['evidence'])
    return out


def bound_bytes(root, entry):
    p = (root / entry['path']).resolve()
    if not p.is_relative_to(root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Training source provenance file changed')
    return p.read_bytes()


def load_revision(root, manifest_entry):
    manifest = json.loads(bound_bytes(root, manifest_entry))
    if manifest['protocol'] != PROTOCOL or manifest['qualified_cases_added'] != 0:
        raise ValueError('Source supplement is not a certified case')
    plans = json.loads(bound_bytes(root, manifest['plan']))
    parents = {r['case_id']:r for r in map(json.loads, bound_bytes(root,manifest['parent_pool']).decode().splitlines())}
    expected = [revise(parents[p['case_id']], p) for p in plans['cases']]
    rows = list(map(json.loads,bound_bytes(root,manifest['dataset']).decode().splitlines()))
    if fingerprint(rows) != fingerprint(expected):
        raise ValueError('Training source revision differs from its explicit plan')
    validate_splits(rows)
    return rows


def main(args):
    pool = ROOT / 'data/processed/five_dimensions_v2/candidates.private.jsonl'
    plans = json.loads(args.plan.read_text())
    parents = {r['case_id']:r for r in map(json.loads,pool.open()) if r['split']=='train'}
    rows = [revise(parents[p['case_id']],p) for p in plans['cases']]
    validate_splits(rows)
    out = ROOT / 'data/processed/training_source_revision' / fingerprint(plans)[:16]
    out.mkdir(parents=True, exist_ok=True)
    def file_entry(path):
        return {'path':str(path.resolve().relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    dataset = out / 'candidates.private.jsonl'
    content = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
    if dataset.exists() and dataset.read_bytes() != content:
        raise ValueError('Refusing to overwrite a different source revision')
    dataset.write_bytes(content)
    manifest = {'protocol':PROTOCOL,'parent_pool':file_entry(pool),'plan':file_entry(args.plan),
        'dataset':file_entry(dataset),'cases':len(rows),'train_only':True,'qualified_cases_added':0}
    path = out / 'manifest.private.json'
    manifest_bytes = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    if path.exists() and path.read_bytes() != manifest_bytes:
        raise ValueError('Refusing to overwrite a different source manifest')
    path.write_bytes(manifest_bytes)
    load_revision(ROOT, file_entry(path))
    print(json.dumps({'manifest':str(path.relative_to(ROOT)),'cases':len(rows),'qualified_cases_added':0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    main(parser.parse_args())
