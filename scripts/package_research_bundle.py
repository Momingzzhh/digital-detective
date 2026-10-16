#!/usr/bin/env python3
"""Package existing private exports without calling any model or network service."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'dev', 'test')
STRATA = {'modern_chinese': 'v3_7', 'england_historical_1834_1913': 'oldbailey_v1'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def check_visible(row, historical):
    allowed = {'case_id', 'group_id', 'split', 'evidence'}
    if historical:
        allowed |= {'stratum', 'jurisdiction_context', 'participant_references'}
    if set(row) != allowed:
        raise ValueError('Unexpected visible fields')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', row['case_id']):
        raise ValueError('Unsafe case ID')
    if row['split'] not in SPLITS or not row['evidence']:
        raise ValueError('Invalid split or empty evidence')
    if any(set(e) != {'id', 'kind', 'text'} for e in row['evidence']):
        raise ValueError('Unexpected evidence fields')
    if len({e['id'] for e in row['evidence']}) != len(row['evidence']):
        raise ValueError('Duplicate evidence ID')
    if historical:
        if set(row['jurisdiction_context']) != {'country', 'source_period', 'trial_date', 'interpretation'}:
            raise ValueError('Unexpected jurisdiction fields')
        for person in row['participant_references']:
            if set(person) != {'name_en', 'source_person_id', 'source_reference_terms', 'meaning'}:
                raise ValueError('Unexpected participant fields')


def visible_text(row, historical):
    check_visible(row, historical)
    if historical:
        lines = ['英国历史庭审研究材料', '中文内容为原始英文记录的译文。',
                 '法域：England；庭审日期：' + str(row['jurisdiction_context']['trial_date']),
                 '中性称谓对应（不代表有罪）：']
        lines.extend('/'.join(p['source_reference_terms']) + ' = ' + p['name_en']
                     for p in row['participant_references'])
    else:
        lines = ['现代中文案件研究材料', '以下为原始材料中的线索，内容仍需复核。']
    lines.append('可用材料：')
    lines.extend(e['id'] + '（' + e['kind'] + '）\n' + e['text'] for e in row['evidence'])
    return '\n\n'.join(lines) + '\n'


def load_export(directory, historical):
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = {'machine_curated.unverified.jsonl'} | {
        f'{kind}/{split}.jsonl' for kind in ('visible', 'hidden') for split in SPLITS}
    if set(manifest['artifacts']) != expected:
        raise ValueError('Unexpected export artifact inventory')
    for name in sorted(expected):
        path = directory / name
        meta = manifest['artifacts'][name]
        if path.stat().st_size != meta['bytes'] or digest(path) != meta['sha256']:
            raise ValueError('Export changed: ' + name)
    curated = list(rows(directory / 'machine_curated.unverified.jsonl'))
    index = {r['case_id']: r for r in curated}
    if len(index) != len(curated) or len({r['group_id'] for r in curated}) != len(curated):
        raise ValueError('Duplicate case/group, including across splits')
    if len(curated) != manifest['machine_curated_cases']:
        raise ValueError('Export count changed')
    for split in SPLITS:
        visible = list(rows(directory / 'visible' / (split + '.jsonl')))
        hidden = list(rows(directory / 'hidden' / (split + '.jsonl')))
        selected = [r for r in curated if r['split'] == split]
        ids = [r['case_id'] for r in selected]
        if [r['case_id'] for r in visible] != ids or [r['case_id'] for r in hidden] != ids:
            raise ValueError('Visible/hidden split membership changed')
        for view in visible:
            check_visible(view, historical)
            original = index[view['case_id']]
            if view['split'] != split or view['group_id'] != original['group_id']:
                raise ValueError('Visible identity changed')
            for key in ('id', 'kind', 'text'):
                expected_values = [e[key] for e in original['evidence']]
                if key == 'kind' and not historical:
                    expected_values = [s.replace('被告人', '当事人').replace('犯罪嫌疑人', '当事人') for s in expected_values]
                if [e[key] for e in view['evidence']] != expected_values:
                    raise ValueError('Visible evidence changed')
        for answer in hidden:
            original = index[answer['case_id']]
            if answer['hidden_gold'] != original['hidden_gold'] or answer['source'] != original['source']:
                raise ValueError('Hidden answer/source changed')
    return manifest, curated


def selected_sources(root, curated, historical):
    # Recheck quotations and bindings against original records; never fetch data.
    sys.path.insert(0, str(ROOT / 'scripts'))
    if historical:
        from export_oldbailey import verify
        from prepare_oldbailey import extract
        files = ['oldbailey_candidates.jsonl']
    else:
        from export_review_bundle import verify
        files = ['candidates.jsonl', 'lecardv2_candidates.jsonl',
                 'lecardv2_mteb_candidates.jsonl', 'mnbvc_candidates.jsonl']
    wanted = {r['case_id']: r for r in curated}
    found = {}
    for name in files:
        for source in rows(root / 'data/processed' / name):
            cid = source['case_id']
            if cid not in wanted:
                continue
            if cid in found:
                raise ValueError('Duplicate selected source ID')
            if historical:
                rebuilt, reason = extract(source['raw_record'])
                if rebuilt is None:
                    raise ValueError('Historical source filter changed: ' + str(reason))
                rebuilt['source'].update({k: source['source'][k] for k in
                    ('dataset', 'revision', 'parquet', 'row_index', 'license')})
                verify(wanted[cid], rebuilt)
            else:
                verify(wanted[cid], source)
            found[cid] = source
    if set(found) != set(wanted):
        raise ValueError('Selected original source missing')
    return [found[r['case_id']] for r in curated]


def archive(directory, destination):
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(directory))


def public_report(report):
    # Keep all individual case IDs and test membership in the private package.
    return report | {'artifacts': {k: v for k, v in report['artifacts'].items()
                                   if k in ('visible-only.zip', 'review-private.zip', 'README.md')}}


def main():
    exports = {}
    for stratum, version in STRATA.items():
        exports[stratum] = load_export(ROOT / 'data/processed/review_bundle' / version,
                                       stratum != 'modern_chinese')
    binding = {s: m['artifacts'] for s, (m, _) in exports.items()}
    binding['packager_sha256'] = digest(Path(__file__))
    package_id = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:16]
    out = ROOT / 'data/processed/research_bundle' / package_id
    if out.exists():
        # Existing version is immutable: check rather than silently repair it.
        report = json.loads((out / 'manifest.json').read_text())
        for name, meta in report['artifacts'].items():
            if digest(out / name) != meta['sha256']:
                raise ValueError('Existing package changed; preserve for investigation')
        write_json(ROOT / 'data/research_bundle_report.json', public_report(report))
        print(json.dumps({'directory': str(out), 'reused': True}, ensure_ascii=False))
        return
    # Verify all source material before creating a delivery directory.
    sources = {s: selected_sources(ROOT, curated, s != 'modern_chinese')
               for s, (_, curated) in exports.items()}
    total = sum(len(curated) for _, curated in exports.values())
    readme = f'''# 本地研究材料包（未验收）

本包共{total}案，现代中文与英国1834—1913年历史案件分别保存。仅完成机器整理，人工验收0案；1000个线索充分案件及90%整案准确率均未达标。未进行模型微调。按用户要求暂停新增付费调用，此打包流程不调用API。

## 文件与使用

- visible-only.zip：可见线索JSONL及逐案中文TXT，不含隐藏答案字段、原始全文或审核意见。解压后可在本地网页上传TXT。文件中的内容本身是否泄漏结局仍需人工核验。
- review-private.zip：隐藏结局、机器审查记录、原始文书或英文XML及来源版本。仅供隔离的复核人员使用，不可作为agent输入或公开上传。
- 两个压缩包的解压内容分别对应visible/与review_private/目录，便于检查；manifest.json记录每个文件的SHA-256。
- train/dev/test沿用原导出划分，同案分组不跨集合。日常演示使用train或dev。test不得用于调参；此包向持有人提供test，不能据此宣称其仍是独立盲测。后续正式测试需隔离访问并冻结核验后的集合。

## 分组与限制

现代材料沿用v3.7导出，原始连续引文及经过句子偏移已复查。原始来源使用限制见docs/data_sources.md，不授予公开再分发权。

历史材料来自Old Bailey Proceedings，经biglam/old_bailey_proceedings固定版本8caa27eb4445098598d38ef8df04a79dd9d88903获取。来源：https://www.dhi.ac.uk/blogs/old-bailey/data/ ，原始XML DOI：https://doi.org/10.15131/shef.data.4775434.v2 。来源及译文依CC-BY-NC-4.0作署名非商业使用：https://creativecommons.org/licenses/by-nc/4.0/ 。历史分组仅含单一被告、单一罪类及有罪裁决样本，且使用承认词优先级，选择偏差明显；必须单独报告，不能将其成绩代表现代侦查能力或现代中国法律责任。

机器审核可能遗漏共同作案人、证言歧义或翻译问题。分组可能漏检关联，基础模型预训练污染未知；隐藏的裁决是来源记录，不等于客观真实。后续需按docs/human_review_protocol.md逐案核验，再冻结正式基准。
'''
    final_out = out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete directory atomically; failed builds never shadow it.
    out = Path(tempfile.mkdtemp(prefix='.' + package_id + '-', dir=out.parent))
    (out / 'README.md').write_text(readme, encoding='utf-8')
    catalog = []
    stats = {}
    for stratum, (manifest, curated) in exports.items():
        historical = stratum != 'modern_chinese'
        original = ROOT / 'data/processed/review_bundle' / STRATA[stratum]
        private = out / 'review_private' / stratum
        write_json(private / 'export_manifest.json', manifest)
        write_rows(private / 'machine_curated.unverified.jsonl', curated)
        write_rows(private / 'source_records.jsonl', sources[stratum])
        for split in SPLITS:
            views = list(rows(original / 'visible' / (split + '.jsonl')))
            write_rows(out / 'visible' / stratum / (split + '.jsonl'), views)
            write_rows(private / 'hidden' / (split + '.jsonl'), rows(original / 'hidden' / (split + '.jsonl')))
            for view in views:
                path = out / 'visible' / stratum / 'text' / split / (view['case_id'] + '.txt')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(visible_text(view, historical), encoding='utf-8')
                catalog.append({k: view[k] for k in ('case_id', 'group_id', 'split')} | {
                    'stratum': stratum, 'evidence_count': len(view['evidence']), 'status': 'machine_unverified'})
        stats[stratum] = {'cases': len(curated), 'split_counts': dict(Counter(r['split'] for r in curated)),
                          'source_version': STRATA[stratum], 'source_records': len(sources[stratum])}
    write_rows(out / 'review_private/review_catalog.jsonl', catalog)
    # README in both archives explains their intended isolation and limitations.
    for folder in ('visible', 'review_private'):
        (out / folder / 'README.md').write_text(readme, encoding='utf-8')
    archive(out / 'visible', out / 'visible-only.zip')
    archive(out / 'review_private', out / 'review-private.zip')
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'package_id': package_id,
              'local_directory': str(final_out.relative_to(ROOT)), 'strata': stats, 'cases': total,
              'qualified': 0, 'human_reviewed': 0, 'target_1000_complete': False,
              'accuracy_90_verified': False, 'paid_api_calls': 0, 'binding': binding,
              'artifacts': {str(p.relative_to(out)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                            for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out / 'manifest.json', report)
    out.rename(final_out)
    write_json(ROOT / 'data/research_bundle_report.json', public_report(report))
    print(json.dumps({'directory': str(final_out), 'cases': total, 'strata': stats}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
