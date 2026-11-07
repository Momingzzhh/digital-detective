#!/usr/bin/env python3
"""Prepare a private reading pack for automatic labels still awaiting a human."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.auxiliary_training import bound_file
from app.five_dimensions import fingerprint
from app.small_batch_training import load_dataset
from scripts.prepare_calibration_review import annotation_html

PROTOCOL = 'small-batch-pending-human-reading-v1'


def write_immutable(path, data):
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Reading pack is immutable; prepare a new bound version')
    path.write_bytes(data)


def prepare(manifest_path):
    rows, manifest = load_dataset(ROOT, manifest_path)
    pending = [r for r in rows if r['review_status'] == 'automatically_audited_pending_human_review']
    if not pending:
        raise ValueError('No automatically audited cases awaiting human review')
    cases = []
    for row in pending:
        if row['split'] != 'train' or row['human_reviewed'] or row['formal_approval']:
            raise ValueError('Reading pack accepts only unreviewed training material')
        saved = json.loads(bound_file(ROOT, row['training_provenance']['annotation_file']))
        cases.append({k: row[k] for k in ('case_id', 'group_id', 'source_binding', 'source',
                                          'evidence', 'sentence_spans', 'raw_record', 'five_dimensions')} |
                     {'audit_file': row['training_provenance']['annotation_file'],
                      'audit_wire': saved.get('semantic_audit_wire'),
                      'single_actor_without_conflict': row.get('single_actor_without_conflict', False)})
    pack = {'protocol': PROTOCOL, 'dataset_binding': manifest['dataset_binding'],
            'human_approvals_created': 0, 'purpose': 'reading_only_pending_actual_human_review',
            'renderer_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                               ('scripts/prepare_small_batch_review.py', 'scripts/prepare_calibration_review.py')},
            'cases': cases}
    pack['bundle_binding'] = fingerprint(pack)
    folder = ROOT / 'data/review/small_batch_pending_review' / pack['bundle_binding'][:16]
    folder.mkdir(parents=True, exist_ok=True)
    esc = lambda v: html.escape(str(v))
    sections = []
    for case in cases:
        sections.append('<section><h2>' + esc(case['case_id']) + '</h2><p>实际标注深度：' +
                        esc(case['five_dimensions']['reasoning_depth']) + '；单行为人无矛盾：' +
                        ('是' if case['single_actor_without_conflict'] else '否') + '</p>' +
                        '<p>来源绑定：<code>' + esc(case['source_binding']) + '</code></p>' +
                        '<details><summary>原始来源信息</summary><pre>' +
                        esc(json.dumps(case['source'], ensure_ascii=False, indent=2)) + '</pre></details>' +
                        '<div class="columns"><article><h3>当前可见证据及原句</h3>')
        for evidence in case['evidence']:
            sections.append('<h4>' + esc(evidence['id']) + ' · ' + esc(evidence['kind']) + '</h4><pre>' +
                            esc(evidence['text']) + '</pre><details><summary>句子编号和精确原文</summary>')
            for sentence in case['sentence_spans']:
                if sentence['evidence_id'] == evidence['id']:
                    sections.append('<p><b>' + esc(sentence['id']) + '</b> ' + esc(sentence['text']) + '</p>')
            sections.append('</details>')
        sections.append('</article><article><h3>助手草稿，自动审核后待人工复核</h3>' +
                        annotation_html(case['five_dimensions']) +
                        '<details><summary>本地自动审核原始分项</summary><pre>' +
                        esc(json.dumps(case['audit_wire'], ensure_ascii=False, indent=2)) +
                        '</pre></details></article></div><details><summary>原始完整文书（含裁判内容，仅供来源核对）</summary><pre>' +
                        esc(case['raw_record']['qw']) + '</pre></details></section>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>小批量训练待人工复核阅读包</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1400px;margin:28px auto;padding:0 20px;color:#182738;background:#f4f6f8}header,section{background:white;border:1px solid #d5dde5;border-radius:10px;padding:24px;margin:20px 0}h2,code{overflow-wrap:anywhere}.columns{display:grid;grid-template-columns:1fr 1fr;gap:28px}article{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.8 system-ui}blockquote{margin:10px 0;padding:12px;background:#f4f6f8}summary{cursor:pointer;color:#165b86}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style>
<header><h1>小批量训练待人工复核阅读包</h1><p>只包含新增的自动审核案例；先前具体接受稿不重复要求核对。优先真实三层，必要时两层；用户已允许单行为人且无真实矛盾的小批量案例，仍与三层复杂基准分开。</p><p>请核对来源、人物与事件支持、冲突是否真实和每条依赖是否必要。自动审核可能共享错误，概括性意见不能替代逐句核验。不确定处保持待核。</p><p>这是本机只读材料，没有签署或提交按钮，不产生任何人工批准，也不计入验证／测试。请勿上传公共服务；页面不加载外部资源。</p></header>'''
    document += ''.join(sections) + '</html>'
    write_immutable(folder / 'reading.private.json', (json.dumps(pack, ensure_ascii=False, indent=2) + '\n').encode())
    write_immutable(folder / 'review.html', document.encode())
    public = {'protocol': PROTOCOL, 'dataset_binding': manifest['dataset_binding'],
              'pending_training_cases_in_reading_pack': len(cases),
              'single_actor_without_conflict_cases': sum(c['single_actor_without_conflict'] for c in cases),
              'bundle_binding': pack['bundle_binding'], 'contains_only_train': True,
              'human_reviews_completed': 0, 'human_approvals_created': 0,
              'reading_only_no_signoff_controls': True, 'private_materials_excluded_from_git': True}
    (ROOT / 'data/small_batch_pending_review_report.json').write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'private_reading_pack': str((folder / 'review.html').relative_to(ROOT)),
                      'cases': len(cases), 'bundle_binding': pack['bundle_binding'],
                      'human_approvals_created': 0}, ensure_ascii=False), flush=True)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    prepare((ROOT / args.manifest).resolve())
