#!/usr/bin/env python3
"""Render a complete corrected auxiliary draft with specific human check points."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.five_dimensions import fingerprint, validate_auxiliary_annotation
from scripts.export_review_bundle import verify
from scripts.prepare_calibration_review import annotation_html

PROTOCOL = 'auxiliary-correction-targeted-review-v1'


def load_draft(root, draft_path):
    draft = json.loads(draft_path.read_text())
    if (draft.get('split') != 'train' or draft.get('human_reviewed') is not False or
        draft.get('formal_approval') is not False or draft.get('qualified_cases_added') != 0 or
        draft.get('status') != 'auxiliary_annotation_pending_human_review'):
        raise ValueError('Review requires a pending training-only correction, without synthesized approval')
    spec = draft['source_dataset']; source_path = (root / spec['path']).resolve()
    if (not source_path.is_relative_to(root.resolve()) or
        hashlib.sha256(source_path.read_bytes()).hexdigest() != spec['sha256']):
        raise ValueError('Corrected draft source file changed')
    rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    matches = [r for r in rows if r['case_id'] == draft['case_id'] and r['source_binding'] == draft['source_binding']]
    if len(matches) != 1 or matches[0]['split'] != 'train':
        raise ValueError('Corrected source must resolve exactly once in the training split')
    row = matches[0]
    verify(row, {'raw_record':row['raw_record']})
    checked = validate_auxiliary_annotation(row,draft['raw_annotation'],row['sentence_spans'])
    if (fingerprint(checked) != fingerprint(draft['five_dimensions']) or
        checked['reasoning_depth'] != draft['computed_graph_depth']):
        raise ValueError('Correction differs from its complete checked annotation')
    spans = {s['id']:s for s in row['sentence_spans']}
    if not draft.get('review_checks') or any(not set(q['sentence_ids']) <= spans.keys() for q in draft['review_checks']):
        raise ValueError('Human check points must cite known source sentences')
    return draft,row


def render(draft,row,binding):
    esc=html.escape;spans={s['id']:s for s in row['sentence_spans']}
    ann=draft['five_dimensions'];check_count=len(draft['review_checks'])
    questions=[]
    for number,q in enumerate(draft['review_checks'],1):
        quotes=''.join(f'<blockquote><b>{esc(sid)}</b> {esc(spans[sid]["text"])}</blockquote>' for sid in q['sentence_ids'])
        questions.append(f'<section class="check"><h3>{number}．{esc(q["question"])}</h3>{quotes}</section>')
    evidence=[]
    for item in row['evidence']:
        lines=''.join(f'<p id="{esc(s["id"])}"><b>{esc(s["id"])}</b> {esc(s["text"])}</p>'
                      for s in row['sentence_spans'] if s['evidence_id']==item['id'])
        evidence.append(f'<details><summary>{esc(item["id"])} · {esc(item["kind"])}</summary>{lines}</details>')
    limits=''.join(f'<li>{esc(t)}</li>' for t in draft.get('limitations',[]))
    excluded=''.join(f'<li><b>{esc(name)}</b>：{esc(reason)}</li>'
                     for name,reason in draft.get('excluded_from_behavior_participants',{}).items())
    annotation=annotation_html(draft['five_dimensions'])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>辅助训练新修订稿核对</title><style>
body{{font:16px/1.75 system-ui,sans-serif;max-width:1220px;margin:32px auto;padding:0 24px;background:#f4f6f8;color:#182738}}
header,section,article{{background:#fff;border:1px solid #d5dde5;border-radius:10px;padding:22px;margin-bottom:20px}}
h1,h2,h3,h4{{line-height:1.5;overflow-wrap:anywhere}}blockquote{{margin:12px 0;padding:12px 16px;background:#edf3f7;border-left:3px solid #44738c}}
details{{margin:12px 0}}summary{{cursor:pointer;color:#165b86;font-weight:600}}pre,p,li,blockquote{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;font:14px/1.8 system-ui}}
.columns{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}article{{min-width:0}}.notice{{color:#8b391e}}small{{color:#516371}}
@media(max-width:850px){{.columns{{grid-template-columns:1fr}}}}@media print{{body{{background:white;margin:0}}header,section,article{{break-inside:avoid}}}}
</style><header><h1>辅助训练新修订稿：核对下面{check_count}处</h1>
<p>案例：{esc(draft['case_id'])}<br>材料版本：<b>{esc(binding[:16])}</b> · 训练来源 · 助手修订，尚未人工确认</p>
<p>本稿有{len(ann['actor_assessments'])}名待排查对象、{len(ann['events'])}项经过和{len(ann['contradictions'])}项待核对差异。图深度为<b>{draft['computed_graph_depth']}层</b>；尚未人工确认复杂度，不计入600案正式基准。</p>
<p>已按此前意见修订人物、经过和来源支持。无需重填旧Word；请核对下面各项，回复“正确”或指出应改之处。</p>
<p class="notice">数量的样品对应、先后顺序差异和人员范围仍待你的实际核对。用途沿用已选的辅助训练，本新版本尚未加入训练集，也没有生成正式审批。</p></header>
<section><h2>需要人工核对的具体内容</h2>{''.join(questions)}
<p>可直接在聊天按问题编号回复。若全部无异议，可回复“此版本各项均正确，按{draft['computed_graph_depth']}层修订稿使用”；若材料不足，直接写明哪项不能确认。</p></section>
<div class="columns"><article><h2>完整修订标注</h2>{annotation}<h3>未列为本次贩卖行为参与者</h3><ul>{excluded}</ul></article>
<article><h2>全套可见原始材料</h2><p>点击各证据条目展开逐句原文。原始文字及句子ID未改写。</p>{''.join(evidence)}
<h3>保留的限制</h3><ul>{limits}</ul></article></div>
<section><details><summary>完整来源文书（含裁判记录，仅供人工对照；不进入Agent输入）</summary><pre>{esc(row['raw_record']['qw'])}</pre></details>
<details><summary>标注JSON及版本信息</summary><pre>{esc(json.dumps(draft,ensure_ascii=False,indent=2))}</pre></details></section></html>'''


def main(args):
    path=args.draft.resolve();draft,row=load_draft(ROOT,path)
    inputs={'protocol':PROTOCOL,'draft_file':{'path':str(path.relative_to(ROOT)),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'source_file':draft['source_dataset'],
            'source_binding':row['source_binding'],'code_sha256':{
                p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                ('scripts/prepare_auxiliary_correction_review.py','scripts/prepare_calibration_review.py','app/five_dimensions.py')}}
    binding=fingerprint(inputs);out=ROOT/'artifacts/auxiliary_correction_review'/binding[:16]
    out.mkdir(parents=True,exist_ok=True)
    page=render(draft,row,binding).encode()
    manifest=inputs|{'review_binding':binding,'case_id':row['case_id'],'split':'train',
                     'index_sha256':hashlib.sha256(page).hexdigest(),'human_reviewed':False,
                     'formal_approval':False,'qualified_cases_added':0}
    for target,content in ((out/'index.html',page),(out/'manifest.private.json',
            (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())):
        if target.exists() and target.read_bytes()!=content:
            raise ValueError('Refusing to overwrite a different review version')
        target.write_bytes(content)
    if args.copy_to:
        if args.copy_to.exists() and args.copy_to.read_bytes()!=page:
            raise ValueError('Choose a new delivery filename for a different review version')
        args.copy_to.parent.mkdir(parents=True,exist_ok=True);args.copy_to.write_bytes(page)
    print(json.dumps({'review_binding':binding,'review_page':str((out/'index.html').relative_to(ROOT)),
                      'delivered_to':str(args.copy_to) if args.copy_to else None,
                      'human_check_points':len(draft['review_checks']),'qualified_cases_added':0},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',type=Path,required=True)
    parser.add_argument('--copy-to',type=Path)
    main(parser.parse_args())
