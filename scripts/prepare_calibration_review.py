#!/usr/bin/env python3
"""Prepare train-only, provisional source/label review without certifying cases."""
import argparse
from datetime import datetime, timezone
import html
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.five_dimensions import fingerprint

RUBRIC = 'train-only-calibration-v1'
QUESTIONS = {
    'actors': '人物与角色：是否漏掉有犯罪行为线索的人，或误收证人、被害人、普通交易者？请列姓名、主要动作及支持句子ID。',
    'events': '关键经过：是否覆盖发起、实施、结果？请指出漏项、无依据内容及支持句子ID。',
    'conflicts': '真实矛盾：哪些陈述确实相冲突？请列两句ID；互补分工或不知情不自动算矛盾。没有可写“无”。',
    'depth': '必要推理：是否确实需要至少三层证据关联？请给简短可核验的依赖关系，不能仅按时间排序凑层数。',
    'evidence': '材料质量：证据层级是否恰当，线索是否足够，是否混入裁判结论或不相干内容？请指出ID。',
}
DECISIONS = ('needs_revision', 'exclude', 'no_correction_found')


def pack_case(row, saved):
    if row['split'] != 'train' or saved['split'] != 'train':
        raise ValueError('Calibration may only expose training cases')
    if row['source_binding'] != saved['source_binding']:
        raise ValueError('Source binding differs')
    from scripts.export_review_bundle import verify
    verify(row, {'raw_record':row['raw_record']})
    annotation = saved.get('five_dimensions') or saved.get('raw_annotation')
    if not annotation:
        raise ValueError('No provisional annotation to inspect')
    return {'case_id':row['case_id'],'group_id':row['group_id'],'split':'train',
        'source_binding':row['source_binding'],'annotation_result_binding':fingerprint(saved),
        'source':row['source'],'source_fulltext':row['raw_record']['qw'],
        'evidence':row['evidence'],'sentence_spans':row['sentence_spans'],
        'annotation':annotation,'machine_status':saved['status'],
        'machine_feedback':saved.get('semantic_audit',saved.get('error',''))}


def annotation_html(annotation):
    def esc(value):return html.escape(str(value))
    def refs(values):return '、'.join(esc(v) for v in values) or '无'
    parts=['<h4>人物与主要动作</h4>']
    for actor in annotation.get('actor_assessments',[]):
        parts.append(f'<p><b>{esc(actor["name"])} · {esc(actor["participation"])}</b><br>{esc(actor["role"])}<br>支持句子：{refs(actor["sentence_ids"])}；别名：{refs(actor.get("aliases",[]))}</p>')
    parts.append('<h4>关键经过</h4><ol>')
    for event in annotation.get('events',[]):
        parts.append(f'<li>{esc(event["claim"])}<br><small>支持句子：{refs(event["sentence_ids"])}</small></li>')
    parts.append('</ol><h4>证据关联与依赖</h4>')
    for step in annotation.get('logic_chain',[]):
        parts.append(f'<p><b>{esc(step["id"])}</b> {esc(step["claim"])}<br><small>证据：{refs(step["evidence_ids"])}；依赖先前步骤：{refs(step["depends_on_steps"])}</small></p>')
    parts.append('<h4>候选矛盾点</h4>')
    conflicts=annotation.get('contradictions',[])
    if not conflicts:parts.append('<p>未提出矛盾；仍需人工核对是否确实没有。</p>')
    for conflict in conflicts:
        parts.append(f'<p><b>{esc(conflict["issue"])}</b></p><blockquote>{esc(conflict["left_id"])}：{esc(conflict["left_quote"])}<br>{esc(conflict["right_id"])}：{esc(conflict["right_quote"])}</blockquote><p>处理状态：{esc(conflict["resolution"])}</p>')
    parts.append('<details><summary>逐条证据层级</summary>')
    for tier in annotation.get('evidence_hierarchy',[]):
        parts.append(f'<p>{esc(tier["evidence_id"])}：{esc(tier["tier"])}</p>')
    parts.append('</details>')
    coverage=lambda key:'自报完整（须核实）' if annotation.get(key) else '自报不完整'
    parts.append(f'<p>人物覆盖：{coverage("complete_actor_coverage")}；经过覆盖：{coverage("complete_event_coverage")}。</p>')
    return ''.join(parts)


def page(cases, bundle_binding):
    esc = html.escape
    parts = []
    for number, case in enumerate(cases,1):
        evidence = []
        for item in case['evidence']:
            spans=[s for s in case['sentence_spans'] if s['evidence_id']==item['id']]
            lines=''.join(f'<p><b>{esc(s["id"])}</b> {esc(s["text"])}</p>' for s in spans)
            evidence.append(f'<details><summary>{esc(item["id"])} · {esc(item["kind"])}</summary>{lines}</details>')
        questions=''.join(f'<label>{esc(title)}<textarea data-answer="{key}" rows="3" required></textarea></label>' for key,title in QUESTIONS.items())
        annotation=esc(json.dumps(case['annotation'],ensure_ascii=False,indent=2))
        feedback=esc(json.dumps(case['machine_feedback'],ensure_ascii=False,indent=2))
        parts.append(f'''<section data-case="{esc(case['case_id'])}"><h2>{number:02d} · {esc(case['case_id'])}</h2>
<p>训练集 · 机器状态：{esc(case['machine_status'])} · 尚未人工确认</p>
<details><summary>完整来源文书（含裁判记录，仅供标注核对）</summary><pre>{esc(case['source_fulltext'])}</pre></details>
<div class="columns"><article><h3>可见证据与句子ID</h3>{''.join(evidence)}</article>
<article><h3>机器候选标注（可能有误）</h3>{annotation_html(case['annotation'])}<details><summary>查看候选标注JSON</summary><pre>{annotation}</pre></details><details><summary>机器拒绝理由（也需核实）</summary><pre>{feedback}</pre></details></article></div>
{questions}<label>处理建议<select data-decision><option value="">尚未核对</option><option value="needs_revision">需要修正标注</option><option value="exclude">材料不足／不满足复杂度，应排除</option><option value="no_correction_found">本次核对未发现需修改项</option></select></label>
<label class="attest"><input type="checkbox" data-attestation>我已亲自对照本案原文完成以上核对，未以机器意见代替人工判断。</label></section>''')
    metadata = json.dumps({'rubric':RUBRIC,'bundle_binding':bundle_binding,
        'cases':[{k:c[k] for k in ('case_id','split','source_binding','annotation_result_binding')} for c in cases]},ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练集人工校准包</title><style>
body{font:16px/1.7 system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 24px;color:#182738;background:#f4f6f8}
header,section{background:white;border:1px solid #d5dde5;border-radius:10px;padding:24px;margin-bottom:24px}
h1,h2,h3{line-height:1.4;overflow-wrap:anywhere}summary{cursor:pointer;color:#165b86;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.8 system-ui}blockquote{margin:12px 0;padding:12px;background:#f4f6f8}h4{margin:24px 0 8px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{min-width:0}label{display:block;margin:16px 0}textarea{box-sizing:border-box;width:100%;font:inherit}
input,select,button{font:inherit;padding:8px}button{background:#165b86;color:white;border:0;border-radius:6px;cursor:pointer}.attest{font-weight:600}
@media(max-width:800px){.columns{grid-template-columns:1fr}}.notice{color:#8b391e}
</style><header><h1>训练集人工校准包</h1>
<p>目的：纠正人物、经过、矛盾与必要推理深度的标注标准。全部为训练集，候选标注尚未通过质量门槛；本次核对不计入正式合格数据，不会替代验证／测试的完整人工审核。</p>
<p>先做前3案即可。逐案先读原文和证据，再对照候选，填写5项意见并选择处理建议；不确定时写明原因，不必强行通过。请保留证据或句子ID。</p>
<p class="notice">材料只应留在本机，不要上传到GitHub或公共服务。页面没有外部网络请求。关闭前请导出，未导出的文字不会保存。</p>
<label>实际核对人姓名或研究编号 <input id="reviewer" maxlength="120" autocomplete="off"></label>
<button id="export">导出已完成核对（JSON）</button><p id="message" role="status"></p></header>'''+''.join(parts)+f'''<script type="application/json" id="metadata">{metadata}</script>'''+'''
<script>
document.getElementById('export').addEventListener('click',()=>{
  const reviewer=document.getElementById('reviewer').value.trim();
  const status=document.getElementById('message');
  if(!reviewer){status.textContent='请填写实际核对人姓名或研究编号。';return;}
  const metadata=JSON.parse(document.getElementById('metadata').textContent), reviews=[];
  for(const section of document.querySelectorAll('section[data-case]')){
    const decision=section.querySelector('[data-decision]').value;
    if(!decision)continue;
    const answers=Object.fromEntries([...section.querySelectorAll('[data-answer]')].map(e=>[e.dataset.answer,e.value.trim()]));
    if(!section.querySelector('[data-attestation]').checked || Object.values(answers).some(v=>!v)){
      status.textContent='已选处理建议的案件必须填写全部5项并确认亲自核对；未完成案件请选“尚未核对”。';return;
    }
    reviews.push({...metadata.cases.find(c=>c.case_id===section.dataset.case),decision,answers,
      reviewer_id:reviewer,reviewed_at:new Date().toISOString(),human_attestation:true});
  }
  if(!reviews.length){status.textContent='尚无已完成核对的案件。';return;}
  const output={rubric:metadata.rubric,bundle_binding:metadata.bundle_binding,reviews};
  const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)+'\\n'],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='training-calibration-review.json';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent=`已导出 ${reviews.length} 案。请把导出文件的本机路径告诉Codex；这不会自动批准任何正式数据。`;
});
</script></html>'''


def validate_submission(value, manifest):
    expected=fingerprint({'rubric':manifest.get('rubric'),'cases':manifest['cases']})
    if manifest.get('rubric') != RUBRIC or manifest['bundle_binding'] != expected:
        raise ValueError('Stored calibration snapshot changed')
    if value.get('rubric') != RUBRIC or value.get('bundle_binding') != manifest['bundle_binding']:
        raise ValueError('Review bundle binding differs')
    index={c['case_id']:c for c in manifest['cases']};seen=set()
    if not value.get('reviews'):raise ValueError('No completed reviews')
    for review in value['reviews']:
        cid=review.get('case_id')
        if cid not in index or cid in seen or not re.fullmatch('[A-Za-z0-9_-]+',cid):raise ValueError('Unknown or duplicate case')
        seen.add(cid);case=index[cid]
        if any(review.get(k)!=case[k] for k in ('split','source_binding','annotation_result_binding')) or review['split']!='train':
            raise ValueError('Review source/annotation binding differs')
        if review.get('decision') not in DECISIONS or review.get('human_attestation') is not True:
            raise ValueError('Explicit human review required')
        if not isinstance(review.get('reviewer_id'),str) or not review['reviewer_id'].strip():raise ValueError('Missing reviewer')
        try:stamp=datetime.fromisoformat(review['reviewed_at'].replace('Z','+00:00'))
        except (ValueError,KeyError,TypeError):raise ValueError('Missing review time')
        if stamp.tzinfo is None:raise ValueError('Review time needs timezone')
        answers=review.get('answers',{})
        if set(answers)!=set(QUESTIONS) or any(not isinstance(v,str) or not v.strip() for v in answers.values()):
            raise ValueError('All calibration questions need a human answer')
    return value['reviews']


def main(args):
    output=ROOT/'artifacts/calibration'/RUBRIC
    if args.import_review:
        value=json.loads(Path(args.import_review).expanduser().read_text())
        binding=value.get('bundle_binding','')
        if not re.fullmatch('[0-9a-f]{64}',binding):raise ValueError('Invalid bundle binding')
        # Locate the exact immutable snapshot, even after newer packages exist.
        manifest=json.loads((output/binding[:16]/'manifest.private.json').read_text())
        reviews=validate_submission(value,manifest)
        for review in reviews:
            path=ROOT/'data/review/five_dimensions_v2_calibration'/review['case_id']
            path.mkdir(parents=True,exist_ok=True)
            # Append-only: calibration feedback never writes formal approval paths.
            destination=path/(fingerprint(review)+'.json')
            if not destination.exists():destination.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'imported':len(reviews),'formal_approvals_created':0}));return
    rows={r['case_id']:r for r in map(json.loads,(ROOT/'data/processed/five_dimensions_v2/candidates.private.jsonl').open()) if r['split']=='train'}
    saved=[json.loads(p.read_text()) for p in (ROOT/'data/review/five_dimensions_v2').glob('*.json')]
    saved=[s for s in saved if s['case_id'] in rows and (s.get('five_dimensions') or s.get('raw_annotation'))]
    saved.sort(key=lambda s:fingerprint(rows[s['case_id']]['group_id']))
    cases=[pack_case(rows[s['case_id']],s) for s in saved[:args.limit]]
    if not cases:raise ValueError('No train-only provisional annotations available')
    binding=fingerprint({'rubric':RUBRIC,'cases':cases})
    output=output/binding[:16];output.mkdir(parents=True,exist_ok=True)
    manifest={'rubric':RUBRIC,'bundle_binding':binding,'generated_at':datetime.now(timezone.utc).isoformat(),'cases':cases}
    (output/'manifest.private.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (output/'index.html').write_text(page(cases,binding))
    # A pointer only, so previous immutable packages and their reviews survive.
    (output.parent/'manifest.private.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'html':str(output/'index.html'),'train_cases':len(cases),'formal_approvals_created':0},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--import-review')
    args=parser.parse_args()
    if not 1 <= args.limit <= 20:parser.error('Calibration packages contain 1–20 training cases')
    main(args)
