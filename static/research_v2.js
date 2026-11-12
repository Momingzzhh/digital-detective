'use strict';
const $ = s => document.querySelector(s);
let csrf = '', current = null;
async function api(path, body) {
  const r = await fetch('/api' + path, {method: body ? 'POST' : 'GET',
    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf},
    body: body ? JSON.stringify(body) : undefined});
  const d = await r.json();
  if (!r.ok) throw Error(typeof d.detail === 'string' ? d.detail : '请检查填写格式。');
  return d;
}
async function safe(fn) { try { await fn(); } catch (e) { $('#notice').textContent = e.message; } }
async function open() {
  current = null; $('#work').hidden = true;
  const d = await api('/research/v2/items/' + encodeURIComponent($('#cases').value));
  current = d;
  $('#source').textContent = JSON.stringify({source: d.source_metadata, text: d.source}, null, 2);
  $('#case').textContent = JSON.stringify(d.case, null, 2);
  $('#legacy').textContent = JSON.stringify({annotation: d.legacy_annotation, sentences: d.sentence_spans}, null, 2);
  $('#audit').textContent = JSON.stringify({attempts: d.local_audits, assistant_disposition: d.assistant_audit_disposition}, null, 2);
  $('#solvability').textContent = JSON.stringify(d.solvability, null, 2);
  $('#checks').replaceChildren();
  for (const [key, text] of Object.entries(d.required_checks)) {
    const label = document.createElement('label'), box = document.createElement('input');
    box.type = 'checkbox'; box.dataset.check = key; box.checked = false;
    label.append(box, document.createTextNode(text)); $('#checks').append(label);
  }
  $('#notes').value = ''; $('#minutes').value = ''; $('#attest').checked = false;
  $('#work').hidden = false;
  $('#notice').textContent = '当前复核状态：' + d.receipt.status + '。修改来源或标注后旧记录失效。';
}
async function submit(decision) {
  if (!current) throw Error('请先打开材料。');
  if (!$('#attest').checked) throw Error('请由本人完成核对后确认。');
  const checks = Object.fromEntries([...document.querySelectorAll('[data-check]')].map(b => [b.dataset.check, b.checked]));
  const result = await api('/research/v2/items/' + encodeURIComponent(current.case.case_id) + '/review', {
    review_binding: current.review_binding, decision, checks, notes: $('#notes').value,
    human_minutes: Number($('#minutes').value), reviewer_level: 'single_nonexpert', personally_reviewed: true});
  $('#notice').textContent = '记录已保存：' + result.decision + '。未开启正式评测。';
}
$('#open').onclick = () => safe(open);
$('#approve').onclick = () => safe(() => submit('approve_curation'));
$('#reject').onclick = () => safe(() => submit('reject'));
safe(async () => {
  csrf = (await api('/me')).csrf;
  const items = await api('/research/v2/items');
  for (const item of items) {
    const o = document.createElement('option'); o.value = item.case_id;
    o.textContent = item.case_id + ' · ' + item.evidence_count + '条证据'; $('#cases').append(o);
  }
  $('#notice').textContent = '已准备 ' + items.length + ' 个训练案例；请选择并打开。';
});
