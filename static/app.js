'use strict';
const $ = s => document.querySelector(s);
let csrf = '', current = null, polling = null, pollVersion = 0, navigationVersion = 0, statusData = null;
function stopPolling() { clearTimeout(polling); pollVersion++; }
function el(tag, text, cls) { const n = document.createElement(tag); if(text !== undefined) n.textContent = text; if(cls) n.className = cls; return n; }
function toast(text) { $('#toast').textContent=text; $('#toast').hidden=false; clearTimeout(toast.timer); toast.timer=setTimeout(()=>$('#toast').hidden=true,6500); }
async function api(path, options={}) {
  const headers = {'X-CSRF-Token':csrf, ...options.headers};
  if(options.body && !(options.body instanceof FormData)) { headers['Content-Type']='application/json'; options.body=JSON.stringify(options.body); }
  const r=await fetch('/api'+path, {...options,headers});
  let data; try { data=await r.json(); } catch { data={detail:'服务响应异常。'}; }
  if(!r.ok) { if(r.status===401 && !path.includes('login')) $('#auth-dialog').showModal(); throw Error(typeof data.detail==='string'?data.detail:'输入格式不正确，请核对内容。'); }
  return data;
}
function show(view) { if(view!=='workspace')stopPolling(); for(const id of ['welcome','workspace','dataset']) $('#'+id).hidden=id!==view; $('#workspace-nav').classList.toggle('active',view!=='dataset');$('#dataset-nav').classList.toggle('active',view==='dataset');$('.breadcrumb').textContent='研究空间 / '+(view==='dataset'?'数据与评测':'案件工作台'); }
function tab(name) { document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('selected',b.dataset.tab===name);b.setAttribute('aria-selected',b.dataset.tab===name);}); for(const id of ['overview','evidence','graph','conclusion']) $('#tab-'+id).hidden=id!==name; }
async function refreshCases() {
  const cases=await api('/cases'); $('#case-list').replaceChildren(); $('#case-count').textContent=cases.length;
  for(const c of cases) { const b=el('button','◌  '+c.title,'case-link'+(current?.id===c.id?' active':'')); b.onclick=()=>openCase(c.id).catch(e=>toast(e.message)); $('#case-list').append(b); }
}
async function refreshStatus() {
  statusData=await api('/status'); $('#model-label').textContent=statusData.model; $('#model-dot').classList.toggle('ready',statusData.configured);
  const d=$('#dataset-content');d.replaceChildren(el('h2','当前本机版本'));
  const release=statusData.deployment;
  d.append(el('p',`当前模型：${statusData.model}。`));
  if(release?.training_verified) {
    d.append(el('p',`已完成小批量训练：${release.training_cases} 案、${release.train_examples} 条训练示例、${release.optimizer_updates} 次参数更新。`));
    d.append(el('p',`已启用训练版本：${release.run_id}；权重校验：${release.adapter_sha256.slice(0,16)}…`));
  } else d.append(el('p','尚未启用经过训练记录核验的小批量权重。'));
  if(release?.error)d.append(el('p',release.error,'danger'));
  d.append(el('p','本轮以本机完整流程可用为交付目标；独立案件准确率尚未验证。'),el('h2','后续研究验收'));
  const ds=statusData.dataset;
  const f=statusData.five_dimensions;
  for(const text of ['推理方式：本地离线模型；外部模型API已停用。', `长期五维复杂案件目标：600案，训练420／验证90／测试90；整案准确率目标70%。`, `机器标注材料：${f?.machine_annotated||0}；人工复核通过：${f?.human_reviewed||0}；研究验收：${f?.training_complete?'已通过':'未完成'}。`, '候选数、机器标注数、人工复核数和验收成绩分别记录。']) d.append(el('p',text));
  if(f) d.append(el('p',`当前结构候选：${f.pool_cases||0}；数据集完整导出：${f.dataset_target_assembled?'是':'否'}；验证集通过：${f.validation_passed?'是':'否'}。`));
  d.append(el('p','历史分组与旧版1000案90%目标记录作为历史实验保留，不并入本轮复杂刑案评测。'));

}
function empty(node, text) { node.replaceChildren(); const d=el('div',undefined,'empty');d.append(el('div','◇','empty-symbol'),el('p',text));node.append(d); }
function refs(parent, ids) { for(const id of ids) {const b=el('button',id,'citation'); b.onclick=()=>{tab('evidence'); const n=document.getElementById(id); n?.scrollIntoView({behavior:'smooth',block:'center'});};parent.append(b);} }
function section(parent,title) {const n=el('section',undefined,'result-section');n.append(el('h3',title));parent.append(n);return n;}
function render(caseData) {
  current=caseData; show('workspace');$('#current-title').textContent=caseData.title;$('#case-meta').textContent=`${caseData.sources.length} 份材料 · ${caseData.evidence.length} 条原文证据 · ${caseData.confidence_history?.length||0} 轮分析`;
  const overview=$('#tab-overview');overview.replaceChildren();
  for(const source of caseData.sources) {const block=el('section',undefined,'source-block');block.append(el('h3',source.id+' / '+source.name),el('pre',source.text));overview.append(block);}
  const ev=$('#tab-evidence');ev.replaceChildren();$('#evidence-count').textContent=caseData.evidence.length;
  if(!caseData.evidence.length) empty(ev,'开始分析后，这里会展示定位到原文的证据。');
  for(const e of caseData.evidence) { const d=el('article',undefined,'evidence-card');d.id=e.id;d.append(el('span',e.kind+' · '+e.reliability,'tag'),el('h3',e.summary),el('blockquote',e.quote),el('small',`${e.id} · 来源 ${e.source_id} · 字符 ${e.start}–${e.end}`));ev.append(d); }
  const messages=$('#messages');messages.replaceChildren();
  if(!caseData.messages.length) {const intro=el('div',undefined,'bubble assistant');intro.append(el('span','数字神探','role'),el('span','案情已保存。点击下方「开始分析」，我会先核对现有证据，再询问关键缺口。'));messages.append(intro);}
  for(const m of caseData.messages) {const d=el('div',undefined,'bubble '+m.role);d.append(el('span',m.role==='user'?'研究人员':'数字神探','role'),el('span',m.text));messages.append(d);}
  messages.scrollTop=messages.scrollHeight;
  $('#questions').replaceChildren();
  if(caseData.analysis?.questions?.length) $('#questions').append(el('div','待补线索：'+caseData.analysis.questions[0],'question'));
  renderConclusion(caseData);drawGraph(caseData.graph);renderJob(caseData.job||{state:'idle'});
}
function renderConclusion(c) {
  const node=$('#tab-conclusion');node.replaceChildren();const a=c.analysis;
  if(!a) {empty(node,'完成分析后，将展示结论、证据依据、竞争假设和未解问题。');return;}
  node.append(el('div',a.status,'result-status'),el('p',a.summary,'summary'));
  const principal=section(node,'当前判断');principal.append(el('p','罪名：'+(a.charges.join('、')||'暂不确定')),el('p','作案者：'+(a.perpetrators.join('、')||'暂不确定')));
  const people=section(node,'全案参与人及角色');for(const p of a.participants||[]) {const line=el('p',`${p.name} · ${p.participation} · ${p.role}`);refs(line,p.evidence_ids);people.append(line);}
  for(const [title,key] of [['证据依据','findings'],['关键经过','reconstruction']]) {const s=section(node,title);if(!a[key].length)s.append(el('p','尚无足够依据。'));for(const f of a[key]) {const p=el('p',f.claim);refs(p,f.evidence_ids);s.append(p);}}
  const conflicts=section(node,'矛盾与冲突');if(!a.conflicts.length)conflicts.append(el('p','当前未识别出明确矛盾；不代表材料已被证实。'));for(const f of a.conflicts) {const p=el('p',`${f.severity} · ${f.description}`);refs(p,f.evidence_ids);conflicts.append(p);}
  const hypotheses=section(node,'竞争假设');for(const h of a.hypotheses) {hypotheses.append(el('p',h.description));const p=el('p','支持：');refs(p,h.supporting);p.append(el('span',' 反证：'));refs(p,h.opposing);hypotheses.append(p);}
  const gaps=section(node,'需要继续核验');for(const s of a.missing_information)gaps.append(el('p','• '+s));
  const review=section(node,'第二轮复核');review.append(el('p',c.review?.supported?'同一本地模型在第二轮未发现明确的无依据结论；仍需研究人员复核。':'第二轮复核未通过，已将结论降为证据不足。'));for(const issue of [...(c.review?.unsupported_claims||[]),...(c.review?.unresolved_issues||[])])review.append(el('p','• '+issue));
  const binding=c.model_history?.at(-1);if(binding)review.append(el('small',binding.adapter_sha256?`本轮微调权重：${binding.adapter_sha256.slice(0,16)}…`:'本轮使用基座模型。'));
  const confidence=section(node,'置信度变化');confidence.append(el('p',`当前 ${(a.confidence*100).toFixed(0)}% · 模型主观估计，未经校准，不代表实测准确率。`,'confidence'));const bars=el('div',undefined,'history-bar');for(const p of c.confidence_history||[]) {const bar=el('div');bar.style.height=(p.value*65)+'px';bar.title=`第 ${p.round} 轮：${Math.round(p.value*100)}%`;bars.append(bar);}confidence.append(bars);
  const trace=section(node,'工具执行记录');for(const t of c.trace||[])trace.append(el('p',`${t.action} · ${t.summary}`));
}
function drawGraph(graph) {
  const svg=$('#evidence-graph');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
  const make=(tag,attrs)=>{const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);return n;};
  if(!graph?.nodes.length){$('#graph-detail').textContent='提取证据后，显示证据、人物实体与分析结论之间的关系。';return;}
  $('#graph-detail').textContent='点击节点查看原文或依据。实体被提及不代表其参与犯罪。';
  const groups=['evidence','entity','finding'].map(k=>graph.nodes.filter(n=>n.kind===k));
  const height=Math.max(560,...groups.map(g=>g.length*62+50));svg.setAttribute('viewBox',`0 0 760 ${height}`);svg.style.height=Math.min(height,1200)+'px';
  const positions={};groups.forEach((g,j)=>g.forEach((n,i)=>positions[n.id]={x:100+j*280,y:40+i*62}));
  for(const e of graph.edges) {const a=positions[e.source],b=positions[e.target];if(!a||!b)continue;svg.append(make('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#d5e0d3','stroke-width':1.3}));}
  for(const n of graph.nodes) {const p=positions[n.id];const g=make('g',{tabindex:0,role:'button','aria-label':n.label});const fill={evidence:'#7fa68a',entity:'#c3ac76',finding:'#8c9fb7'}[n.kind];g.append(make('circle',{cx:p.x,cy:p.y,r:8,fill}));const text=make('text',{x:p.x,y:p.y+24,'text-anchor':'middle'});text.textContent=n.label.slice(0,16);g.append(text);const select=()=>$('#graph-detail').textContent=n.label+'\n'+n.detail;g.onclick=select;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' ')select();};svg.append(g);}
}
function renderJob(job) { const active=job.state==='running';$('#analyze').disabled=active;$('#message-form button').disabled=active;$('#delete').disabled=active;$('#analyze').textContent=active?'分析进行中…':current?.analysis?'根据现有材料重新分析':'开始分析现有材料';const n=$('#job-status');n.hidden=!['running','error'].includes(job.state);n.classList.toggle('error',job.state==='error');n.textContent=job.state==='error'?job.error:'◌ '+job.stage+'…'; }
async function openCase(id) {stopPolling();const version=++navigationVersion;const c=await api('/cases/'+id);if(version!==navigationVersion)return;render(c);await refreshCases();if(version===navigationVersion && !$('#workspace').hidden && c.job.state==='running')startPoll(id);}
function startPoll(id) {
  stopPolling();const version=pollVersion;
  const active=()=>version===pollVersion && current?.id===id && !$('#workspace').hidden;
  async function poll() {try {
    const c=await api('/cases/'+id);if(!active())return;
    renderJob(c.job);
    if(c.job.state==='running')polling=setTimeout(poll,1500);
    else {render(c);if(c.job.state==='completed')tab('conclusion');await refreshStatus();}
  } catch(e) {if(active()){stopPolling();toast(e.message);}}}
  polling=setTimeout(poll,1500);
}
async function startAnalysis() {if(!current)return;const id=current.id;try {const job=await api('/cases/'+id+'/analyze',{method:'POST'});if(current?.id===id && !$('#workspace').hidden){renderJob(job);startPoll(id);}}catch(e){toast(e.message);}}
async function login(register=false) {const name=$('#auth-name').value,password=$('#auth-password').value;if(!$('#auth-form').reportValidity())return;$('#auth-error').textContent='';try {if(register)await api('/register',{method:'POST',body:{name,password}});const data=await api('/login',{method:'POST',body:{name,password}});csrf=data.csrf;$('#username').textContent=data.name;$('#auth-dialog').close();$('#auth-password').value='';await Promise.all([refreshCases(),refreshStatus()]);}catch(e){$('#auth-error').textContent=e.message;}}
$('#auth-form').onsubmit=e=>{e.preventDefault();login();};$('#register').onclick=()=>login(true);$('#auth-dialog').addEventListener('cancel',e=>e.preventDefault());
$('#logout').onclick=async()=>{try{await api('/logout',{method:'POST'});location.reload();}catch(e){toast(e.message);}};
$('#new-case').onclick=()=>{navigationVersion++;stopPolling();current=null;show('welcome');$('#case-title').focus();};
$('#workspace-nav').onclick=()=>{if(current)openCase(current.id).catch(e=>toast(e.message));else show('welcome');};$('#dataset-nav').onclick=()=>{navigationVersion++;show('dataset');refreshStatus().catch(e=>toast(e.message));};
$('#load-demo').onclick=()=>{if($('#case-text').value.trim() && !confirm('用合成示例替换当前尚未保存的案情？'))return;$('#case-title').value='合成示例：办公室电脑去向';$('#case-text').value='合成软件演示，不是真实案件。失主林某报告，18时发现办公室的一台电脑不见了。证人赵某称，17时看到维修员陈某进入办公室，但没有看到陈某带走电脑。维修工单记载陈某当天到场维修打印机。监控已经损坏，没有录像。目前不清楚电脑是否由其他工作人员移走。';$('#case-text').focus();};
$('#help').onclick=()=>$('#help-dialog').showModal();
$('#case-form').onsubmit=async e=>{e.preventDefault();const b=e.submitter;b.disabled=true;try {const c=await api('/cases',{method:'POST',body:{title:$('#case-title').value,text:$('#case-text').value}});render(c);tab('overview');$('#case-form').reset();await refreshCases();}catch(err){toast(err.message);}finally{b.disabled=false;}};
$('#case-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;const form=new FormData();form.append('file',file);try {toast('正在读取材料…');const result=await api('/extract-file',{method:'POST',body:form});$('#case-text').value=result.text;if(!$('#case-title').value)$('#case-title').value=result.name;toast('材料已导入，请核对后创建案件。');}catch(err){toast(err.message);}finally{e.target.value='';}};
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>tab(b.dataset.tab));$('#analyze').onclick=startAnalysis;
$('#message-form').onsubmit=async e=>{e.preventDefault();if(!current)return;const id=current.id,text=$('#message').value;try {const c=await api('/cases/'+id+'/messages',{method:'POST',body:{text}});if(current?.id===id && !$('#workspace').hidden){$('#message').value='';render(c);await startAnalysis();}}catch(err){toast(err.message);}};
$('#export').onclick=()=>{if(current)window.location.href='/api/cases/'+current.id+'/export';};
$('#delete').onclick=async()=>{if(!current||!confirm('删除此案件及其分析记录？此操作不可恢复。'))return;try {await api('/cases/'+current.id,{method:'DELETE'});current=null;show('welcome');await refreshCases();}catch(e){toast(e.message);}};
(async()=>{try {const user=await api('/me');csrf=user.csrf;$('#username').textContent=user.name;await Promise.all([refreshCases(),refreshStatus()]);}catch(e){if(!$('#auth-dialog').open)$('#auth-dialog').showModal();}})();
