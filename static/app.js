'use strict';
const $ = s => document.querySelector(s);
let csrf = '', current = null, polling = null, statusData = null;
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
function show(view) { for(const id of ['welcome','workspace','dataset']) $('#'+id).hidden=id!==view; }
function tab(name) { document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('selected',b.dataset.tab===name);b.setAttribute('aria-selected',b.dataset.tab===name);}); for(const id of ['overview','evidence','graph','conclusion']) $('#tab-'+id).hidden=id!==name; }
async function refreshCases() {
  const cases=await api('/cases'); $('#case-list').replaceChildren(); $('#case-count').textContent=cases.length;
  for(const c of cases) { const b=el('button','◌  '+c.title,'case-link'+(current?.id===c.id?' active':'')); b.onclick=()=>openCase(c.id).catch(e=>toast(e.message)); $('#case-list').append(b); }
}
async function refreshStatus() {
  statusData=await api('/status'); $('#model-label').textContent=statusData.model; $('#model-dot').classList.toggle('ready',statusData.configured);
  const d=$('#dataset-content');d.replaceChildren(el('h2','数据状态'));
  const ds=statusData.dataset;
  for(const text of [ds.status||'数据核验中',`符合完整验收条件的案件：${ds.qualified||0}`,`目标：至少 1000 个真实案件，关键证据全部可追溯`, `本机模型调用已记账：¥${statusData.budget.cost.toFixed(3)}；预占：¥${statusData.budget.reserved.toFixed(3)}；上限：¥${statusData.limit}`]) d.append(el('p',text));
  if(ds.candidates) d.append(el('p',`原始去重候选：${ds.candidates}；候选数不等于合格数。`));
  d.append(el('p','账本只统计本项目通过此实例发出的请求。云账号的其他调用及价格变化请在百炼控制台核对。'));
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
  for(const [title,key] of [['证据依据','findings'],['关键经过','reconstruction']]) {const s=section(node,title);if(!a[key].length)s.append(el('p','尚无足够依据。'));for(const f of a[key]) {const p=el('p',f.claim);refs(p,f.evidence_ids);s.append(p);}}
  const conflicts=section(node,'矛盾与冲突');if(!a.conflicts.length)conflicts.append(el('p','当前未识别出明确矛盾；不代表材料已被证实。'));for(const f of a.conflicts) {const p=el('p',`${f.severity} · ${f.description}`);refs(p,f.evidence_ids);conflicts.append(p);}
  const hypotheses=section(node,'竞争假设');for(const h of a.hypotheses) {hypotheses.append(el('p',h.description));const p=el('p','支持：');refs(p,h.supporting);p.append(el('span',' 反证：'));refs(p,h.opposing);hypotheses.append(p);}
  const gaps=section(node,'需要继续核验');for(const s of a.missing_information)gaps.append(el('p','• '+s));
  const review=section(node,'独立复核');review.append(el('p',c.review?.supported?'复核模型未发现明确的无依据结论；仍需研究人员复核。':'复核未通过，已将结论降为证据不足。'));for(const issue of [...(c.review?.unsupported_claims||[]),...(c.review?.unresolved_issues||[])])review.append(el('p','• '+issue));
  const confidence=section(node,'置信度变化');confidence.append(el('p',`当前 ${(a.confidence*100).toFixed(0)}% · 模型主观估计，未经校准，不代表实测准确率。`,'confidence'));const bars=el('div',undefined,'history-bar');for(const p of c.confidence_history||[]) {const bar=el('div');bar.style.height=(p.value*65)+'px';bar.title=`第 ${p.round} 轮：${Math.round(p.value*100)}%`;bars.append(bar);}confidence.append(bars);
  const trace=section(node,'工具执行记录');for(const t of c.trace||[])trace.append(el('p',`${t.action} · ${t.summary}`));
}
function drawGraph(graph) {
  const svg=$('#evidence-graph');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
  const make=(tag,attrs)=>{const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);return n;};
  if(!graph?.nodes.length){$('#graph-detail').textContent='提取证据后，显示证据、人物实体与分析结论之间的关系。';return;}
  const groups=['evidence','entity','finding'].map(k=>graph.nodes.filter(n=>n.kind===k));
  const height=Math.max(560,...groups.map(g=>g.length*62+50));svg.setAttribute('viewBox',`0 0 760 ${height}`);svg.style.height=Math.min(height,1200)+'px';
  const positions={};groups.forEach((g,j)=>g.forEach((n,i)=>positions[n.id]={x:100+j*280,y:40+i*62}));
  for(const e of graph.edges) {const a=positions[e.source],b=positions[e.target];if(!a||!b)continue;svg.append(make('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#d5e0d3','stroke-width':1.3}));}
  for(const n of graph.nodes) {const p=positions[n.id];const g=make('g',{tabindex:0,role:'button','aria-label':n.label});const fill={evidence:'#7fa68a',entity:'#c3ac76',finding:'#8c9fb7'}[n.kind];g.append(make('circle',{cx:p.x,cy:p.y,r:8,fill}));const text=make('text',{x:p.x,y:p.y+24,'text-anchor':'middle'});text.textContent=n.label.slice(0,16);g.append(text);const select=()=>$('#graph-detail').textContent=n.label+'\n'+n.detail;g.onclick=select;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' ')select();};svg.append(g);}
}
function renderJob(job) { const active=job.state==='running';$('#analyze').disabled=active;$('#message-form button').disabled=active;$('#delete').disabled=active;$('#analyze').textContent=active?'分析进行中…':current?.analysis?'根据现有材料重新分析':'开始分析现有材料';const n=$('#job-status');n.hidden=!['running','error'].includes(job.state);n.classList.toggle('error',job.state==='error');n.textContent=job.state==='error'?job.error:'◌ '+job.stage+'…'; }
async function openCase(id) {clearInterval(polling);const c=await api('/cases/'+id);render(c);await refreshCases();if(c.job.state==='running')startPoll(id);}
function startPoll(id) {clearInterval(polling);polling=setInterval(async()=>{try {const c=await api('/cases/'+id);if(current?.id!==id){clearInterval(polling);return;}renderJob(c.job);if(c.job.state!=='running'){clearInterval(polling);render(c);await refreshStatus();if(c.job.state==='completed')tab('conclusion');}}catch(e){clearInterval(polling);toast(e.message);}},1500);}
async function startAnalysis() {if(!current)return;try {const job=await api('/cases/'+current.id+'/analyze',{method:'POST'});renderJob(job);startPoll(current.id);}catch(e){toast(e.message);}}
async function login(register=false) {const name=$('#auth-name').value,password=$('#auth-password').value;if(!$('#auth-form').reportValidity())return;$('#auth-error').textContent='';try {if(register)await api('/register',{method:'POST',body:{name,password}});const data=await api('/login',{method:'POST',body:{name,password}});csrf=data.csrf;$('#username').textContent=data.name;$('#auth-dialog').close();$('#auth-password').value='';await Promise.all([refreshCases(),refreshStatus()]);}catch(e){$('#auth-error').textContent=e.message;}}
$('#auth-form').onsubmit=e=>{e.preventDefault();login();};$('#register').onclick=()=>login(true);$('#auth-dialog').addEventListener('cancel',e=>e.preventDefault());
$('#logout').onclick=async()=>{try{await api('/logout',{method:'POST'});location.reload();}catch(e){toast(e.message);}};
$('#new-case').onclick=()=>{clearInterval(polling);current=null;show('welcome');$('#case-title').focus();};
$('#workspace-nav').onclick=()=>show(current?'workspace':'welcome');$('#dataset-nav').onclick=()=>{show('dataset');refreshStatus().catch(e=>toast(e.message));};
$('#help').onclick=()=>$('#help-dialog').showModal();
$('#case-form').onsubmit=async e=>{e.preventDefault();const b=e.submitter;b.disabled=true;try {const c=await api('/cases',{method:'POST',body:{title:$('#case-title').value,text:$('#case-text').value}});render(c);tab('overview');$('#case-form').reset();await refreshCases();}catch(err){toast(err.message);}finally{b.disabled=false;}};
$('#case-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;const form=new FormData();form.append('file',file);try {toast('正在读取材料…');const result=await api('/extract-file',{method:'POST',body:form});$('#case-text').value=result.text;if(!$('#case-title').value)$('#case-title').value=result.name;toast('材料已导入，请核对后创建案件。');}catch(err){toast(err.message);}finally{e.target.value='';}};
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>tab(b.dataset.tab));$('#analyze').onclick=startAnalysis;
$('#message-form').onsubmit=async e=>{e.preventDefault();if(!current)return;const text=$('#message').value;try {const c=await api('/cases/'+current.id+'/messages',{method:'POST',body:{text}});$('#message').value='';render(c);await startAnalysis();}catch(err){toast(err.message);}};
$('#export').onclick=()=>{if(current)window.location.href='/api/cases/'+current.id+'/export';};
$('#delete').onclick=async()=>{if(!current||!confirm('删除此案件及其分析记录？此操作不可恢复。'))return;try {await api('/cases/'+current.id,{method:'DELETE'});current=null;show('welcome');await refreshCases();}catch(e){toast(e.message);}};
(async()=>{try {const user=await api('/me');csrf=user.csrf;$('#username').textContent=user.name;await Promise.all([refreshCases(),refreshStatus()]);}catch(e){if(!$('#auth-dialog').open)$('#auth-dialog').showModal();}})();
