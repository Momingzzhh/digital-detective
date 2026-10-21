import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import agent, llm
from app.models import Conclusion, Extraction, Decision, Review
from app.store import initialize, reserve, settle, budget_status


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RUNTIME_DIR', str(tmp_path))
    monkeypatch.setenv('DASHSCOPE_API_KEY', '')
    monkeypatch.setenv('LOCAL_MODEL_PATH',str(tmp_path/'missing-local-model'))
    with TestClient(app) as c:
        yield c


def account(c, name='研究甲'):
    value={'name': name, 'password': 'long-test-password'}
    assert c.post('/api/register',json=value).status_code == 200
    r=c.post('/api/login',json=value)
    assert r.status_code == 200
    return {'X-CSRF-Token':r.json()['csrf']}


def test_auth_csrf_and_case_isolation(client):
    assert client.get('/api/cases').status_code == 401
    h=account(client)
    value={'title':'研究案', 'text':'证人称在21点看到甲进入大楼。'}
    assert client.post('/api/cases', json=value).status_code == 403
    r=client.post('/api/cases',json=value,headers=h)
    assert r.status_code == 200
    cid=r.json()['id']
    assert client.post('/api/cases/'+cid+'/analyze',headers=h).status_code == 503
    assert client.get('/api/cases/'+cid+'/export').status_code == 200
    account(client,'研究乙')
    assert client.get('/api/cases/'+cid).status_code == 404
    assert client.get('/api/cases').json() == []


def test_upload_and_origin(client):
    h=account(client)
    assert client.post('/api/extract-file',headers=h,files={'file':('案.txt','这是现场证据材料。'.encode(),'text/plain')}).json()['text']=='这是现场证据材料。'
    assert client.post('/api/extract-file',headers=h,files={'file':('a.exe',b'12345')}).status_code==415
    assert client.post('/api/extract-file',headers=h,files={'file':('a.docx',b'not a zip')}).status_code==422
    assert client.post('/api/cases',headers={**h,'Origin':'https://evil.example'},json={'title':'x','text':'12345'}).status_code==403


def test_budget_reserves_before_request(client):
    x=reserve('test','model',400,500)
    with pytest.raises(ValueError):reserve('test','model',101,500)
    settle(x,2,100,100)
    assert budget_status()['cost']==2
    assert budget_status()['reserved']==0


def test_offline_model_does_not_reserve_or_spend_cloud_budget(tmp_path,monkeypatch):
    monkeypatch.setenv('RUNTIME_DIR',str(tmp_path));initialize()
    monkeypatch.setattr(llm,'settings',lambda:{'MODEL_BACKEND':'local','DASHSCOPE_API_KEY':'synthetic-test-key'})
    calls=[]
    def complete(messages,max_tokens,schema):
        calls.append(messages)
        return json.dumps({'supported':True,'unsupported_claims':[],'unresolved_issues':[]})
    monkeypatch.setattr(llm,'complete',complete)
    before=budget_status()
    result=asyncio.run(llm.structured('test',{},Review,'test',100))
    assert result.supported and len(calls)==1
    assert 'synthetic-test-key' not in str(calls)
    assert budget_status()==before


def test_cloud_backend_and_legacy_model_override_cannot_make_requests(monkeypatch):
    def forbidden(*args):raise AssertionError('Must not call any model')
    monkeypatch.setattr(llm,'complete',forbidden)
    monkeypatch.setattr(llm,'settings',lambda:{'MODEL_BACKEND':'cloud'})
    with pytest.raises(llm.ModelError):asyncio.run(llm.structured('test',{},Review,'test'))
    monkeypatch.setattr(llm,'settings',lambda:{'MODEL_BACKEND':'local'})
    with pytest.raises(llm.ModelError):asyncio.run(llm.structured('test',{},Review,'test',model_override='qwen3.8-max'))


def sample_result():
    return Conclusion(status='暂定结论', summary='甲涉嫌盗窃。', charges=['盗窃罪'],perpetrators=['甲'],
                      reconstruction=[],findings=[],conflicts=[],hypotheses=[],questions=['有监控吗？'],missing_information=[],confidence=.95)


def test_exact_quotes_reject_hallucinations(monkeypatch):
    async def fake(*args,**kwargs):
        return Extraction(evidence=[{'quote':'甲进入大楼','kind':'证言','summary':'进入','entities':['甲']},
                                    {'quote':'DNA匹配甲','kind':'DNA','summary':'无依据','entities':['甲']}])
    monkeypatch.setattr(agent,'structured',fake)
    ev,rejected=asyncio.run(agent.extract({'id':'s1','text':'证人称甲进入大楼。'}))
    assert len(ev)==1 and len(rejected)==1
    assert ev[0]['start']==3


def test_unknown_citations_and_unsupported_findings_rejected():
    r=sample_result()
    assert agent.citations_valid(r,[])


def test_independent_review_downgrades_conclusion(monkeypatch):
    async def fake(system,payload,schema,*args):
        if schema is Decision:return Decision(action='finish',purpose='检查结束')
        if schema is Conclusion:return sample_result()
        if schema is Review:return Review(supported=False,unsupported_claims=['没有盗窃证据'],unresolved_issues=[])
        raise AssertionError(schema)
    monkeypatch.setattr(agent,'structured',fake)
    c=asyncio.run(agent.analyze({'sources':[],'evidence':[],'messages':[]}))
    assert c['analysis']['status']=='证据不足'
    assert c['analysis']['perpetrators']==[] and c['analysis']['charges']==[]
    assert c['analysis']['confidence']<=.49


def test_source_bound_tool_context_preserves_quotes_without_copying_them(monkeypatch):
    evidence=[{'id':'E1','kind':'证言','quote':'唯一原文：证人称没有看清面部。',
               'summary':'展示摘要不作为第二来源','entities':[]}]
    seen=[]
    async def fake(system,payload,schema,*args):
        if schema is Decision:return Decision(action='inspect_evidence',evidence_ids=['E1'],purpose='核对否定陈述')
        if schema is Conclusion:
            seen.append(json.loads(json.dumps(payload)))
            return sample_result()
        if schema is Review:return Review(supported=False,unsupported_claims=[],unresolved_issues=[])
        raise AssertionError(schema)
    monkeypatch.setattr(agent,'structured',fake)
    asyncio.run(agent.analyze({'sources':[],'evidence':evidence,
        'research':{'source_only':True},'messages':[{'role':'user','text':evidence[0]['quote']}]}))
    payload=seen[0]
    assert json.dumps(payload,ensure_ascii=False).count(evidence[0]['quote'])==1
    assert payload['evidence'][0]['text']==evidence[0]['quote']
    assert all(o['evidence_ids']==['E1'] for o in payload['observations'])


def test_no_secret_in_status(client):
    account(client)
    s=client.get('/api/status').text
    assert 'API_KEY' not in s
    assert client.get('/.env').status_code==404
    assert client.get('/data/raw/lecard/candidates1.zip').status_code==404


def test_delete_and_message(client):
    h=account(client)
    c=client.post('/api/cases',headers=h,json={'title':'test','text':'原始材料没有指纹。'}).json()
    r=client.post('/api/cases/'+c['id']+'/messages',headers=h,json={'text':'监控设备已损坏。'})
    assert len(r.json()['sources'])==2
    assert client.delete('/api/cases/'+c['id'],headers=h).status_code==200
    assert client.get('/api/cases/'+c['id']).status_code==404
