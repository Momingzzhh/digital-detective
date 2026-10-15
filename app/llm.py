import json
import httpx
from .config import settings
from .store import reserve, settle, uncertain


class ModelError(Exception):
    pass


async def structured(system, payload, schema, purpose, max_tokens=5000):
    cfg = settings()
    key = cfg.get('DASHSCOPE_API_KEY', '')
    if not key:
        raise ModelError('请先在本机 .env 配置 DASHSCOPE_API_KEY。')
    base = cfg.get('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1').rstrip('/')
    # Keep secrets confined to the configured Aliyun endpoint.
    if base != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ModelError('当前版本限定北京百炼接口，请核对 QWEN_BASE_URL。')
    model = cfg.get('QWEN_MODEL', 'qwen3.8-flash')
    system += '\n仅输出符合以下 JSON Schema 的 JSON 对象，不要 Markdown：\n' + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    ip = float(cfg.get('INPUT_CNY_PER_MILLION', '0.8'))
    op = float(cfg.get('OUTPUT_CNY_PER_MILLION', '2.7'))
    if ip <= 0 or op <= 0:
        raise ModelError('模型单价必须为正数。')
    # UTF-8 bytes plus framing overhead is a deliberately conservative token bound.
    estimated_input = len(json.dumps(messages, ensure_ascii=False).encode()) + 2048
    amount = (estimated_input * ip + max_tokens * op) / 1_000_000
    call_id = reserve(purpose, model, amount, min(float(cfg.get('BUDGET_CNY', '500')), 500))
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(base + '/chat/completions', headers={'Authorization': 'Bearer ' + key}, json={
                'model': model, 'messages': messages, 'temperature': 0.1,
                'max_tokens': max_tokens, 'enable_thinking': False,
                'response_format': {'type': 'json_object'},
            })
        if response.status_code != 200:
            raise ModelError(f'百炼接口返回 HTTP {response.status_code}，请检查模型权限、余额和地域。')
        body = response.json()
        usage = body.get('usage', {})
        if not usage or 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
            raise ModelError('接口未返回用量，已保留预算预占，请核对账单。')
        ni, no = int(usage['prompt_tokens']), int(usage['completion_tokens'])
        settle(call_id, (ni * ip + no * op) / 1_000_000, ni, no)
        choice = body['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise ModelError('模型输出超过长度限制，请减少本轮材料。')
        return schema.model_validate_json(choice['message']['content'])
    except ModelError:
        uncertain(call_id)
        raise
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
        uncertain(call_id)
        raise ModelError('模型请求或结构化输出失败，未生成结论；可重试。') from exc
