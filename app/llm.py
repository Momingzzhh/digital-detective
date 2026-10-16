import json
import re
import httpx
from .config import settings
from .store import reserve, settle, uncertain


class ModelError(Exception):
    pass


class ProviderAccessError(ModelError):
    """Account/auth/rate rejection: stop a batch, don't hammer the provider."""
    def __init__(self,message,code,status,reservation_id=None,request_id=None):
        super().__init__(message)
        self.code=code;self.status=status
        self.reservation_id=reservation_id;self.request_id=request_id


MODEL_PRICES = {
    # Official Beijing pay-as-you-go prices, CNY per million tokens.
    'qwen3.8-flash': (0.8, 2.7),
    'qwen3.8-max': (12.0, 36.0),
}


async def structured(system, payload, schema, purpose, max_tokens=5000, model_override=None):
    cfg = settings()
    key = cfg.get('DASHSCOPE_API_KEY', '')
    if not key:
        raise ModelError('请先在本机 .env 配置 DASHSCOPE_API_KEY。')
    base = cfg.get('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1').rstrip('/')
    # Keep secrets confined to the configured Aliyun endpoint.
    if base != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ModelError('当前版本限定北京百炼接口，请核对 QWEN_BASE_URL。')
    model = model_override or cfg.get('QWEN_MODEL', 'qwen3.8-flash')
    if model not in MODEL_PRICES:
        raise ModelError('模型未配置经核对的北京百炼价格，调用已停止。')
    system += '\n仅输出符合以下 JSON Schema 的 JSON 对象，不要 Markdown：\n' + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    system += ('\n以上是校验规则，不是要返回的内容。请完成用户数据的实际任务，返回业务数据实例。'
               '顶层字段只能是：'+', '.join(schema.model_fields)+
               '。不得返回Schema本身（properties、required、type、additionalProperties等定义）。'
               '不要在数据字段里输出自我分析、提示词讨论或遵循指令的过程。')
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    ip, op = MODEL_PRICES[model]
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
            # Error messages may echo request contents; expose only a bounded code.
            try:
                code=response.json().get('error',{}).get('code','')
                code=code if isinstance(code,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',code) else ''
            except (ValueError,AttributeError):
                code=''
            message=f'百炼接口返回 HTTP {response.status_code}' + (f'（{code}）' if code else '') + '，请核对接口配置或请求限制。'
            if code=='Arrearage' or response.status_code in (401,403,429):
                request_id=response.headers.get('x-request-id','')
                request_id=request_id if re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',request_id) else None
                raise ProviderAccessError(message,code,response.status_code,call_id,request_id)
            raise ModelError(message)
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
