"""Offline structured generation. Cloud credentials are never read."""
import asyncio
import json
from .config import settings
from .local_model import complete


class ModelError(Exception):
    pass


class ProviderAccessError(ModelError):
    # Compatibility for archived account failure records; no provider is called.
    def __init__(self, message, code, status, reservation_id=None, request_id=None):
        super().__init__(message)
        self.code = code; self.status = status
        self.reservation_id = reservation_id; self.request_id = request_id


async def structured(system, payload, schema, purpose, max_tokens=5000, model_override=None):
    if settings().get('MODEL_BACKEND', 'local') != 'local':
        raise ModelError('外部模型API已停用，当前只支持本地离线模型。')
    if model_override:
        raise ModelError('旧脚本指定了云端模型，请改用本地五维标注流程；外部API不会被调用。')
    instructions = (system + '\n仅输出JSON数据实例，禁止Markdown和Schema定义。顶层字段只能为：' +
        ', '.join(schema.model_fields) + '\nJSON Schema：' + json.dumps(schema.model_json_schema(), ensure_ascii=False))
    messages = [{'role': 'system', 'content': instructions},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    # Legacy model_override values cannot activate a provider or reserve money.
    try:
        text = await asyncio.to_thread(complete, messages, max_tokens, schema)
        return schema.model_validate_json(text)
    except (ValueError, RuntimeError, ImportError) as exc:
        raise ModelError('本地推理未生成有效结果：' + str(exc)[:240]) from exc
