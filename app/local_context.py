"""One tokenization policy for preflight, generation, and training length gates."""

TOKEN_POLICY = 'explicit-flat-chat-token-ids-v1'


def decoder_schema(schema):
    """Bound short citation lists without expanding large nested repetitions.

    Finite repetitions such as 80 evidence items of 5,000 characters can expand
    the regex decoder into an enormous state index. The token budget still
    bounds generation, and the original Pydantic schema remains authoritative.
    Small scalar lists retain their bounds so a citation field cannot enumerate
    the entire source and consume the output budget before the audit finishes.
    """
    original = schema.model_json_schema()

    def scalar(item, seen=frozenset()):
        if '$ref' in item:
            ref = item['$ref']
            if not ref.startswith('#/$defs/') or ref in seen:
                return False
            return scalar(original.get('$defs', {}).get(ref[8:], {}), seen | {ref})
        if 'anyOf' in item:
            return all(scalar(branch, seen) for branch in item['anyOf'])
        return item.get('type') in ('string', 'integer', 'number', 'boolean', 'null')

    def visit(value):
        if isinstance(value, dict):
            maximum = value.get('maxItems')
            bounded = (value.get('type') == 'array' and type(maximum) is int
                       and 0 <= maximum <= 8
                       and (maximum <= 1 or scalar(value.get('items', {}))))
            removed = {'minLength', 'maxLength'}
            if not bounded:
                removed |= {'minItems', 'maxItems'}
            return {k: visit(v) for k, v in value.items()
                    if k not in removed}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value
    return visit(original)


def chat_token_ids(tokenizer, messages, *, generation=True, thinking=None):
    options = {'tokenize': True, 'return_dict': False, 'add_generation_prompt': generation}
    if thinking is not None:
        options['enable_thinking'] = thinking
    tokens = tokenizer.apply_chat_template(messages, **options)
    # Fail closed if a future tokenizer changes the shape again. In particular,
    # len(BatchEncoding) counts fields, not tokens.
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise ValueError('Expected a flat list of chat token IDs; length was not guessed')
    return tokens


class ContextLimitError(ValueError):
    def __init__(self, prompt_tokens, max_tokens, limit, index=0):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.limit = limit
        self.index = index
        super().__init__(f'材料超出本地上下文上限：{prompt_tokens} + {max_tokens} > {limit} tokens；未截断证据。')


def check_context(tokens, max_tokens, limit, index=0):
    if max_tokens <= 0 or limit <= 0:
        raise ValueError('Token budgets must be positive')
    if len(tokens) + max_tokens > limit:
        raise ContextLimitError(len(tokens), max_tokens, limit, index)


def partition_requests(tokenizer, requests, max_tokens, limit):
    """Keep source-order indices so oversized cases cannot poison their peers."""
    fitting = []
    reserved = []
    for index, (messages, _) in enumerate(requests):
        tokens = chat_token_ids(tokenizer, messages, thinking=False)
        try:
            check_context(tokens, max_tokens, limit, index)
        except ContextLimitError as exc:
            reserved.append(exc)
        else:
            fitting.append((index, len(tokens)))
    return fitting, reserved
