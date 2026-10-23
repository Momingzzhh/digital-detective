"""One tokenization policy for preflight, generation, and training length gates."""

TOKEN_POLICY = 'explicit-flat-chat-token-ids-v1'


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
