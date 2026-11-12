"""Project top-level metadata without decoding sealed JSON values.

Only use on hash-bound, previously validated JSONL. This is not a replacement
JSON validator. Skipped text is scanned for delimiters, never deserialized.
"""
import json

ALLOWED = frozenset({'case_id', 'group_id', 'source_group_id', 'split', 'source_binding'})


def project(line, fields=ALLOWED):
    if not set(fields) <= ALLOWED:
        raise ValueError('Only identity metadata may be projected')
    decoder = json.JSONDecoder()
    i = 0
    def whitespace(pos):
        while pos < len(line) and line[pos].isspace(): pos += 1
        return pos
    i = whitespace(i)
    if line[i:i+1] != '{': raise ValueError('Expected object')
    i += 1
    result = {}; seen = set()
    while True:
        i = whitespace(i)
        if line[i:i+1] == '}': break
        key, i = decoder.raw_decode(line, i)
        if not isinstance(key, str) or key in seen: raise ValueError('Invalid/duplicate key')
        seen.add(key); i = whitespace(i)
        if line[i:i+1] != ':': raise ValueError('Expected colon')
        i = whitespace(i + 1); start = i
        depth = 0; quoted = False; escape = False
        while i < len(line):
            c = line[i]
            if quoted:
                if escape: escape = False
                elif c == '\\': escape = True
                elif c == '"': quoted = False
            elif c == '"': quoted = True
            elif c in '[{': depth += 1
            elif c in ']}':
                if depth == 0: break
                depth -= 1
            elif c == ',' and depth == 0: break
            i += 1
        if quoted or depth or i == start: raise ValueError('Unterminated value')
        if key in fields:
            value = json.loads(line[start:i])
            if not isinstance(value, str): raise ValueError('Identity must be a string')
            result[key] = value
        i = whitespace(i)
        if line[i:i+1] == '}': break
        if line[i:i+1] != ',': raise ValueError('Expected comma')
        i += 1
    if line[i+1:].strip(): raise ValueError('Trailing content')
    return result
