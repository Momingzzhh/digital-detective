"""Conservative identity citation screening, not semantic certification."""


def identity_errors(proposals, sentences, known_names):
    index = {s['id']: s['text'] for s in sentences}
    errors = []
    for number, proposal in enumerate(proposals, 1):
        refs = proposal['identity_sentence_ids']
        if not refs or set(refs)-index.keys():
            errors.append(f'proposal {number}: missing or unknown identity reference')
            continue
        identity_text = '\n'.join(index[s] for s in refs)
        statement = proposal['claim']+' '+proposal['source_attribution']
        for name in sorted(set(known_names)):
            if name and name in statement and name not in identity_text:
                errors.append(f'proposal {number}: named actor lacks explicit identity citation: {name}')
    return errors
