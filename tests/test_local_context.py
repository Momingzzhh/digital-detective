"""Synthetic mechanism tests; no real source case or performance claims."""
import pytest
from app.local_context import chat_token_ids, check_context, ContextLimitError
from app import local_model
from scripts import annotate_five_dimensions_batch as batch


class ChangedDefaultTokenizer:
    def apply_chat_template(self, messages, **options):
        # transformers 5 defaults to a mapping, whose length is just two fields.
        ids = list(range(messages[0]['tokens']))
        if options.get('return_dict') is False:
            return ids
        return {'input_ids':ids, 'attention_mask':[1]*len(ids)}


def request(tokens):
    return ([{'role':'user','tokens':tokens}], object)


def test_chat_length_uses_ids_despite_changed_library_default():
    tokenizer = ChangedDefaultTokenizer()
    assert len(tokenizer.apply_chat_template(request(900)[0])) == 2
    tokens = chat_token_ids(tokenizer, request(900)[0], thinking=False)
    assert len(tokens) == 900
    check_context(tokens,100,1000)
    with pytest.raises(ContextLimitError) as caught:
        check_context(tokens,101,1000)
    assert caught.value.prompt_tokens == 900


def test_unexpected_tokenizer_shapes_fail_closed():
    class BrokenTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {'input_ids':[1,2]}
    with pytest.raises(ValueError,match='flat list'):
        chat_token_ids(BrokenTokenizer(), request(1)[0])


def test_mixed_batch_reserves_only_oversized_case_and_preserves_pairing(monkeypatch):
    calls=[]
    def complete(requests,max_tokens,progress=None):
        calls.append([r[0][0]['tokens'] for r in requests])
        return ['first','third'],{'generation_tokens':2}
    monkeypatch.setattr(batch,'complete_batch',complete)
    generated,_,reserved=batch.bounded_generation(ChangedDefaultTokenizer(),
        [request(20),request(95),request(30)],10,100)
    assert calls == [[20,30]]
    assert generated == [(0,'first',20),(2,'third',30)]
    assert [(e.index,e.prompt_tokens) for e in reserved] == [(1,95)]


def test_all_oversized_including_audit_do_not_load_model(monkeypatch):
    monkeypatch.setattr(batch,'complete_batch',lambda *a,**k:pytest.fail('Must not load GPU model'))
    generated,stats,reserved=batch.bounded_generation(ChangedDefaultTokenizer(),[request(99)],10,100)
    assert generated == [] and stats == {} and reserved[0].index == 0
    monkeypatch.setattr(local_model,'_load',lambda:pytest.fail('Empty batch must not load GPU model'))
    assert local_model.complete_batch([]) == ([],{})


def test_runtime_template_difference_reserves_case_and_continues(monkeypatch):
    calls=[]
    def complete(requests,max_tokens,progress=None):
        calls.append(len(requests))
        if len(requests)==3:
            raise ContextLimitError(95,max_tokens,100,index=1)
        return ['first','third'],{}
    monkeypatch.setattr(batch,'complete_batch',complete)
    generated,_,reserved=batch.bounded_generation(ChangedDefaultTokenizer(),
        [request(20),request(20),request(30)],10,100)
    assert calls == [3,2]
    assert [i for i,_,_ in generated] == [0,2]
    assert reserved[0].index == 1 and reserved[0].prompt_tokens == 95


def test_partial_batch_outputs_are_never_paired_silently(monkeypatch):
    monkeypatch.setattr(batch,'complete_batch',lambda *a,**k:(['only_one'],{}))
    with pytest.raises(ValueError,match='output count'):
        batch.bounded_generation(ChangedDefaultTokenizer(),[request(20),request(20)],10,100)


def test_single_generation_uses_exact_checked_ids(monkeypatch):
    from contextlib import nullcontext
    tokenizer=ChangedDefaultTokenizer();received=[]
    class Memory:
        @staticmethod
        def clear_cache():pass
    def generate(model,tokenizer,**kwargs):
        received.append(kwargs['prompt']);return 'synthetic'
    monkeypatch.setattr(local_model,'_load',lambda:(object(),tokenizer,generate,lambda **k:None,Memory))
    monkeypatch.setattr(local_model,'model_resource',lambda *a:nullcontext())
    monkeypatch.setattr(local_model,'settings',lambda:{'LOCAL_CONTEXT_TOKENS':'100'})
    assert local_model.complete(request(20)[0],max_tokens=10) == 'synthetic'
    assert received == [list(range(20))]
    with pytest.raises(ContextLimitError):local_model.complete(request(95)[0],max_tokens=10)
    assert len(received)==1
