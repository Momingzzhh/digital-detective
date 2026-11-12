import json
import pytest
from pydantic import ValidationError
from app.environment_models import (PrivateCase, PrivateEvidence, SourceSpan, ReleaseRule,
    SupportedClaim, InvestigationAction, Observation)
from app.interactive_curation import observe, act, state_support, reachability, PUBLIC_BRIEFING
from app.sealed_metadata import project


def synthetic_case():
    """Invented fixture, never mixed into genuine training material."""
    texts = ['甲称看见乙。', '视频显示有人。', '甲更正先前说法。']
    refs = [{'evidence_id': f'E{i}', 'start': 0, 'end': len(t), 'text': t} for i,t in enumerate(texts,1)]
    return PrivateCase(case_id='synthetic-only', group_id='synthetic-group', split='train',
        source_sha256='fixture', source_fact_sha256='fixture', source_row_sha256='fixture',
        briefing=PUBLIC_BRIEFING, initial_evidence_ids=[],
        evidence=[PrivateEvidence(id=f'E{i}', text=t, kind='证言',
                  source=SourceSpan(start=0, end=len(t), quote=t))
                  for i, t in enumerate(texts, 1)],
        release_rules=[ReleaseRule(kind='证言', evidence_ids=['E1']),
            ReleaseRule(kind='视听记录', person='乙', time='21:00', prerequisites=['E1'], evidence_ids=['E2']),
            ReleaseRule(kind='书面记录', evidence_ids=['E3'])],
        claims=[SupportedClaim(id='claim', category='event', claim='合成支持关系',
                               support_sets=[['E1'], ['E2']], support_citations=[[refs[0]],[refs[1]]], counterevidence=['E3']),
                SupportedClaim(id='conjunction', category='event', claim='合成联合关系', support_sets=[['E1','E2']],
                               support_citations=[[refs[0],refs[1]]]),
                SupportedClaim(id='unknown', category='legal_element', claim='未知',
                               support_sets=[], support_citations=[], quality='unknown', unknown_reason='未审查')])


def test_projection_never_decodes_skipped_values(monkeypatch):
    import app.sealed_metadata as m
    original = json.loads; decoded = []
    def tracked(s): decoded.append(s); return original(s)
    monkeypatch.setattr(m.json, 'loads', tracked)
    line = '{"raw_record":{"text":"sealed \\\" [ , }","nested":[1,{"a":2}]},"split":"test","case_id":"x"}'
    assert project(line) == {'split':'test','case_id':'x'}
    assert decoded == ['"test"','"x"']
    with pytest.raises(ValueError): project(line, {'raw_record'})
    with pytest.raises(ValueError): project('{"split":"train","split":"test"}')


def test_no_unreleased_ids_or_private_rules_and_strict_schema():
    case = synthetic_case()
    value = observe(case, [], 8).model_dump()
    assert value['released'] == []
    encoded = json.dumps(value, ensure_ascii=False)
    assert all(s not in encoded for s in ('E1','E2','E3','support_sets','release_rules','synthetic-group'))
    with pytest.raises(ValidationError): Observation.model_validate(value | {'hidden_gold': {}})
    raw = case.model_dump(); raw['split'] = 'dev'
    with pytest.raises(ValidationError): PrivateCase.model_validate(raw)


def test_wrong_filters_prerequisites_and_budget():
    case = synthetic_case()
    query = InvestigationAction(kind='视听记录', person='乙', time='21:00')
    assert act(case, [], 8, query).status == 'prerequisite_missing'
    assert act(case, ['E1'], 8, query).observation.released[-1].id == 'E2'
    for changed in ({'person':'丙'}, {'time':'22:00'}, {'kind':'不存在'}):
        result = act(case, ['E1'], 8, query.model_copy(update=changed))
        assert result.status == 'no_match'
        assert result.observation.remaining_actions == 7
        assert [e.id for e in result.observation.released] == ['E1']
    assert act(case, [], 0, query).status == 'budget_exhausted'


def test_alternative_support_decisive_removal_counterevidence():
    case = synthetic_case()
    assert state_support(case, []).supported_claim_ids == []
    assert state_support(case, ['E1']).supported_claim_ids == ['claim']
    assert state_support(case, ['E2']).supported_claim_ids == ['claim']
    assert state_support(case, ['E1','E2']).supported_claim_ids == ['claim','conjunction']
    assert state_support(case, ['E1','E3']).disputed_claim_ids == ['claim']
    assert state_support(case, ['E1','E2','E3']).legal_conclusion == 'unresolved'
    assert reachability(case)['unreachable'] == []
    case.release_rules[0].prerequisites = ['E2']
    assert reachability(case)['unreachable'] == ['E1','E2']


def test_reject_dangling_empty_and_ambiguous_rules():
    case = synthetic_case().model_dump()
    case['claims'][0]['support_sets'] = [[]]
    with pytest.raises(ValidationError): PrivateCase.model_validate(case)
    case = synthetic_case().model_dump(); case['claims'][0]['support_citations'][0][0]['text'] = '被改写'
    with pytest.raises(ValidationError): PrivateCase.model_validate(case)
    case = synthetic_case().model_dump(); case['release_rules'][0]['evidence_ids'] = ['MISSING']
    with pytest.raises(ValidationError): PrivateCase.model_validate(case)
    case = synthetic_case().model_dump(); case['release_rules'].append(case['release_rules'][0])
    with pytest.raises(ValidationError): PrivateCase.model_validate(case)
