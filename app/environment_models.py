"""Research v2 curation contracts. Private models must never be agent inputs."""
from typing import Literal
from pydantic import Field, model_validator
from .models import StrictModel


class SourceSpan(StrictModel):
    field: Literal['ajjbqk'] = 'ajjbqk'
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class PrivateEvidence(StrictModel):
    id: str
    text: str = Field(min_length=1)
    kind: str
    source: SourceSpan


class ReleasedEvidence(StrictModel):
    id: str
    text: str
    kind: str


class Observation(StrictModel):
    case_token: str
    briefing: str
    released: list[ReleasedEvidence]
    remaining_actions: int = Field(ge=0)


class InvestigationAction(StrictModel):
    action: Literal['request_material'] = 'request_material'
    kind: str = Field(min_length=1)
    person: str = ''
    time: str = ''


class ReleaseRule(StrictModel):
    kind: str
    person: str = ''
    time: str = ''
    prerequisites: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)


class ActionResult(StrictModel):
    status: Literal['released', 'no_match', 'prerequisite_missing', 'budget_exhausted']
    observation: Observation


class SentenceReference(StrictModel):
    evidence_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)


class SupportedClaim(StrictModel):
    id: str
    category: Literal['event', 'actor', 'inference', 'legal_element']
    claim: str
    # Each inner list is one sufficient conjunction, outer list is alternatives.
    support_sets: list[list[str]]
    support_citations: list[list[SentenceReference]]
    counterevidence: list[str] = Field(default_factory=list)
    unknown_reason: str = ''
    quality: Literal['source_bound_unverified', 'unknown'] = 'source_bound_unverified'


class StateSupport(StrictModel):
    supported_claim_ids: list[str]
    unresolved_claim_ids: list[str]
    disputed_claim_ids: list[str]
    legal_conclusion: Literal['unresolved'] = 'unresolved'


class PrivateCase(StrictModel):
    version: Literal['interactive-curation-v1'] = 'interactive-curation-v1'
    case_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')
    group_id: str
    split: Literal['train']
    source_sha256: str
    source_fact_sha256: str
    source_row_sha256: str
    briefing: str
    initial_evidence_ids: list[str]
    evidence: list[PrivateEvidence]
    release_rules: list[ReleaseRule]
    claims: list[SupportedClaim]
    review_status: Literal['pending_human_review'] = 'pending_human_review'

    @model_validator(mode='after')
    def references(self):
        ids = {e.id for e in self.evidence}
        if len(ids) != len(self.evidence): raise ValueError('Duplicate evidence ID')
        if len({c.id for c in self.claims}) != len(self.claims): raise ValueError('Duplicate claim ID')
        refs = list(self.initial_evidence_ids)
        for e in self.evidence:
            if e.source.end <= e.source.start or e.source.end-e.source.start != len(e.source.quote):
                raise ValueError('Invalid source span')
        keys = []
        for r in self.release_rules:
            refs.extend(r.prerequisites + r.evidence_ids)
            keys.append((r.kind, r.person, r.time))
        if len(keys) != len(set(keys)): raise ValueError('Ambiguous query')
        for c in self.claims:
            if any(not s or len(s) != len(set(s)) for s in c.support_sets):
                raise ValueError('Empty/duplicate support set')
            if not c.support_sets and not c.unknown_reason: raise ValueError('Missing unknown reason')
            if len(c.support_sets) != len(c.support_citations): raise ValueError('Unbound support sets')
            index = {e.id: e for e in self.evidence}
            for support, citations in zip(c.support_sets, c.support_citations):
                if set(support) != {s.evidence_id for s in citations}:
                    raise ValueError('Support set differs from citations')
                for s in citations:
                    if s.evidence_id not in index or s.end <= s.start or index[s.evidence_id].text[s.start:s.end] != s.text:
                        raise ValueError('Support citation differs from source text')
            for s in c.support_sets: refs.extend(s)
            refs.extend(c.counterevidence)
        if set(refs) - ids: raise ValueError('Unknown private evidence reference')
        return self
