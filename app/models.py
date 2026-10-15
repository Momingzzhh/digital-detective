from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class EvidenceDraft(StrictModel):
    quote: str = Field(min_length=1, max_length=5000)
    kind: str
    summary: str
    entities: list[str] = Field(default_factory=list)
    reliability: Literal['待核验', '存在争议', '有相互印证'] = '待核验'


class Extraction(StrictModel):
    evidence: list[EvidenceDraft] = Field(max_length=80)


class Finding(StrictModel):
    claim: str
    evidence_ids: list[str]


class Conflict(StrictModel):
    description: str
    evidence_ids: list[str]
    severity: Literal['轻微', '中等', '严重']


class Hypothesis(StrictModel):
    description: str
    supporting: list[str]
    opposing: list[str]


class Conclusion(StrictModel):
    status: Literal['暂定结论', '证据不足']
    summary: str
    charges: list[str]
    perpetrators: list[str]
    reconstruction: list[Finding]
    findings: list[Finding]
    conflicts: list[Conflict]
    hypotheses: list[Hypothesis]
    questions: list[str] = Field(max_length=5)
    missing_information: list[str]
    confidence: float = Field(ge=0, le=1)


class Review(StrictModel):
    supported: bool
    unsupported_claims: list[str]
    unresolved_issues: list[str]


class Decision(StrictModel):
    action: Literal['search_evidence', 'inspect_evidence', 'check_conflicts', 'finish']
    query: str = ''
    evidence_ids: list[str] = Field(default_factory=list)
    purpose: str
