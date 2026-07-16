import json
from typing import Literal

from pydantic import BaseModel, Field

QueryType = Literal["factual", "analytical", "opinion"]
ConfidenceLevel = Literal["high", "medium", "low"]

EvidenceStore = dict[str, str]


class RouterDecision(BaseModel):
    query_type: QueryType
    confidence: ConfidenceLevel
    reasoning: str


class Source(BaseModel):
    title: str
    url: str | None
    relevance: str


class ResearchReport(BaseModel):
    query: str
    query_type: QueryType
    summary: str
    key_findings: list[str]
    sources: list[Source]
    confidence: ConfidenceLevel
    limitations: str
    generated_at: str


class EvaluationResult(BaseModel):
    completeness: int = Field(ge=1, le=10)
    accuracy: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    sourcing: int = Field(ge=1, le=10)
    feedback: str

    @property
    def average(self) -> float:
        return (self.completeness + self.accuracy + self.clarity + self.sourcing) / 4

    @property
    def passed(self) -> bool:
        return self.average >= 8
