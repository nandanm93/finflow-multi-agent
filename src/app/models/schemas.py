from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AssetPosition(BaseModel):
    ticker: str
    allocation_pct: float = Field(ge=0, le=100)
    expected_return: float
    volatility: float


class ClientProfile(BaseModel):
    name: str
    age: int
    risk_aversion: Literal["low", "moderate", "high"]
    annual_income: float
    liquid_assets: float
    current_investments: list[AssetPosition] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)


class AnalystTask(BaseModel):
    task_id: str
    question: str
    required_depth: Literal["quick", "standard", "deep"] = "standard"
    session_id: str | None = None


class EvidenceItem(BaseModel):
    source: str
    snippet: str
    confidence: float = Field(ge=0, le=1)


class AnalystReport(BaseModel):
    task_id: str
    findings: list[EvidenceItem]
    summary: str


class AdvisorResponse(BaseModel):
    message: str
    recommendation: str
    risk_notes: list[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ClientDecision(BaseModel):
    resolved: bool
    follow_up_message: str | None = None
    rationale: str = ""


class GraphState(BaseModel):
    session_id: str
    client_profile: ClientProfile
    client_message: str
    analyst_task: AnalystTask | None = None
    analyst_report: AnalystReport | None = None
    advisor_response: AdvisorResponse | None = None
