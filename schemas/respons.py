from typing import Any

from pydantic import BaseModel, Field


class TaskPlanResponse(BaseModel):
    task: str
    confidence: float
    reasoning: str
    required_inputs: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    requires_image: bool = False
    requires_multitemporal: bool = False
    requires_gis: bool = False
    requires_spectral: bool = False


class AnalysisResponse(BaseModel):
    success: bool
    request_id: str
    query: str
    answer: str
    task: str
    model: str
    processing_time: float
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
