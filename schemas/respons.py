from typing import Any
from pydantic import BaseModel

class TaskPlanResponse(BaseModel):
    task_id: str
    confidence: float
    requires_image: bool
    requires_second_image: bool
    requires_gis: bool
    requires_spectral:  bool
    description: str

class AnalyzeResponse(BaseModel):
    request_id: str
    task: str
    answer:str
    evidence: dict[str, Any]
    metadata: dict[str,Any]
