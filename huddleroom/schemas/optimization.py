from __future__ import annotations
import uuid
from datetime import date, datetime
from pydantic import BaseModel, ConfigDict


class PatternResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    pattern_type: str
    description: str
    confidence: float
    sample_size: int
    context: dict
    created_at: datetime


class OptimizationCreate(BaseModel):
    type: str
    generated_code: str
    pattern_id: uuid.UUID | None = None


class OptimizationUpdate(BaseModel):
    generated_code: str | None = None
    status: str | None = None
    type: str | None = None


class OptimizationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    pattern_id: uuid.UUID | None
    type: str
    generated_code: str
    status: str
    error_rate: float
    fire_count: int
    created_at: datetime
    updated_at: datetime


class CostMetricResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    optimization_id: uuid.UUID | None
    date: date
    llm_calls_saved: int
    estimated_cost_saved_usd: float
    created_at: datetime
