"""Common shared Pydantic models and error responses."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standardized JSON error response without exposing stack traces or server paths."""

    error: str = Field(..., description="Error classification code")
    message: str = Field(..., description="Human-readable safe error message")
    experiment_id: Optional[str] = Field(None, description="Associated experiment ID if available")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = "ok"
    version: str = "0.1.0"
    service: str = "fcHMRF-LIS Neuroimaging Backend"
    storage_accessible: bool = True
