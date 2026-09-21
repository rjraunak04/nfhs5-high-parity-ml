"""Typed input and output contracts shared by the API and dashboard."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Five harmonised DHS-style inputs used by the public research demo."""

    model_config = ConfigDict(extra="forbid")

    current_age: int = Field(ge=15, le=49, examples=[31])
    residence: int = Field(ge=1, le=2, description="1=urban, 2=rural")
    education: int = Field(ge=0, le=3, description="0=none, 1=primary, 2=secondary, 3=higher")
    wealth: int = Field(ge=1, le=5, description="DHS wealth quintile, 1=poorest to 5=richest")
    in_union: int = Field(ge=0, le=1, description="1=married/living together, otherwise 0")


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[PredictionRequest] = Field(min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    probability: float = Field(ge=0, le=1)
    classification: str
    threshold: float = Field(ge=0, le=1)
    model_version: str
    demo_only: bool


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    demo_only: bool
