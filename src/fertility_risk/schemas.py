"""Typed input and output contracts shared by the API, dashboard, and agent."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class AgentRequest(BaseModel):
    """A constrained request for the single-agent research assistant."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["assess_risk", "compare_scenarios", "methodology"]
    record: PredictionRequest | None = None
    comparison_record: PredictionRequest | None = None
    question: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_intent_inputs(self):
        if self.intent == "assess_risk" and self.record is None:
            raise ValueError("record is required for assess_risk")
        if self.intent == "compare_scenarios" and (
            self.record is None or self.comparison_record is None
        ):
            raise ValueError(
                "record and comparison_record are required for compare_scenarios"
            )
        return self


class AgentToolCall(BaseModel):
    tool: str
    status: Literal["completed"] = "completed"


class AgentResponse(BaseModel):
    intent: str
    answer: str
    result: dict
    tool_calls: list[AgentToolCall]
    model_version: str
    demo_only: bool
    disclaimer: str
    report: dict


class AgentChatRequest(BaseModel):
    """Natural-language request plus optional structured scenario context."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=3, max_length=500)
    record: PredictionRequest | None = None
    comparison_record: PredictionRequest | None = None


class AgentPlan(BaseModel):
    intent: Literal["assess_risk", "compare_scenarios", "methodology"]
    confidence: float = Field(ge=0, le=1)
    reason: str
    required_tools: list[str]
    provider: Literal["deterministic", "openai"] = "deterministic"


class AgentChatResponse(BaseModel):
    plan: AgentPlan
    response: AgentResponse
