from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DayInput(BaseModel):
    """One day of conditions for which daily bike rentals are predicted."""

    model_config = ConfigDict(extra="forbid")

    dteday: Optional[date] = Field(default=None, description="Optional ISO date (YYYY-MM-DD), used only for logging")
    season: Literal[1, 2, 3, 4]
    yr: Literal[0, 1, 2]
    mnth: int = Field(ge=1, le=12)
    holiday: Literal[0, 1]
    weekday: int = Field(ge=0, le=6)
    workingday: Literal[0, 1]
    weathersit: Literal[1, 2, 3, 4]
    temp: float = Field(ge=0.0, le=1.0)
    atemp: float = Field(ge=0.0, le=1.0)
    hum: float = Field(ge=0.0, le=1.0)
    windspeed: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def holiday_cannot_be_working_day(self):
        if self.holiday == 1 and self.workingday == 1:
            raise ValueError("holiday=1 and workingday=1 is contradictory")
        return self


class PredictionResponse(BaseModel):
    predicted_rentals: float


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    days: list[DayInput] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
