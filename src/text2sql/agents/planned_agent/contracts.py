"""Contracts for the two places where model output enters application code."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SQLPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sql: str = Field(min_length=1, max_length=20000)


class AnswerSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1, max_length=20000)
    chart_type: Literal["none", "bar", "line", "pie"] = "none"
    x_column: str = ""
    y_column: str = ""
    color_column: str = ""

    @model_validator(mode="after")
    def validate_chart_fields(self):
        if not self.answer.strip():
            raise ValueError("The answer must not be blank.")
        if self.chart_type == "none":
            if self.x_column or self.y_column or self.color_column:
                raise ValueError("A 'none' chart cannot have chart columns.")
        elif not self.x_column or not self.y_column:
            raise ValueError("A chart needs both x_column and y_column.")
        if self.chart_type == "pie" and self.color_column:
            raise ValueError("Pie charts do not support color grouping.")
        return self
