"""Pydantic request/response validation schemas matching contracts/API.md.

Owner: Vaidehi (Role 2).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from contracts.types import CATEGORIES

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class CategorySettingModel(BaseModel):
    enabled: bool
    priority: Literal["critical", "high", "medium", "low"]
    sensitivity: float = Field(ge=0.0, le=1.0)


class QuietHoursModel(BaseModel):
    enabled: bool
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _validate_hhmm(cls, value: str) -> str:
        if not _TIME_RE.match(value):
            raise ValueError("Time must be in HH:MM 24-hour format")
        return value


class TelegramModel(BaseModel):
    enabled: bool
    bot_token: str
    chat_id: str


class SettingsModel(BaseModel):
    categories: dict[str, CategorySettingModel]
    quiet_hours: QuietHoursModel
    alert_style: Literal["full", "compact"]
    tv_guard: bool
    cpu_saver: bool
    custom_method: Literal["embedding", "mfcc"]
    telegram: TelegramModel

    @model_validator(mode="after")
    def _validate_all_categories_present(self) -> "SettingsModel":
        keys = set(self.categories.keys())
        expected = set(CATEGORIES)
        missing = expected - keys
        extra = keys - expected
        if missing:
            raise ValueError(f"Missing categories: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unknown categories: {sorted(extra)}")
        return self


class FeedbackRequest(BaseModel):
    correct: bool


class CreateCustomSoundRequest(BaseModel):
    name: str
    category: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must not be empty")
        return cleaned

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"Unknown category: {value}")
        return value


class WhistleCounterUpdateRequest(BaseModel):
    enabled: bool
    target: int = Field(ge=1)

    @field_validator("target", mode="before")
    @classmethod
    def _reject_bool_target(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("target must be an integer >= 1")
        return value
