"""Tüm analiz sonuçlarının ortak biçimi.

Kalkan kesin hüküm vermez: her model ve araç sonucu risk seviyesi + gerekçeler olarak döner.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisResult(BaseModel):
    risk_level: RiskLevel
    reasons: list[str] = Field(
        default_factory=list, description="Kullanıcıya gösterilecek Türkçe gerekçeler"
    )
    score: float | None = Field(
        default=None, ge=0, le=1, description="Modelin ham risk skoru (0-1)"
    )
    category: str | None = Field(
        default=None, description="ör. kargo, banka, vergi_iadesi, yatirim"
    )
