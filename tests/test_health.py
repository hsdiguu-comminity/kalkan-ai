from fastapi.testclient import TestClient

from app.main import app
from app.schemas.responses import AnalysisResult, RiskLevel


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "kalkan-ai"}


def test_analysis_result_shape() -> None:
    result = AnalysisResult(
        risk_level=RiskLevel.HIGH, reasons=["Link resmi kurum adresine benzemiyor"]
    )
    assert result.model_dump()["risk_level"] == "high"
