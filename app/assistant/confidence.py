"""
Güven Skoru ve Belirsizlik Değerlendirme Modülü.

Asistanın ve alt modellerin ürettiği tahminlerin güvenilirliğini ölçer;
belirsizlik veya çelişki durumunda insana devir (handoff) kararını tetikler.
"""

from dataclasses import dataclass
from typing import Any

DEFAULT_CONFIDENCE_THRESHOLD = 0.70


@dataclass
class ConfidenceEvaluation:
    """Güven değerlendirmesi sonucunu tutan veri yapısı."""

    is_confident: bool
    overall_confidence: float
    needs_handoff: bool
    reasons: list[str]


def evaluate_tool_confidence(
    tool_results: list[dict[str, Any]],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ConfidenceEvaluation:
    """
    Çağrılan araçların güven skorlarını ve risk seviyelerini karşılaştırarak
    asistanın cevabından ne kadar emin olduğunu hesaplar.
    """
    if not tool_results:
        return ConfidenceEvaluation(
            is_confident=False,
            overall_confidence=0.0,
            needs_handoff=True,
            reasons=["Analiz için hiçbir araç sonucu üretilemedi."],
        )

    confidences = [res.get("confidence", 0.0) for res in tool_results]
    risk_levels = {res.get("risk_level") for res in tool_results}
    min_confidence = min(confidences)
    avg_confidence = sum(confidences) / len(confidences)

    reasons: list[str] = []
    needs_handoff = False

    if min_confidence < threshold:
        needs_handoff = True
        reasons.append(
            f"Güven skoru ({min_confidence:.2f}), eşik değerin ({threshold:.2f}) altında."
        )

    if "low" in risk_levels and "high" in risk_levels:
        needs_handoff = True
        reasons.append(
            "Metin ve URL analiz modelleri arasında çelişkili risk seviyeleri tespit edildi."
        )

    if not needs_handoff:
        reasons.append("Tüm modeller tutarlı ve yeterli güven skoruna sahip.")

    return ConfidenceEvaluation(
        is_confident=not needs_handoff,
        overall_confidence=round(avg_confidence, 3),
        needs_handoff=needs_handoff,
        reasons=reasons,
    )
