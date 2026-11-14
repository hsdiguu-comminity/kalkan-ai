"""
İnsana / Canlı Desteğe Devir (Human Handoff) Modülü.

Asistanın emin olamadığı veya kullanıcının özellikle yetkili talep ettiği
durumlarda konuşma bağlamını canlı destek ekibine aktarır.
"""

from datetime import UTC, datetime
from typing import Any

from app.assistant.confidence import ConfidenceEvaluation

HANDOFF_KEYWORDS = [
    "canlı destek",
    "canlı desteğe",
    "müşteri temsilcisi",
    "insanla görüş",
    "yetkili",
    "operatör",
]


def should_trigger_handoff(
    user_message: str,
    confidence_eval: ConfidenceEvaluation,
) -> tuple[bool, list[str]]:
    """
    Kullanıcı mesajındaki açık taleplere veya güven değerlendirmesine göre
    insana devir gerekip gerekmediğini belirler.
    """
    reasons: list[str] = []
    normalized_msg = user_message.lower().strip()

    if any(keyword in normalized_msg for keyword in HANDOFF_KEYWORDS):
        reasons.append("Kullanıcı açıkça canlı destek / yetkili ile görüşmek istedi.")
        return True, reasons

    if confidence_eval.needs_handoff:
        reasons.extend(confidence_eval.reasons)
        return True, reasons

    return False, []


def create_handoff_payload(
    session_id: str,
    user_message: str,
    chat_history: list[dict[str, str]],
    tool_results: list[dict[str, Any]],
    handoff_reasons: list[str],
) -> dict[str, Any]:
    """
    Canlı destek ekibinin ekranına düşecek devir paketini ve kullanıcıya
    gösterilecek bilgilendirme mesajını oluşturur.
    """
    risk_priority = {"high": 3, "medium": 2, "low": 1}
    detected_risk = "medium"
    if tool_results:
        detected_risk = max(
            (res.get("risk_level", "medium") for res in tool_results),
            key=lambda r: risk_priority.get(r, 0),
        )

    return {
        "handoff_required": True,
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "risk_level": detected_risk,
        "reasons": handoff_reasons,
        "user_message_for_display": (
            "Bu durumun kesin güvenliğinden tam olarak emin olamadım. "
            "Sizi riske atmamak adına görüşme geçmişiniz ve ön analiz notlarınızla "
            "birlikte uzman canlı destek ekibimize aktarıyorum. Lütfen bekleyin."
        ),
        "agent_context": {
            "last_user_message": user_message,
            "recent_history": chat_history[-5:],
            "tool_outputs": tool_results,
        },
    }
