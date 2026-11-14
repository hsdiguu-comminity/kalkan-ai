"""
Kalkan Asistanı Araç Çağrıları, Güven Skoru ve İnsana Devir Birim Testleri.
"""

from app.assistant.confidence import evaluate_tool_confidence
from app.assistant.handoff import create_handoff_payload, should_trigger_handoff
from app.assistant.tools import (
    analyze_message_text,
    analyze_url_risk,
    extract_and_analyze_screenshot,
    get_assistant_tools_schema,
)


def test_tools_return_required_format():
    """Tüm araçların standart risk_level ve reasons alanlarını döndürdüğünü test eder."""
    text_res = analyze_message_text("Kargonuz teslim edilemedi, hemen tıklayın.")
    url_res = analyze_url_risk("http://sahte-kargo-takip.xyz")
    ocr_res = extract_and_analyze_screenshot("ornek_ekran.png")

    for res in (text_res, url_res, ocr_res):
        assert "risk_level" in res
        assert res["risk_level"] in {"low", "medium", "high"}
        assert "reasons" in res
        assert isinstance(res["reasons"], list)
        assert len(res["reasons"]) > 0


def test_tools_schema_structure():
    """LLM tool şemalarının geçerli fonksiyon isimlerini içerdiğini test eder."""
    schemas = get_assistant_tools_schema()
    tool_names = {item["function"]["name"] for item in schemas}
    assert tool_names == {
        "analyze_message_text",
        "analyze_url_risk",
        "extract_and_analyze_screenshot",
    }


def test_confidence_high_and_consistent():
    """Yüksek güvenli ve tutarlı araç sonuçlarında devir gerekmediğini test eder."""
    tool_results = [
        {"risk_level": "high", "confidence": 0.91},
        {"risk_level": "high", "confidence": 0.85},
    ]
    eval_result = evaluate_tool_confidence(tool_results, threshold=0.70)

    assert eval_result.is_confident is True
    assert eval_result.needs_handoff is False


def test_confidence_low_triggers_handoff():
    """Güven skoru eşiğin altında kaldığında belirsizlik tespit edildiğini test eder."""
    tool_results = [
        {"risk_level": "medium", "confidence": 0.55},
    ]
    eval_result = evaluate_tool_confidence(tool_results, threshold=0.70)

    assert eval_result.is_confident is False
    assert eval_result.needs_handoff is True


def test_confidence_conflict_triggers_handoff():
    """Modeller arası çelişki (low ve high aynı anda) olduğunda devir gerektiğini test eder."""
    tool_results = [
        {"risk_level": "low", "confidence": 0.88},
        {"risk_level": "high", "confidence": 0.82},
    ]
    eval_result = evaluate_tool_confidence(tool_results, threshold=0.70)

    assert eval_result.needs_handoff is True
    assert any("çelişkili" in r.lower() for r in eval_result.reasons)


def test_handoff_triggered_by_user_keyword():
    """Kullanıcı açıkça yetkili/canlı destek istediğinde devrin tetiklendiğini test eder."""
    confident_eval = evaluate_tool_confidence([{"risk_level": "low", "confidence": 0.95}])
    trigger, reasons = should_trigger_handoff("Beni canlı desteğe bağlar mısın?", confident_eval)

    assert trigger is True
    assert len(reasons) > 0


def test_handoff_payload_structure():
    """İnsana devir paketinin gerekli tüm alanları ve Türkçe mesajı içerdiğini test eder."""
    payload = create_handoff_payload(
        session_id="sess_123",
        user_message="Bu link güvenli mi?",
        chat_history=[{"role": "user", "content": "Merhaba"}],
        tool_results=[{"risk_level": "high", "confidence": 0.60, "reasons": ["Şüpheli alan adı"]}],
        handoff_reasons=["Düşük güven skoru"],
    )

    assert payload["handoff_required"] is True
    assert payload["risk_level"] == "high"
    assert "user_message_for_display" in payload
    assert "canlı destek" in payload["user_message_for_display"].lower()