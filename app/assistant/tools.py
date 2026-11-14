"""
Kalkan Asistanı Araç Çağrıları (Tool Calling) Modülü.

Bu modül; metin sınıflandırıcı, URL risk analizi ve ekran görüntüsü (OCR)
modellerini LLM asistanının çağırabileceği standart araçlar olarak sunar.
"""

from typing import Any, Dict, List


def analyze_message_text(text: str) -> Dict[str, Any]:
    """
    Kullanıcıdan gelen mesaj metnini dolandırıcılık ihtimaline karşı analiz eder.
    Kullanıcı-8 (Yaren) tarafından geliştirilen sınıflandırıcı katmanını çağırır.
    """
    # TODO: Yaren'in predict_text(text) fonksiyonu ile bağlanacak.
    return {
        "tool_name": "text_classifier",
        "risk_level": "high",  # low | medium | high
        "confidence": 0.88,
        "category": "kargo",
        "reasons": [
            "Mesajda aciliyet ve ödeme baskısı olşturan ifadeler tespit edildi.",
            "Bilinen kargo dolandırıcılığı kalıplarıyla eşleşiyor.",
        ],
    }


def analyze_url_risk(url: str) -> Dict[str, Any]:
    """
    Mesaj içerisinde yer alan bağlantının (URL) güvenlik riskini değerlendirir.
    Kullanıcı-13 (Ayna) tarafından geliştirilen URL tahmin katmanını çağırır.
    """
    # TODO: Ayna'nın predict_url(url) fonksiyonu ile bağlanacak.
    return {
        "tool_name": "url_analyzer",
        "risk_level": "medium",
        "confidence": 0.62,
        "reasons": [
            "Alan adı resmi kurum uzantısıyla uyuşmuyor.",
            "URL içerisinde şüpheli alt alan adı (subdomain) yapısı mevcut.",
        ],
    }


def extract_and_analyze_screenshot(image_path: str) -> Dict[str, Any]:
    """
    Ekran görüntüsündeki metni OCR ile çıkarır ve sınıflandırıcıya iletir.
    Kullanıcı-10 (Meryem) tarafından geliştirilen OCR katmanını çağırır.
    """
    # TODO: Meryem'in extract_text_from_image(image_path) fonksiyonu ile bağlanacak.
    extracted_text = "Örnek çıkarılan SMS metni..."
    text_analysis = analyze_message_text(extracted_text)

    return {
        "tool_name": "ocr_analyzer",
        "extracted_text": extracted_text,
        "risk_level": text_analysis["risk_level"],
        "confidence": text_analysis["confidence"],
        "reasons": text_analysis["reasons"],
    }


def get_assistant_tools_schema() -> List[Dict[str, Any]]:
    """
    LLM ana akışının (Kullanıcı-1 / Aleyna) kullanabilmesi için araçların
    OpenAI/LangChain uyumlu JSON şema tanımlarını döndürür.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_message_text",
                "description": "Şüpheli SMS, e-posta veya mesaj metninin dolandırıcılık riskini analiz eder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Analiz edilecek mesaj metni.",
                        }
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_url_risk",
                "description": "Bir web bağlantısının (URL) oltalama veya zararlı yazılım riskini ölçer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Analiz edilecek tam URL adresi.",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_and_analyze_screenshot",
                "description": "Yüklenen ekran görüntüsündeki metni okur ve dolandırıcılık riskini analiz eder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Görsel dosyasının sunucudaki yolu.",
                        }
                    },
                    "required": ["image_path"],
                },
            },
        },
    ]