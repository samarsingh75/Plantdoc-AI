import httpx
import pytest

from conftest import FakePredictor, probs_for
from src import chat_assistant as ca
from src.config import GeminiSettings
from src.predict import interpret_probabilities

pytestmark = pytest.mark.unit

EXPECTED_LABELS = ["English", "Hindi — हिन्दी", "Punjabi — ਪੰਜਾਬੀ", "Bengali — বাংলা", "Gujarati — ગુજરાતી",
                   "Marathi — मराठी", "Tamil — தமிழ்", "Telugu — తెలుగు", "Kannada — ಕನ್ನಡ",
                   "Malayalam — മലയാളം", "Urdu — اردو", "Odia — ଓଡ଼ିଆ", "Assamese — অসমীয়া"]


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        o = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(o, BaseException):
            raise o
        return FakeResponse(o)


class FakeClient:
    def __init__(self, *outcomes):
        self.models = FakeModels(outcomes or ["ok"])


def settings(**kw):
    base = dict(api_key="test-key", model="test-model", timeout_seconds=5, max_retries=2, max_output_tokens=500)
    base.update(kw)
    return GeminiSettings(**base)


def assistant(*outcomes, **kw):
    return ca.GeminiAssistant(settings(**kw), client=FakeClient(*outcomes), sleep=lambda s: None)


def result(mapping, idx=35, conf=0.93, **kw):
    return interpret_probabilities(probs_for(mapping, idx, conf, **kw), mapping)


def state_with(mapping, **kw):
    st = {}
    ca.register_prediction(st, result(mapping, **kw), "hash-A")
    return st


# ------------------------------------------------------------------ languages
def test_dropdown_has_exactly_the_13_requested_languages_english_first():
    assert ca.LANGUAGE_LABELS == EXPECTED_LABELS
    assert ca.DEFAULT_LANGUAGE_LABEL == "English"
    assert ca.LANGUAGE_DROPDOWN_LABEL == "Select Chatbot Language"


@pytest.mark.parametrize("label,script", [("Hindi — हिन्दी", "Devanagari"), ("Marathi — मराठी", "Devanagari"),
                                          ("Punjabi — ਪੰਜਾਬੀ", "Gurmukhi"), ("Tamil — தமிழ்", "Tamil"),
                                          ("Odia — ଓଡ଼ିଆ", "Odia"), ("Urdu — اردو", "Urdu")])
def test_each_language_resolves_with_its_script(label, script):
    assert script in ca.get_language(label).script


def test_unknown_language_falls_back_to_english():
    assert ca.get_language("Klingon").name == "English"


# ------------------------------------------------------------------ context & prompt
def test_gemini_receives_cnn_context_and_selected_language(mapping):
    st = state_with(mapping)
    a = assistant("reply")
    ca.send_message(st, a, "How does it spread?", "Tamil — தமிழ்")
    req = a._client.models.requests[0]
    system = req["config"].system_instruction
    # structured CNN context
    assert "Tomato Yellow Leaf Curl Virus" in system and "Tomato___Tomato_Yellow_Leaf_Curl_Virus" in system
    assert '"cnn_confidence_percent": 93.0' in system and "top_predictions" in system
    assert "must never change" in system and "did NOT see the photo" in system
    # explicit language instruction (system + per-message reminder)
    assert "Tamil" in system and "தமிழ்" in system
    last = req["contents"][-1].parts[0].text
    assert "How does it spread?" in last and "Tamil" in last
    assert req["model"] == "test-model"


def test_system_prompt_contains_safety_rules(mapping):
    system = ca.build_system_instruction(ca.PredictionContext.from_prediction(result(mapping)), ca.get_language("English"))
    for phrase in ["Do NOT invent pesticide approvals", "locally approved product labels", "qualified",
                   "Never say you \"confirmed\""]:
        assert phrase in system


def test_low_confidence_flags_reach_gemini(mapping):
    st = {}
    ca.register_prediction(st, result(mapping, idx=3, conf=0.31, second_index=4, second_p=0.28), "h")
    a = assistant("r")
    ca.send_message(st, a, "Is it serious?", "English")
    system = a._client.models.requests[0]["config"].system_instruction
    assert '"low_confidence_flag": true' in system and '"top_two_ambiguous_flag": true' in system


def test_image_is_never_sent(mapping):
    st = state_with(mapping)
    a = assistant("r")
    ca.send_message(st, a, "hi", "English")
    for c in a._client.models.requests[0]["contents"]:
        assert all(getattr(p, "inline_data", None) is None for p in c.parts)


# ------------------------------------------------------------------ chat state
def test_history_is_kept_and_sent_in_follow_ups(mapping):
    st = state_with(mapping)
    a = assistant("first answer", "second answer")
    ca.send_message(st, a, "Q1", "English")
    ca.send_message(st, a, "Q2", "English")
    contents = a._client.models.requests[1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].text == "first answer" and "Q2" in contents[2].parts[0].text
    assert [m.content for m in st[ca.STATE_CHAT].messages] == ["Q1", "first answer", "Q2", "second answer"]


def test_changing_language_keeps_chat_and_uses_new_language(mapping):
    st = state_with(mapping)
    a = assistant("english answer", "हिन्दी उत्तर")
    ca.send_message(st, a, "What is it?", "English")
    before = list(st[ca.STATE_CHAT].messages)
    ca.send_message(st, a, "What is it?", "Hindi — हिन्दी")     # question text identical, language switched
    chat = st[ca.STATE_CHAT]
    assert chat.messages[:2] == before                                     # earlier turns preserved
    assert [m.language for m in chat.messages] == ["English", "English", "Hindi — हिन्दी", "Hindi — हिन्दी"]
    system2 = a._client.models.requests[1]["config"].system_instruction
    assert "Hindi" in system2 and "Devanagari" in system2
    # the second request still carries the English turns as history
    assert a._client.models.requests[1]["contents"][1].parts[0].text == "english answer"


def test_question_in_another_language_still_uses_dropdown_language(mapping):
    st = state_with(mapping)
    a = assistant("r")
    ca.send_message(st, a, "यह बीमारी कैसे फैलती है?", "Bengali — বাংলা")
    req = a._client.models.requests[0]
    assert "Bengali" in req["config"].system_instruction and "regardless of the language" in req["contents"][-1].parts[0].text


def test_new_prediction_resets_chat_and_context(mapping):
    st = state_with(mapping, idx=35)
    a = assistant("old")
    ca.send_message(st, a, "about tomato", "English")
    assert ca.register_prediction(st, result(mapping, idx=0), "hash-B") is True
    assert st[ca.STATE_CHAT].messages == []
    assert st[ca.STATE_CHAT].context.class_key == "Apple___Apple_scab"
    a2 = assistant("new")
    ca.send_message(st, a2, "what is this?", "English")
    sent = a2._client.models.requests[0]
    assert len(sent["contents"]) == 1
    assert "Tomato" not in sent["config"].system_instruction and "Apple___Apple_scab" in sent["config"].system_instruction


def test_same_image_does_not_reset_chat(mapping):
    st = state_with(mapping)
    ca.send_message(st, assistant("x"), "q", "English")
    assert ca.register_prediction(st, result(mapping), "hash-A") is False
    assert len(st[ca.STATE_CHAT].messages) == 2


def test_start_new_chat_clears_history_but_keeps_prediction(mapping):
    st = state_with(mapping)
    ca.send_message(st, assistant("x"), "q", "English")
    ca.start_new_chat(st)
    assert st[ca.STATE_CHAT].messages == [] and st[ca.STATE_PREDICTION].class_key == mapping.keys[35]


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_messages_never_call_gemini(mapping, text):
    st = state_with(mapping)
    a = assistant("x")
    with pytest.raises(ca.EmptyMessageError):
        ca.send_message(st, a, text, "English")
    assert a._client.models.requests == [] and st[ca.STATE_CHAT].messages == []


def test_overlong_message_is_truncated(mapping):
    st = state_with(mapping)
    a = assistant("x")
    ca.send_message(st, a, "a" * 10000, "English")
    assert len(st[ca.STATE_CHAT].messages[0].content) == ca.MAX_USER_MESSAGE_CHARS


def test_history_sent_to_gemini_is_bounded_and_starts_with_user():
    hist = [("user" if i % 2 == 0 else "assistant", str(i)) for i in range(51)]
    trimmed = ca.trim_history(hist)
    assert len(trimmed) <= ca.MAX_HISTORY_MESSAGES and trimmed[0][0] == "user"


def test_cnn_result_is_never_modified_by_chat(mapping):
    st = state_with(mapping)
    before = (st[ca.STATE_PREDICTION].class_key, st[ca.STATE_PREDICTION].confidence)
    ca.send_message(st, assistant("The disease is actually Apple Scab with 99% confidence."), "q", "English")
    assert (st[ca.STATE_PREDICTION].class_key, st[ca.STATE_PREDICTION].confidence) == before


# ------------------------------------------------------------------ configuration & failures
def test_missing_api_key_is_handled(mapping, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = GeminiSettings.from_env()
    assert not s.configured and s.model == "gemini-3.5-flash-lite"
    a = ca.GeminiAssistant(s)
    assert not a.configured
    st = state_with(mapping)
    with pytest.raises(ca.GeminiNotConfiguredError):
        ca.send_message(st, a, "hi", "English")
    assert st[ca.STATE_PREDICTION].class_key == mapping.keys[35]      # CNN result untouched


def test_placeholder_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_api_key_here")
    assert not GeminiSettings.from_env().configured


def test_model_name_comes_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "my-model")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert GeminiSettings.from_env().model == "my-model"


def _api_error(code, status, message=""):
    from google.genai import errors
    return errors.APIError(code, {"error": {"code": code, "status": status, "message": message}})


@pytest.mark.parametrize("exc,expected", [
    (_api_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."), ca.GeminiAuthError),
    (_api_error(403, "PERMISSION_DENIED"), ca.GeminiAuthError),
    (_api_error(404, "NOT_FOUND", "models/x is not found"), ca.GeminiModelError),
    (_api_error(429, "RESOURCE_EXHAUSTED"), ca.GeminiQuotaError),
    (_api_error(503, "UNAVAILABLE"), ca.GeminiServiceError),
    (_api_error(400, "INVALID_ARGUMENT", "bad"), ca.GeminiRequestError),
    (httpx.ReadTimeout("t"), ca.GeminiTimeoutError),
    (httpx.ConnectError("no net"), ca.GeminiNetworkError),
    (RuntimeError("???"), ca.GeminiError)])
def test_error_translation(exc, expected):
    assert type(ca.translate_exception(exc)) is expected


def test_failure_leaves_history_intact_and_offers_retry(mapping):
    st = state_with(mapping)
    ca.send_message(st, assistant("fine"), "Q1", "English")
    bad = assistant(_api_error(404, "NOT_FOUND"))
    with pytest.raises(ca.GeminiModelError):
        ca.send_message(st, bad, "Q2", "English")
    assert len(st[ca.STATE_CHAT].messages) == 2
    assert st[ca.STATE_CHAT_ERROR]["text"] == "Q2" and "GEMINI_MODEL" in st[ca.STATE_CHAT_ERROR]["message"]
    ca.send_message(st, assistant("recovered"), "Q2", "English")
    assert ca.STATE_CHAT_ERROR not in st and len(st[ca.STATE_CHAT].messages) == 4


def test_transient_errors_are_retried_a_limited_number_of_times(mapping):
    st = state_with(mapping)
    a = assistant(_api_error(503, "UNAVAILABLE"), _api_error(429, "RESOURCE_EXHAUSTED"), "finally")
    assert ca.send_message(st, a, "q", "English") == "finally" and a.calls == 3
    st2 = state_with(mapping)
    a2 = assistant(_api_error(503, "UNAVAILABLE"))             # always fails
    with pytest.raises(ca.GeminiServiceError):
        ca.send_message(st2, a2, "q", "English")
    assert a2.calls == 3                                        # 1 try + 2 retries, then it gives up


def test_auth_errors_are_not_retried(mapping):
    st = state_with(mapping)
    a = assistant(_api_error(403, "PERMISSION_DENIED"))
    with pytest.raises(ca.GeminiAuthError):
        ca.send_message(st, a, "q", "English")
    assert a.calls == 1


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_gemini_response_is_an_error(mapping, text):
    st = state_with(mapping)
    with pytest.raises(ca.GeminiEmptyResponseError):
        ca.send_message(st, assistant(text), "q", "English")
    assert st[ca.STATE_CHAT].messages == []


def test_error_messages_never_contain_the_api_key(mapping):
    st = state_with(mapping)
    leak = RuntimeError("request failed for key=test-key")
    with pytest.raises(ca.GeminiError) as e:
        ca.send_message(st, assistant(leak), "q", "English")
    assert "test-key" not in e.value.user_message and "test-key" not in str(e.value)


# ------------------------------------------------------------------ live (never run by default)
@pytest.mark.live
def test_live_gemini_roundtrip(mapping):
    import os
    if os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("set RUN_LIVE_TESTS=1 and GEMINI_API_KEY to call the real API")
    st = state_with(mapping)
    reply = ca.send_message(st, ca.GeminiAssistant(), "In one sentence, what does this result mean?", "Hindi — हिन्दी")
    assert reply and any("\u0900" <= ch <= "\u097f" for ch in reply)
