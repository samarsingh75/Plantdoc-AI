"""Headless Streamlit tests (AppTest). Gemini is always mocked; the CNN is a fake predictor."""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import FakePredictor, make_image, to_bytes
from src import chat_assistant as ca
from src import predict as pr

pytestmark = pytest.mark.integration
APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py")


class GeminiSpy:
    def __init__(self):
        self.calls = []
        self.fail = None

    def __call__(self, context, history, user_message, language):
        self.calls.append({"class": context.class_key, "history": list(history), "msg": user_message,
                           "language": language.name})
        if self.fail:
            raise self.fail
        return f"[{language.name}] answer #{len(self.calls)}"


@pytest.fixture
def env(monkeypatch, mapping):
    st.cache_resource.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    spy = GeminiSpy()
    monkeypatch.setattr(ca.GeminiAssistant, "generate_reply", spy)
    predictor = FakePredictor(mapping)
    monkeypatch.setattr(pr, "load_predictor", lambda *a, **k: predictor)
    yield spy, predictor
    st.cache_resource.clear()


def open_app(image_seed=1):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    if image_seed is not None:
        at.file_uploader[0].upload("leaf.jpg", to_bytes(make_image(image_seed, size=(120, 90))), "image/jpeg").run()
        assert not at.exception
    return at


def button(at, key):
    return next(b for b in at.button if b.key == key)


def test_language_dropdown_options_and_default(env):
    at = open_app(None)
    sb = at.sidebar.selectbox[0]
    assert sb.label == "Select Chatbot Language" and sb.value == "English"
    assert list(sb.options) == ca.LANGUAGE_LABELS and len(sb.options) == 13


def test_prediction_shown_without_calling_gemini(env):
    spy, predictor = env
    at = open_app()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Plant"] == "Tomato" and metrics["Model confidence"] == "93.0%"
    assert spy.calls == [] and predictor.calls == 1


def test_no_model_shows_training_instructions_and_no_prediction(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(pr, "load_predictor", lambda *a, **k: (_ for _ in ()).throw(pr.ModelNotFoundError("none\n" + pr.TRAIN_HINT)))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and len(at.file_uploader) == 0 and len(at.metric) == 0
    assert any("No trained model" in e.value for e in at.error)
    assert any("python -m src.train" in c.value for c in at.code)
    st.cache_resource.clear()


def test_cnn_works_without_gemini_key(env, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.config.load_dotenv", None)       # do not pick up a developer's real .env
    st.cache_resource.clear()
    at = open_app()
    assert {m.label: m.value for m in at.metric}["Plant"] == "Tomato"
    assert any("not configured" in w.value for w in at.sidebar.warning)
    assert button(at, "ask_gemini").disabled and at.chat_input[0].disabled


def test_gemini_not_called_on_reruns_or_language_change(env):
    spy, _ = env
    at = open_app()
    for _ in range(3):
        at.run()
    at.sidebar.selectbox[0].select("Hindi — हिन्दी").run()
    at.sidebar.selectbox[0].select("Tamil — தமிழ்").run()
    assert spy.calls == [] and not at.exception


def test_explain_button_uses_selected_language_and_only_one_call(env):
    spy, _ = env
    at = open_app()
    at.sidebar.selectbox[0].select("Hindi — हिन्दी").run()
    button(at, "ask_gemini").click().run()
    assert len(spy.calls) == 1 and spy.calls[0]["language"] == "Hindi" and spy.calls[0]["class"] == "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    at.run()
    at.run()
    assert len(spy.calls) == 1                                  # reruns do not re-call Gemini
    assert any("[Hindi] answer #1" in m.value for m in at.chat_message[-1].markdown)


def test_language_switch_preserves_conversation_and_applies_to_next_answer(env):
    spy, _ = env
    at = open_app()
    at.chat_input[0].set_value("How does it spread?").run()
    at.sidebar.selectbox[0].select("Telugu — తెలుగు").run()
    assert len(at.chat_message) == 2                            # conversation still on screen after switching
    at.chat_input[0].set_value("And how can I prevent it?").run()
    assert [c["language"] for c in spy.calls] == ["English", "Telugu"]
    assert len(at.chat_message) == 4
    assert [m for _, m in spy.calls[1]["history"]][0] == "How does it spread?"


def test_new_image_starts_fresh_chat_for_new_prediction(env, mapping):
    spy, predictor = env
    at = open_app(1)
    at.chat_input[0].set_value("tell me about this").run()
    assert len(at.chat_message) == 2
    predictor.class_index = 0                                    # next image is predicted as Apple scab
    at.file_uploader[0].upload("other.jpg", to_bytes(make_image(2, size=(100, 100))), "image/jpeg").run()
    assert len(at.chat_message) == 0
    at.chat_input[0].set_value("what is it?").run()
    assert spy.calls[-1]["class"] == "Apple___Apple_scab" and spy.calls[-1]["history"] == []


def test_start_new_chat_button(env):
    at = open_app()
    at.chat_input[0].set_value("hello").run()
    button(at, "new_chat").click().run()
    assert len(at.chat_message) == 0 and {m.label: m.value for m in at.metric}["Plant"] == "Tomato"


def test_gemini_failure_shows_error_keeps_prediction_and_retry_works(env):
    spy, _ = env
    spy.fail = ca.GeminiQuotaError()
    at = open_app()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert any("quota" in e.value.lower() for e in at.error)
    assert {m.label: m.value for m in at.metric}["Plant"] == "Tomato" and len(at.chat_message) == 0
    spy.fail = None
    button(at, "retry").click().run()
    assert len(at.chat_message) == 2 and not any("quota" in e.value.lower() for e in at.error)
    assert spy.calls[-1]["msg"] == "hi"


def test_corrupt_upload_is_rejected_politely(env):
    spy, predictor = env
    at = open_app(None)
    at.file_uploader[0].upload("bad.jpg", b"definitely not an image", "image/jpeg").run()
    assert not at.exception and any("could not be read" in e.value for e in at.error)
    assert predictor.calls == 0 and len(at.metric) == 0


def test_warning_shown_for_low_confidence(env):
    _, predictor = env
    predictor.confidence = 0.30
    at = open_app()
    assert any("Low model confidence" in w.value for w in at.warning)


def test_disclaimer_is_visible(env):
    at = open_app(None)
    assert any("not a guaranteed agricultural diagnosis" in m.value for m in at.sidebar.markdown)
