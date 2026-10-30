"""Plant Disease Detection - Streamlit app.   Launch with:  streamlit run app.py

Flow: upload a leaf image -> the CNN predicts one of 46 classes (no Gemini involved) -> optionally ask Gemini
to explain THAT result, in the language chosen in the sidebar.
"""
from __future__ import annotations

import hashlib

import streamlit as st

from src import chat_assistant as ca
from src import predict as predict_mod
from src.config import MAX_UPLOAD_MB, GeminiSettings, load_environment
from src.image_utils import ImageValidationError, validate_upload

load_environment()
st.set_page_config(page_title="Plantdoc.AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
:root {
  --forest: #174b35;
  --forest-2: #236847;
  --leaf: #eaf4e9;
  --cream: #f7f8f2;
  --ink: #1c2b23;
  --muted: #5b6b61;
  --line: #d9e5d8;
}
.stApp { background: linear-gradient(180deg, #f3f8ef 0%, #fbfcf8 42%, #f5f8f2 100%); color: var(--ink); }
[data-testid="stHeader"] { background: rgba(247, 250, 244, .96); }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 5rem; }
h1, h2, h3, p, label, li, [data-testid="stMarkdownContainer"] { color: var(--ink) !important; }
h1 { font-size: 2.35rem !important; letter-spacing: -.04em; font-weight: 800 !important; }
h2, h3 { letter-spacing: -.02em; }
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--muted) !important; }
.hero {
  background: linear-gradient(120deg, #173f2e 0%, #286b49 70%, #3f8557 100%);
  border-radius: 24px; padding: 30px 34px; color: #fff; margin: 0 0 24px 0;
  box-shadow: 0 14px 34px rgba(23,75,53,.15);
}
.hero * { color: #fff !important; }
.hero-kicker { font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; opacity: .82; font-weight: 700; }
.hero h1 { color: #fff !important; margin: 8px 0 6px 0; font-size: 2.45rem !important; }
.hero p { color: #e8f3e9 !important; margin: 0; font-size: 1rem; }
.panel {
  background: #fff; border: 1px solid var(--line); border-radius: 20px;
  padding: 22px 24px; box-shadow: 0 8px 24px rgba(30,65,42,.055); margin-bottom: 18px;
}
.result-label { color: #557261 !important; font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 800; }
.result-value { color: #183f2d !important; font-size: 1.45rem; font-weight: 800; margin-top: 5px; }
.pill {
  display:inline-block; padding:6px 11px; border-radius:999px; background:#e9f4e9;
  color:#215b3c !important; font-size:.78rem; font-weight:700; margin:4px 5px 4px 0;
}
div[data-testid="stFileUploader"] {
  background:#fff; border:1.5px dashed #9dbda4; border-radius:18px; padding:14px 18px;
}
div[data-testid="stFileUploader"] section { background:transparent; }
div[data-testid="stFileUploader"] button, .stButton button {
  background:#1d6542 !important; color:#fff !important; border:0 !important;
  border-radius:11px !important; font-weight:700 !important; min-height:2.6rem;
}
.stButton button:hover { background:#154b31 !important; color:#fff !important; }
[data-testid="stSidebar"] { background:#edf5eb; border-right:1px solid #dce8da; }
[data-testid="stSidebar"] * { color:#20392a !important; }
[data-testid="stChatMessage"] {
  background:#fff; border:1px solid #e1e9df; border-radius:16px; padding:14px 16px; margin-bottom:10px;
}
[data-testid="stChatInput"] {
  background:#fff !important; border:1px solid #cbdcca !important; border-radius:16px !important;
}
[data-testid="stChatInput"] textarea { color:#1c2b23 !important; -webkit-text-fill-color:#1c2b23 !important; }
[data-testid="stChatInput"] textarea::placeholder { color:#718176 !important; -webkit-text-fill-color:#718176 !important; opacity:1 !important; }
[data-testid="stAlert"] { border-radius:14px; }
hr { border-color:#dce7da !important; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner="Loading the trained model ...")
def get_predictor():
    """Load the model once per server process. Returns (predictor, error_message)."""
    try:
        return predict_mod.load_predictor("models/experiment_b"), None
    except (predict_mod.ModelNotFoundError, predict_mod.ModelMismatchError) as exc:
        return None, str(exc)


@st.cache_resource
def get_assistant():
    return ca.GeminiAssistant(GeminiSettings.from_env())


def render_sidebar(assistant: ca.GeminiAssistant, predictor) -> str:
    with st.sidebar:
        st.header("Chatbot")
        label = st.selectbox(ca.LANGUAGE_DROPDOWN_LABEL, ca.LANGUAGE_LABELS,
                             index=ca.LANGUAGE_LABELS.index(ca.DEFAULT_LANGUAGE_LABEL), key="chat_language")
        if assistant.configured:
            st.caption(f"Gemini model: `{assistant.settings.model}`")
        else:
            st.warning("Gemini is not configured, so only the CNN prediction is available. "
                       "Copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and restart the app.")
        st.divider()
        if predictor is not None:
            meta = predictor.metadata
            with st.expander("About the model"):
                st.write(f"Architecture: {meta.get('architecture')} ({meta.get('weights_initialisation')})")
                st.write(f"Trained: {meta.get('created_utc')}")
                tm = meta.get("test_metrics")
                if tm:
                    st.write(f"Measured test accuracy: {tm['accuracy']:.3f} (macro F1 {tm['macro_f1']:.3f}, "
                             f"n={tm['n_test']})")
                else:
                    st.write("Test-set evaluation has not been run yet (`python -m src.evaluate`).")
                if predictor.untrained_classes:
                    st.warning("Not trained on: " + ", ".join(predictor.untrained_classes))
        st.divider()
        st.markdown(ca.DISCLAIMER)
    return label


def render_prediction(result) -> None:
    plant = result.plant
    condition = "Healthy" if result.is_healthy else result.condition
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">CNN analysis</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin:6px 0 16px 0;color:#183f2d !important;">Prediction result</h2>',
                unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown(f'<div class="result-label">Plant identified</div><div class="result-value">{plant}</div>',
                    unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="result-label">Detected condition</div><div class="result-value">{condition}</div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="result-label">Other possible matches</div>',
                unsafe_allow_html=True)
    labels = [t["display_name"] for t in result.top_k[:3]]
    if labels:
        st.markdown("".join(f'<span class="pill">{label}</span>' for label in labels),
                    unsafe_allow_html=True)
    else:
        st.caption("No additional class labels are available.")
    st.markdown('</div>', unsafe_allow_html=True)
    for warning in result.warnings:
        st.warning(warning)


def process_chat_input(assistant: ca.GeminiAssistant, language_label: str) -> None:
    """Handle the explain button / chat box / retry. Gemini is called ONLY from here, on explicit user action."""
    pending = None
    if st.button("Ask Gemini about this result", key="ask_gemini", disabled=not assistant.configured):
        pending = (ca.INITIAL_EXPLANATION_REQUEST, "explain")
    typed = st.chat_input("Ask a question about this result ...", disabled=not assistant.configured)
    if typed is not None:
        try:
            ca.clean_user_text(typed)
            pending = (typed, "question")
        except ca.EmptyMessageError:
            st.info("Please type a question first.")
    if pending:
        with st.spinner("Gemini is thinking ..."):
            try:
                ca.send_message(st.session_state, assistant, pending[0], language_label, pending[1])
            except ca.GeminiError:
                pass   # the message is stored in session state and rendered below


def render_chat(assistant: ca.GeminiAssistant, language_label: str) -> None:
    st.subheader("🌱 Ask the Plantdoc assistant")
    st.caption("Ask about symptoms, causes, prevention and crop care. Gemini explains the CNN result; it does not inspect the photo or verify the diagnosis.")
    process_chat_input(assistant, language_label)
    chat: ca.ChatSession = st.session_state[ca.STATE_CHAT]
    for m in chat.messages:
        with st.chat_message(m.role):
            if m.kind == "explain" and m.role == "user":
                st.write("*Explain this result*")
            else:
                st.markdown(m.content)
    err = st.session_state.get(ca.STATE_CHAT_ERROR)
    if err:
        st.error(err["message"])
        if st.button("Retry last message", key="retry"):
            with st.spinner("Gemini is thinking ..."):
                try:
                    ca.send_message(st.session_state, assistant, err["text"], language_label, err["kind"])
                except ca.GeminiError:
                    pass
            st.rerun()
    if chat.messages and st.button("Start new chat", key="new_chat"):
        ca.start_new_chat(st.session_state)
        st.rerun()


def main() -> None:
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">AI-powered crop health assistant</div>
      <h1>Plantdoc.AI</h1>
      <p>Understand plant leaf conditions with deep learning, then explore practical guidance with Gemini.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Analyse a leaf")
    st.write("Upload a clear photo of one leaf. The CNN checks it against the 46 classes supported by the trained model.")
    predictor, model_error = get_predictor()
    assistant = get_assistant()
    language_label = render_sidebar(assistant, predictor)

    if predictor is None:
        st.error("No trained model is available, so no prediction can be made (the app never substitutes a "
                 "random or fake model).")
        st.code(model_error or predict_mod.TRAIN_HINT)
        return

    uploaded = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"],
                                help=f"JPG or PNG, up to {MAX_UPLOAD_MB} MB")
    if uploaded is None:
        ca.clear_prediction(st.session_state)
        st.info("Upload a JPG or PNG leaf photo to begin the analysis.")
        return
    data = uploaded.getvalue()
    try:
        image = validate_upload(data)
    except ImageValidationError as exc:
        ca.clear_prediction(st.session_state)
        st.error(str(exc))
        return

    left, right = st.columns([1, 2])
    with left:
        st.image(image, caption="Your uploaded leaf photo", use_container_width=True)
    image_hash = hashlib.sha256(data).hexdigest()
    if st.session_state.get(ca.STATE_IMAGE_HASH) != image_hash:      # predict once per distinct image
        try:
            with st.spinner("Running the CNN ..."):
                result = predictor.predict_image(image)
        except predict_mod.PredictionError as exc:
            right.error(f"Prediction failed: {exc}")
            return
        ca.register_prediction(st.session_state, result, image_hash)
    with right:
        render_prediction(st.session_state[ca.STATE_PREDICTION])
    st.divider()
    render_chat(assistant, language_label)


main()
