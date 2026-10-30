"""Central configuration: paths, defaults and environment handling.

Nothing in this module imports TensorFlow, so it is cheap to import everywhere
(tests, data scripts, the Streamlit app start-up).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is a declared dependency, but never let a missing .env crash us
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- directory layout (see README) -----------------------------------------
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

CLASS_MAPPING_PATH = CONFIG_DIR / "class_mapping.json"
DATASET_SOURCES_PATH = CONFIG_DIR / "dataset_sources.json"
TRAIN_CONFIG_PATH = CONFIG_DIR / "train_config.json"

MODEL_FILENAME = "plant_disease_model.keras"
MODEL_METADATA_FILENAME = "model_metadata.json"

# --- model / preprocessing defaults ------------------------------------------
IMAGE_SIZE = 224          # square model input (pixels)
SEED = 42                 # global deterministic seed
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_UPLOAD_MB = 10
MAX_IMAGE_PIXELS = 50_000_000   # decompression-bomb guard

# --- prediction warnings (configurable through the environment) ---------------
DEFAULT_LOW_CONFIDENCE = 0.60   # top-1 softmax below this => "low confidence" warning
DEFAULT_MIN_MARGIN = 0.15       # top-1 minus top-2 below this => "ambiguous" warning

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def load_environment() -> None:
    """Load variables from a local ``.env`` file (if present) without overriding
    variables that are already set in the real environment."""
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int
    max_output_tokens: int

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "GeminiSettings":
        """Read settings from the environment. The key is never logged or printed."""
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        # Ignore the placeholder from .env.example so it doesn't count as a real key.
        if key.lower() in {"your_api_key_here", "changeme", "none", "null"}:
            key = ""
        return cls(
            api_key=key,
            model=os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL,
            timeout_seconds=_float_env("GEMINI_TIMEOUT_SECONDS", 30.0),
            max_retries=max(0, min(_int_env("GEMINI_MAX_RETRIES", 2), 4)),
            max_output_tokens=_int_env("GEMINI_MAX_OUTPUT_TOKENS", 1500),
        )


def low_confidence_threshold() -> float:
    return _float_env("LOW_CONFIDENCE_THRESHOLD", DEFAULT_LOW_CONFIDENCE)


def min_margin() -> float:
    return _float_env("MIN_TOP2_MARGIN", DEFAULT_MIN_MARGIN)
