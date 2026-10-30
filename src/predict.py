"""Inference: load the trained model + metadata and turn an image into a structured result.

The CNN is the ONLY producer of the label and confidence shown to the user. Gemini only
receives this structured result as context and can never change it.

TensorFlow/Keras is imported lazily, so the parts of this module that do not need a model
(``interpret_probabilities``) can be used - and tested - without loading TensorFlow.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from .config import (IMAGE_SIZE, MODEL_FILENAME, MODEL_METADATA_FILENAME, MODELS_DIR,
                      low_confidence_threshold, min_margin)
from .image_utils import ImageValidationError, open_rgb, prepare_batch
from .labels import ClassMapping, load_mapping

TRAIN_HINT = ("No trained model was found. Prepare the data and train one first:\n"
              "  python -m src.data_download all\n  python -m src.preprocess\n  python -m src.train")


class ModelNotFoundError(FileNotFoundError):
    """No trained 46-class model is available. The message contains training instructions."""


class ModelMismatchError(RuntimeError):
    """The model file / metadata do not match the current class mapping."""


class PredictionError(RuntimeError):
    """Inference failed for a reason that is not the user's image."""


@dataclass
class PredictionResult:
    class_index: int
    class_key: str
    plant: str
    condition: str
    display_name: str
    is_healthy: bool
    confidence: float                       # top-1 softmax probability, 0..1 (NOT a correctness guarantee)
    top_k: List[dict] = field(default_factory=list)   # [{index, key, display_name, confidence}]
    low_confidence: bool = False
    ambiguous: bool = False
    warnings: List[str] = field(default_factory=list)
    probabilities: tuple = ()               # all class probabilities, model output order

    @property
    def confidence_percent(self) -> float:
        return self.confidence * 100.0


def interpret_probabilities(probs: Sequence[float], mapping: ClassMapping, top_k: int = 3,
                            low_conf: Optional[float] = None, margin: Optional[float] = None,
                            untrained_classes: Sequence[str] = ()) -> PredictionResult:
    """Convert a probability vector into a PredictionResult (pure function, no TensorFlow)."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.shape[0] != mapping.num_classes:
        raise PredictionError(f"model returned {p.shape[0]} probabilities, expected {mapping.num_classes}")
    if not np.all(np.isfinite(p)):
        raise PredictionError("model returned non-finite probabilities")
    low_conf = low_confidence_threshold() if low_conf is None else low_conf
    margin = min_margin() if margin is None else margin
    order = np.argsort(-p)
    top = []
    for idx in order[:max(1, top_k)]:
        info = mapping.by_index(int(idx))
        top.append({"index": int(idx), "key": info.key, "display_name": info.display_name,
                    "confidence": float(p[idx])})
    best = mapping.by_index(int(order[0]))
    conf = float(p[order[0]])
    gap = conf - float(p[order[1]]) if len(order) > 1 else 1.0
    warnings: List[str] = []
    low = conf < low_conf
    ambiguous = gap < margin
    if low:
        warnings.append(f"Low model confidence ({conf * 100:.0f}%). The image may be unclear, show a plant or "
                        f"disease the model was not trained on, or be hard to classify. Treat this result with caution.")
    if ambiguous:
        second = mapping.by_index(int(order[1])).display_name
        warnings.append(f"The top two predictions are close ({conf * 100:.0f}% vs {float(p[order[1]]) * 100:.0f}% "
                        f"for '{second}'); the model cannot clearly separate them.")
    if best.key in set(untrained_classes):
        warnings.append("This class had no training images, so this prediction is unreliable.")
    return PredictionResult(
        class_index=best.index, class_key=best.key, plant=best.plant, condition=best.condition,
        display_name=best.display_name, is_healthy=best.is_healthy, confidence=conf, top_k=top,
        low_confidence=low, ambiguous=ambiguous, warnings=warnings, probabilities=tuple(float(x) for x in p))


class Predictor:
    """Loaded model + validated metadata. Create with :func:`load_predictor`."""

    def __init__(self, model, metadata: dict, mapping: ClassMapping):
        self.model = model
        self.metadata = metadata
        self.mapping = mapping
        self.image_size = int(metadata.get("input_size", IMAGE_SIZE))
        self.untrained_classes = list(metadata.get("classes_without_training_data", []))

    def predict_probabilities(self, image: Image.Image) -> np.ndarray:
        batch = prepare_batch(image, self.image_size)
        try:
            out = self.model(batch, training=False)
            probs = np.asarray(out)[0]
        except Exception as exc:  # pragma: no cover - depends on TF internals
            raise PredictionError(f"model inference failed: {exc}") from exc
        return probs

    def predict_image(self, source: Union[Image.Image, bytes, str, Path], top_k: int = 3) -> PredictionResult:
        """Predict from a PIL image, raw bytes or a path. Raises ImageValidationError for bad images."""
        img = source if isinstance(source, Image.Image) else open_rgb(source)
        probs = self.predict_probabilities(img.convert("RGB"))
        return interpret_probabilities(probs, self.mapping, top_k=top_k, untrained_classes=self.untrained_classes)


def read_metadata(models_dir: Path = MODELS_DIR) -> dict:
    path = Path(models_dir) / MODEL_METADATA_FILENAME
    if not path.is_file():
        raise ModelNotFoundError(f"{path} not found.\n{TRAIN_HINT}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelMismatchError(f"{path} is not valid JSON: {exc}") from exc


def check_metadata(metadata: dict, mapping: ClassMapping) -> None:
    """Refuse to pair a model with a label order other than the one it was trained with."""
    names = metadata.get("class_names")
    if names != mapping.keys:
        raise ModelMismatchError(
            "The model's saved class order does not match configs/class_mapping.json. The model must be "
            "retrained (or the mapping restored) - predictions would otherwise carry wrong labels.")
    if metadata.get("num_classes") != mapping.num_classes:
        raise ModelMismatchError("model metadata num_classes disagrees with the class mapping")


def load_predictor(models_dir: Path = MODELS_DIR) -> Predictor:
    """Load the trained model. Never falls back to a random/untrained model."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    models_dir = Path(models_dir)
    mapping = load_mapping()
    metadata = read_metadata(models_dir)
    check_metadata(metadata, mapping)
    model_path = models_dir / metadata.get("model_file", MODEL_FILENAME)
    if not model_path.is_file():
        raise ModelNotFoundError(f"{model_path} not found.\n{TRAIN_HINT}")
    import keras
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise ModelMismatchError(f"could not load {model_path.name}: {exc}") from exc
    out_dim = model.output_shape[-1]
    if out_dim != mapping.num_classes:
        raise ModelMismatchError(f"model has {out_dim} outputs but the taxonomy has {mapping.num_classes} classes "
                                 f"(the supplied legacy 38-class model cannot be used; retrain).")
    return Predictor(model, metadata, mapping)
