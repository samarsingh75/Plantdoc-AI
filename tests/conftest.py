"""Shared fixtures. Synthetic images here are TEST FIXTURES only - they are never used to train a model."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels import load_mapping  # noqa: E402
from src.predict import interpret_probabilities  # noqa: E402


def make_image(seed: int, size=(96, 80), mode="RGB") -> Image.Image:
    """A smooth random 'blob' picture; different seeds give clearly different dHashes."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.BICUBIC).convert(mode)


def to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def mapping():
    return load_mapping()


@pytest.fixture
def jpeg_bytes():
    return to_bytes(make_image(1))


def probs_for(mapping, top_index: int, top_p: float, second_index: int = 0, second_p: float = None):
    p = np.full(mapping.num_classes, 0.0)
    second_p = (1 - top_p) / 2 if second_p is None else second_p
    p[top_index] = top_p
    p[second_index if second_index != top_index else (top_index + 1) % 46] = second_p
    rest = 1.0 - p.sum()
    mask = p == 0
    p[mask] = rest / mask.sum()
    return p


class FakePredictor:
    """Duck-typed stand-in for src.predict.Predictor (no TensorFlow)."""
    def __init__(self, mapping, class_index=35, confidence=0.93):
        self.mapping = mapping
        self.metadata = {"architecture": "mobilenetv2", "weights_initialisation": "imagenet",
                         "created_utc": "test", "test_metrics": None}
        self.untrained_classes = []
        self.image_size = 224
        self.class_index, self.confidence = class_index, confidence
        self.calls = 0

    def predict_image(self, source, top_k=3):
        self.calls += 1
        return interpret_probabilities(probs_for(self.mapping, self.class_index, self.confidence), self.mapping, top_k)


@pytest.fixture
def fake_predictor(mapping):
    return FakePredictor(mapping)
