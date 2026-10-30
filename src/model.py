"""CNN definition: transfer learning with MobileNetV2 (default) or EfficientNetB0.

The network accepts RAW pixels (float32, 0-255). Input scaling lives *inside* the model
(MobileNetV2: x/127.5 - 1; EfficientNetB0 has its own built-in rescaling), so training,
evaluation and inference all feed the same kind of array and cannot drift apart.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # keep the console readable

import keras  # noqa: E402
from keras import layers  # noqa: E402

from .config import IMAGE_SIZE  # noqa: E402
from .labels import EXPECTED_NUM_CLASSES  # noqa: E402

ARCHITECTURES = ("mobilenetv2", "efficientnetb0")


class WeightsUnavailableError(RuntimeError):
    """Pretrained ImageNet weights could not be loaded (usually: no internet)."""


def build_model(num_classes: int = EXPECTED_NUM_CLASSES, image_size: int = IMAGE_SIZE,
                architecture: str = "mobilenetv2", weights: str | None = "imagenet",
                dropout: float = 0.3) -> keras.Model:
    """Return an *uncompiled* classifier. The pretrained backbone is the layer named 'backbone'.

    weights='imagenet' loads pretrained weights (downloaded once by Keras); weights=None gives a
    randomly initialised backbone (only sensible for tests or when you accept much lower accuracy).
    """
    architecture = architecture.lower()
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    shape = (image_size, image_size, 3)
    try:
        if architecture == "mobilenetv2":
            backbone = keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights=weights)
        else:
            backbone = keras.applications.EfficientNetB0(input_shape=shape, include_top=False, weights=weights)
    except Exception as exc:  # download / network / cache problems
        if weights is not None:
            raise WeightsUnavailableError(
                f"Could not load pretrained '{weights}' weights for {architecture}: {exc}. "
                f"Check your internet connection and retry, or pass --weights none to train from random "
                f"initialisation (expect clearly lower accuracy).") from exc
        raise
    backbone.name = "backbone"
    backbone.trainable = False

    inputs = keras.Input(shape=shape, name="image")
    x = inputs
    if architecture == "mobilenetv2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_to_minus1_1")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"plant_disease_{architecture}")


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )
    return model


def set_finetune(model: keras.Model, n_layers: int) -> int:
    """Unfreeze the last ``n_layers`` layers of the backbone (BatchNorm layers stay frozen,
    which keeps fine-tuning stable with small batches). Returns the number of trainable layers."""
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    cut = max(0, len(backbone.layers) - n_layers)
    trainable = 0
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cut and not isinstance(layer, layers.BatchNormalization)
        trainable += int(layer.trainable)
    return trainable
