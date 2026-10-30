import json

import numpy as np
import pytest

from conftest import make_image, probs_for
from src import predict as pr
from src.labels import load_mapping

# ------------------------------------------------------------------ unit (no TensorFlow)


@pytest.mark.unit
def test_prediction_has_46_classes_and_correct_labels(mapping):
    res = pr.interpret_probabilities(probs_for(mapping, 40, 0.9), mapping)
    assert len(res.probabilities) == 46
    assert res.class_index == 40 and res.class_key == mapping.by_index(40).key == "Wheat___Stripe_rust"
    assert res.plant == "Wheat" and not res.is_healthy
    assert res.top_k[0]["index"] == 40 and len(res.top_k) == 3
    assert res.confidence == pytest.approx(0.9)


@pytest.mark.unit
def test_every_output_index_maps_to_its_label(mapping):
    for i in range(46):
        p = np.zeros(46)
        p[i] = 1.0
        assert pr.interpret_probabilities(p, mapping).class_key == mapping.keys[i]


@pytest.mark.unit
def test_wrong_probability_length_is_an_error(mapping):
    with pytest.raises(pr.PredictionError):
        pr.interpret_probabilities(np.ones(38) / 38, mapping)
    with pytest.raises(pr.PredictionError):
        pr.interpret_probabilities(np.full(46, np.nan), mapping)


@pytest.mark.unit
def test_low_confidence_is_flagged(mapping):
    res = pr.interpret_probabilities(probs_for(mapping, 3, 0.35, second_index=4, second_p=0.30), mapping)
    assert res.low_confidence and res.ambiguous
    assert any("Low model confidence" in w for w in res.warnings)
    assert any("close" in w for w in res.warnings)


@pytest.mark.unit
def test_confident_prediction_has_no_warnings(mapping):
    res = pr.interpret_probabilities(probs_for(mapping, 3, 0.97, second_p=0.01), mapping)
    assert not res.low_confidence and not res.ambiguous and res.warnings == []


@pytest.mark.unit
def test_thresholds_are_configurable(mapping, monkeypatch):
    p = probs_for(mapping, 3, 0.7, second_p=0.05)
    assert not pr.interpret_probabilities(p, mapping).low_confidence
    monkeypatch.setenv("LOW_CONFIDENCE_THRESHOLD", "0.9")
    assert pr.interpret_probabilities(p, mapping).low_confidence


@pytest.mark.unit
def test_untrained_class_prediction_is_warned(mapping):
    res = pr.interpret_probabilities(probs_for(mapping, 43, 0.9), mapping, untrained_classes=["Rice___Blast"])
    assert any("no training images" in w for w in res.warnings)


@pytest.mark.unit
def test_missing_model_gives_training_instructions_not_a_fake_model(tmp_path):
    with pytest.raises(pr.ModelNotFoundError) as e:
        pr.load_predictor(tmp_path)
    assert "python -m src.train" in str(e.value)


@pytest.mark.unit
def test_metadata_with_different_class_order_is_rejected(mapping):
    meta = {"class_names": list(reversed(mapping.keys)), "num_classes": 46}
    with pytest.raises(pr.ModelMismatchError):
        pr.check_metadata(meta, mapping)
    pr.check_metadata({"class_names": mapping.keys, "num_classes": 46}, mapping)


@pytest.mark.unit
def test_legacy_38_class_model_is_refused(tmp_path):
    (tmp_path / "model_metadata.json").write_text(json.dumps({"class_names": load_mapping().keys[:38], "num_classes": 38}))
    with pytest.raises(pr.ModelMismatchError):
        pr.load_predictor(tmp_path)


# ------------------------------------------------------------------ integration (TensorFlow)
@pytest.mark.integration
def test_model_has_46_outputs_and_inference_matches_training_preprocessing(tmp_path, mapping):
    """Builds a random (untrained) MobileNetV2, saves it with metadata, reloads through load_predictor and checks that
    (a) output size is 46, (b) probabilities sum to 1, (c) the training-side decode path and the inference-side PIL
    path yield (almost) the same array."""
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from PIL import Image
    from src.image_utils import open_rgb, resize_for_model, to_model_array
    from src.model import build_model

    model = build_model(46, 224, "mobilenetv2", weights=None)
    assert model.output_shape == (None, 46)
    model.save(tmp_path / "plant_disease_model.keras")
    (tmp_path / "model_metadata.json").write_text(json.dumps({
        "model_file": "plant_disease_model.keras", "input_size": 224, "num_classes": 46,
        "class_names": mapping.keys, "classes_without_training_data": []}))
    predictor = pr.load_predictor(tmp_path)
    img = make_image(11, size=(300, 200))
    res = predictor.predict_image(img)
    assert len(res.probabilities) == 46 and sum(res.probabilities) == pytest.approx(1.0, abs=1e-4)
    assert res.class_key in mapping.keys

    # dataset preparation writes a 224x224 JPEG with PIL; training decodes it with tf.io; inference uses PIL directly
    path = tmp_path / "prepared.jpg"
    resize_for_model(open_rgb(__import__("conftest").to_bytes(img)), 224).save(path, format="JPEG", quality=95)
    tf_arr = tf.cast(tf.io.decode_jpeg(tf.io.read_file(str(path)), channels=3, dct_method="INTEGER_ACCURATE"),
                     tf.float32).numpy()
    pil_arr = to_model_array(Image.open(path).convert("RGB"))
    assert tf_arr.shape == pil_arr.shape == (224, 224, 3)
    assert np.abs(tf_arr - pil_arr).max() <= 4.0      # decoder rounding only
