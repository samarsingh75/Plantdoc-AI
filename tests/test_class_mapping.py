import json

import pytest

from src.config import CLASS_MAPPING_PATH
from src.labels import ClassMapping, MappingError, load_mapping, normalise_label

pytestmark = pytest.mark.unit


def test_exactly_46_unique_classes(mapping):
    assert mapping.num_classes == 46
    assert len(set(mapping.keys)) == 46
    assert len({c.display_name for c in mapping.classes}) == 46


def test_group_composition(mapping):
    groups = [c.group for c in mapping.classes]
    assert groups.count("plantvillage") == 38 and groups.count("wheat") == 4 and groups.count("rice") == 4


def test_required_wheat_and_rice_classes(mapping):
    names = {c.display_name for c in mapping.classes}
    for n in ["Wheat — Healthy", "Wheat — Leaf Rust", "Wheat — Stripe Rust (Yellow Rust)", "Wheat — Powdery Mildew",
              "Rice — Healthy", "Rice — Blast", "Rice — Bacterial Leaf Blight", "Rice — Brown Spot"]:
        assert n in names


def test_indices_match_positions_and_keys(mapping):
    for i, c in enumerate(mapping.classes):
        assert c.index == i
        assert mapping.by_index(i).key == c.key
        assert mapping.index_of(c.key) == i
    assert mapping.keys == [c.key for c in mapping.classes]


def test_legacy_38_class_order_preserved(mapping):
    # indices 0-37 are the alphabetical PlantVillage order used by the supplied 38-class model
    assert mapping.keys[:38] == sorted(mapping.keys[:38])
    assert mapping.by_index(0).key == "Apple___Apple_scab" and mapping.by_index(37).key == "Tomato___healthy"


def test_out_of_range_index_rejected(mapping):
    with pytest.raises(IndexError):
        mapping.by_index(46)
    with pytest.raises(KeyError):
        mapping.by_key("Banana___Sigatoka")


def test_plantvillage_labels_map_exactly(mapping):
    for c in mapping.classes[:38]:
        assert mapping.map_source_label("plantvillage", c.key) == (c.key, "mapped")
    assert mapping.map_source_label("plantvillage", "Background_without_leaves")[0] is None
    assert mapping.map_source_label("plantvillage", "apple___apple_scab")[0] is None   # case-sensitive


@pytest.mark.parametrize("label,expected", [
    ("Brown Rust", "Wheat___Leaf_rust"), ("leaf_rust", "Wheat___Leaf_rust"),
    ("Yellow-Rust", "Wheat___Stripe_rust"), ("Stripe rust", "Wheat___Stripe_rust"),
    ("Powdery mildew", "Wheat___Powdery_mildew"), ("Healthy", "Wheat___healthy")])
def test_wheat_aliases(mapping, label, expected):
    assert mapping.map_source_label("wheat_wfd", label) == (expected, "mapped")


@pytest.mark.parametrize("label,expected", [
    ("Bacterial leaf blight", "Rice___Bacterial_leaf_blight"), ("Brown_spot", "Rice___Brown_spot"),
    ("Rice Blast", "Rice___Blast"), ("healthy", "Rice___healthy")])
def test_rice_aliases(mapping, label, expected):
    assert mapping.map_source_label("rice_mendeley", label) == (expected, "mapped")


def test_ambiguous_unsupported_and_unknown_labels_are_excluded(mapping):
    assert mapping.map_source_label("wheat_wfd", "Rust") == (None, "ambiguous")
    assert mapping.map_source_label("rice_mendeley", "Blight") == (None, "ambiguous")
    assert mapping.map_source_label("rice_mendeley", "Tungro") == (None, "unsupported")
    assert mapping.map_source_label("wheat_wfd", "Septoria") == (None, "unsupported")
    assert mapping.map_source_label("wheat_wfd", "mystery folder") == (None, "unknown")
    assert mapping.map_source_label("no_such_source", "healthy") == (None, "unknown")


def test_normalise_label():
    assert normalise_label("  Bacterial-Leaf  Blight!! ") == "bacterial_leaf_blight"


def test_fingerprint_changes_with_order(mapping):
    doc = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
    doc["classes"][0], doc["classes"][1] = doc["classes"][1], doc["classes"][0]
    for i, c in enumerate(doc["classes"]):
        c["index"] = i
    assert ClassMapping(doc).sha256() != mapping.sha256()


def test_validate_rejects_broken_mappings():
    doc = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
    bad = json.loads(json.dumps(doc))
    bad["classes"] = bad["classes"][:45]
    with pytest.raises(MappingError):
        ClassMapping(bad).validate()
    dup = json.loads(json.dumps(doc))
    dup["classes"][5]["key"] = dup["classes"][4]["key"]
    with pytest.raises(MappingError):
        ClassMapping(dup).validate()
    clash = json.loads(json.dumps(doc))
    clash["classes"][39]["source_labels"]["wheat_wfd"].append("healthy")   # alias of Wheat healthy
    with pytest.raises(MappingError):
        ClassMapping(clash).validate()


def test_loading_is_cached_and_valid():
    assert load_mapping() is load_mapping()
