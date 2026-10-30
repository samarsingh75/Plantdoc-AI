import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import make_image, to_bytes
from src import data_download as dd
from src import dataset_validation as dv
from src import preprocess as pp
from src.image_utils import (ImageValidationError, open_rgb, prepare_batch, to_model_array, validate_upload)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- image utilities
def test_model_array_shape_dtype_and_range():
    arr = to_model_array(make_image(3, size=(300, 120)))
    assert arr.shape == (224, 224, 3) and arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 255.0 and arr.max() > 1.0   # raw 0-255, not pre-scaled
    assert prepare_batch(make_image(3)).shape == (1, 224, 224, 3)


def test_png_with_alpha_and_grayscale_are_converted():
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    assert open_rgb(to_bytes(rgba, "PNG")).getpixel((0, 0)) == (255, 255, 255)   # transparent -> white
    assert open_rgb(to_bytes(make_image(1, mode="L"), "PNG")).mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\xff\xd8\xff\xe0 truncated"])
def test_invalid_uploads_raise_friendly_error(data):
    with pytest.raises(ImageValidationError):
        validate_upload(data)


def test_truncated_jpeg_is_rejected(jpeg_bytes):
    with pytest.raises(ImageValidationError):
        validate_upload(jpeg_bytes[: len(jpeg_bytes) // 3])


def test_too_small_and_too_large_rejected():
    with pytest.raises(ImageValidationError):
        validate_upload(to_bytes(Image.new("RGB", (8, 8))))
    with pytest.raises(ImageValidationError):
        validate_upload(b"x" * (11 * 1024 * 1024))


# ---------------------------------------------------------------- hashing / clustering
def test_dhash_flip_and_rotation_are_near_duplicates():
    base = make_image(7, size=(128, 128))
    h0 = pp.dhash_variants(base)
    flipped = pp.dhash_variants(base.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    rotated = pp.dhash_variants(base.transpose(Image.Transpose.ROTATE_90))
    other = pp.dhash_variants(make_image(8, size=(128, 128)))
    assert min(pp.hamming(flipped[0], v) for v in h0) == 0
    assert min(pp.hamming(rotated[0], v) for v in h0) == 0
    assert min(pp.hamming(other[0], v) for v in h0) > 40


def test_clustering_groups_variants_not_strangers():
    imgs = [make_image(s, size=(128, 128)) for s in (1, 2, 3)]
    imgs += [imgs[0].transpose(Image.Transpose.FLIP_LEFT_RIGHT), imgs[1].transpose(Image.Transpose.ROTATE_180)]
    groups, refused = pp.cluster_near_duplicates([pp.dhash_variants(i) for i in imgs], threshold=10)
    assert refused == 0
    assert groups[0] == groups[3] and groups[1] == groups[4]
    assert len({groups[0], groups[1], groups[2]}) == 3


def test_exact_duplicates_and_label_conflicts():
    def rec(path, cls, sha):
        return pp.Record("s", Path(path), path, "lab", cls, sha256=sha, status="ok")
    recs = [rec("a", "X", "h1"), rec("b", "X", "h1"), rec("c", "X", "h2"), rec("d", "Y", "h3"), rec("e", "Z", "h3")]
    kept, dropped = pp.remove_exact_duplicates(recs)
    assert [r.rel_path for r in kept] == ["a", "c"]
    reasons = {d["path"]: d["reason"] for d in dropped}
    assert reasons == {"b": "exact_duplicate", "d": "label_conflict", "e": "label_conflict"}


# ---------------------------------------------------------------- splitting
def _records(n_per_class=40, classes=("A", "B", "C"), group_size=1):
    recs, g = [], 0
    for c in classes:
        for i in range(n_per_class):
            r = pp.Record("s", Path(f"{c}{i}"), f"{c}{i}", c, c, status="ok", sha256=f"{c}{i}")
            r.group = g // group_size
            g += 1
            recs.append(r)
    return recs


def test_split_is_stratified_close_to_70_15_15_and_deterministic():
    r1, r2 = _records(), _records()
    c1 = pp.assign_splits(r1, 0.15, 0.15, seed=42)
    pp.assign_splits(r2, 0.15, 0.15, seed=42)
    assert [r.split for r in r1] == [r.split for r in r2]
    for c in "ABC":
        assert c1["val"][c] == 6 and c1["test"][c] == 6 and c1["train"][c] == 28
    r3 = _records()
    pp.assign_splits(r3, 0.15, 0.15, seed=7)
    assert [r.split for r in r3] != [r.split for r in r1]


def test_split_never_separates_a_group():
    recs = _records(n_per_class=60, group_size=3)
    pp.assign_splits(recs, 0.15, 0.15, seed=1)
    by_group = {}
    for r in recs:
        by_group.setdefault(r.group, set()).add(r.split)
    assert all(len(s) == 1 for s in by_group.values())


def test_tiny_classes_all_go_to_train():
    recs = _records(n_per_class=2, classes=("A",))
    pp.assign_splits(recs, 0.15, 0.15, seed=1)
    assert {r.split for r in recs} == {"train"}


# ---------------------------------------------------------------- discovery / corrupt files / end-to-end
def _make_raw(tmp_path):
    raw = tmp_path / "raw"
    pv = raw / "plantvillage"
    for ci, cls in enumerate(["Apple___Apple_scab", "Tomato___healthy"]):
        for i in range(24):
            (pv / cls).mkdir(parents=True, exist_ok=True)
            (pv / cls / f"img{i}.jpg").write_bytes(to_bytes(make_image(100 * ci + i, size=(64, 64))))
    (pv / "Apple___Apple_scab" / "dup.jpg").write_bytes((pv / "Apple___Apple_scab" / "img0.jpg").read_bytes())
    (pv / "Apple___Apple_scab" / "broken.jpg").write_bytes(b"\xff\xd8\xff garbage")
    (pv / "Unknown_Plant___rot").mkdir()
    (pv / "Unknown_Plant___rot" / "x.jpg").write_bytes(to_bytes(make_image(999)))
    (pv / "grayscale" / "Apple___Apple_scab").mkdir(parents=True)
    (pv / "grayscale" / "Apple___Apple_scab" / "g.jpg").write_bytes(to_bytes(make_image(5)))
    # unconfirmed wheat source must be ignored by default
    (raw / "wheat_wfd" / "Brown Rust").mkdir(parents=True)
    (raw / "wheat_wfd" / "Brown Rust" / "w.jpg").write_bytes(to_bytes(make_image(321)))
    return raw


def _cfg(tmp_path, raw, **kw):
    return pp.PrepConfig(raw_dir=raw, interim_dir=tmp_path / "interim", processed_dir=tmp_path / "processed",
                         reports_dir=tmp_path / "reports", workers=2, image_size=64, **kw)


def test_pipeline_end_to_end_on_synthetic_data(tmp_path):
    raw = _make_raw(tmp_path)
    report = pp.run(_cfg(tmp_path, raw), log=lambda *_: None)
    t = report["totals"]
    assert t["usable_images"] == 48                   # 2 classes x 24 (dup removed, corrupt removed)
    assert t["corrupt_or_unreadable"] == 1 and t["exact_duplicates_removed"] == 1
    excluded = {(e["source_id"], e["source_label"]): e["reason"] for e in report["excluded_labels"]}
    assert excluded[("plantvillage", "Unknown_Plant___rot")] == "unknown"
    assert "wheat_wfd" in {k[0] for k in excluded}   # unconfirmed source excluded
    assert not any(r["total"] for r in report["classes"] if r["class_key"].startswith("Wheat"))
    assert any("MISSING CLASSES" in w for w in report["warnings"])
    assert (tmp_path / "reports" / "dataset_report.md").is_file() and (tmp_path / "reports" / "class_distribution.png").is_file()
    assert (tmp_path / "interim" / "quarantine").is_dir()
    # processed images are the requested size; raw data untouched
    some = next((tmp_path / "processed" / "train").rglob("*.jpg"))
    assert Image.open(some).size == (64, 64)
    assert (raw / "plantvillage" / "Apple___Apple_scab" / "broken.jpg").exists()
    # the produced dataset passes validation (no leakage, correct sizes)
    result = dv.validate(tmp_path / "processed", image_size=64)
    assert result["ok"], result["errors"]
    assert result["classes_present"] == 2


def test_rerun_is_deterministic_and_uses_cache(tmp_path):
    raw = _make_raw(tmp_path)
    pp.run(_cfg(tmp_path, raw), log=lambda *_: None)
    first = (tmp_path / "processed" / "manifest.csv").read_text()
    logs = []
    pp.run(_cfg(tmp_path, raw), log=logs.append)
    assert (tmp_path / "processed" / "manifest.csv").read_text() == first
    assert any("reused from cache" in str(l) and "0 images (" in str(l) for l in logs) or any("reused" in str(l) for l in logs)


def test_validation_detects_leakage_and_missing_files(tmp_path):
    raw = _make_raw(tmp_path)
    pp.run(_cfg(tmp_path, raw), log=lambda *_: None)
    manifest = tmp_path / "processed" / "manifest.csv"
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    train = next(r for r in rows if r["split"] == "train")
    clone = dict(train, split="test")           # same content in two splits
    rows.append(clone)
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = dv.validate(tmp_path / "processed", image_size=64)
    assert not res["ok"] and any("LEAKAGE" in e for e in res["errors"])
    # deleting a processed file must be reported as an error too
    (tmp_path / "processed" / train["processed_path"]).unlink()
    res = dv.validate(tmp_path / "processed", image_size=64)
    assert any("missing" in e for e in res["errors"])


# ---------------------------------------------------------------- acquisition helpers
def test_safe_extract_blocks_traversal_and_corruption(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(dd.AcquisitionError):
        dd.safe_extract(evil, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()

    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK\x03\x04 definitely not a zip")
    with pytest.raises(dd.AcquisitionError):
        dd.safe_extract(bad, tmp_path / "out2")

    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("lab/a.jpg", to_bytes(make_image(1)))
    assert dd.safe_extract(good, tmp_path / "out3") == 1
    assert (tmp_path / "out3" / "lab" / "a.jpg").is_file()


def test_ingest_records_provenance_and_status(tmp_path):
    arc = tmp_path / "rice.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        for i in range(3):
            zf.writestr(f"Brown spot/{i}.jpg", to_bytes(make_image(i)))
    raw = tmp_path / "raw"
    st = dd.ingest_archive("rice_mendeley", arc, raw_dir=raw)
    assert st["image_count"] == 3 and not st["labels_confirmed"]
    prov = json.loads((raw / "provenance.json").read_text())
    assert prov["sources"]["rice_mendeley"]["verified_image_count"] == 3
    assert prov["sources"]["rice_mendeley"]["archive_sha256"]
    assert st["label_mapping"]["Brown spot"]["class_key"] == "Rice___Brown_spot"
    dd.confirm_labels("rice_mendeley", raw_dir=raw)
    assert dd.source_status("rice_mendeley", raw)["labels_confirmed"]


def test_confirmed_wheat_source_is_included(tmp_path):
    raw = tmp_path / "raw"
    for lab, n in (("Yellow rust", 6), ("Septoria", 4), ("Rust", 2)):
        d = raw / "wheat_wfd" / lab
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i}.jpg").write_bytes(to_bytes(make_image(hash((lab, i)) % 10000)))
    dd.confirm_labels("wheat_wfd", raw_dir=raw)
    report = pp.run(_cfg(tmp_path, raw, sources=["wheat_wfd"]), log=lambda *_: None)
    counts = {r["class_key"]: r["total"] for r in report["classes"]}
    assert counts["Wheat___Stripe_rust"] == 6
    reasons = {e["source_label"]: e["reason"] for e in report["excluded_labels"]}
    assert reasons == {"Septoria": "unsupported", "Rust": "ambiguous"}


def test_near_duplicate_survives_jpeg_recompression_and_brightness_but_not_a_different_picture():
    base = make_image(21, size=(160, 160))
    recompressed = Image.open(io.BytesIO(to_bytes(base, "JPEG"))).convert("RGB")
    from PIL import ImageEnhance
    brighter = ImageEnhance.Brightness(base).enhance(1.12)
    hs = [pp.dhash_variants(i) for i in (base, recompressed, brighter, make_image(22, size=(160, 160)))]
    groups, _ = pp.cluster_near_duplicates(hs, threshold=10)
    assert groups[0] == groups[1] == groups[2] != groups[3]


def test_group_size_cap_prevents_chaining():
    """A chain a-b-c-d... where only neighbours are close must not collapse into one giant group."""
    rng = __import__("random").Random(0)
    base = rng.getrandbits(256)
    chain = []
    h = base
    for _ in range(30):
        chain.append((h,))
        flip = rng.sample(range(256), 8)               # each step moves 8 bits: neighbours match, ends do not
        for bit in flip:
            h ^= 1 << bit
    groups, refused = pp.cluster_near_duplicates(chain, threshold=10, max_group=5)
    assert max(__import__("collections").Counter(groups).values()) <= 5 and refused > 0
    unlimited, _ = pp.cluster_near_duplicates(chain, threshold=10, max_group=1000)
    assert max(__import__("collections").Counter(unlimited).values()) > 5
