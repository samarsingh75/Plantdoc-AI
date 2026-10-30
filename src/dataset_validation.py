"""Validate a prepared dataset (data/processed) before training.

    python -m src.dataset_validation

Errors (exit code 1) block training; warnings are reported but allow it.
Errors: missing/invalid manifest, unknown class keys, missing or unreadable files, wrong image size,
        duplicate content or near-duplicate groups spanning splits (leakage), empty train/val split.
Warnings: classes without images, classes with very few images, tiny val/test sets.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .config import IMAGE_SIZE, PROCESSED_DIR, REPORTS_DIR
from .labels import load_mapping

REQUIRED_COLUMNS = ["split", "class_key", "class_index", "processed_path", "source_id", "source_path",
                    "sha256", "group_id"]
SPLITS = ("train", "val", "test")


def read_manifest(processed_dir: Path) -> List[dict]:
    path = Path(processed_dir) / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run:  python -m src.preprocess")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest.csv is missing columns: {missing}")
        return list(reader)


def validate(processed_dir: Path = PROCESSED_DIR, image_size: int = IMAGE_SIZE, deep_sample: int = 300,
             seed: int = 42) -> dict:
    mapping = load_mapping()
    errors: List[str] = []
    warnings: List[str] = []
    rows = read_manifest(processed_dir)
    processed_dir = Path(processed_dir)

    counts: Dict[str, Counter] = {s: Counter() for s in SPLITS}
    bad_split = bad_key = bad_index = missing_files = bad_size = 0
    sha_splits: Dict[str, set] = defaultdict(set)
    group_splits: Dict[str, set] = defaultdict(set)
    for r in rows:
        s, key = r["split"], r["class_key"]
        if s not in SPLITS:
            bad_split += 1
            continue
        try:
            if mapping.index_of(key) != int(r["class_index"]):
                bad_index += 1
        except KeyError:
            bad_key += 1
            continue
        counts[s][key] += 1
        sha_splits[r["sha256"]].add(s)
        group_splits[r["group_id"]].add(s)
        f = processed_dir / r["processed_path"]
        if not f.is_file():
            missing_files += 1
            continue
        try:
            with Image.open(f) as im:
                if im.size != (image_size, image_size):
                    bad_size += 1
        except Exception:
            bad_size += 1
    if bad_split:
        errors.append(f"{bad_split} rows have an invalid split name")
    if bad_key:
        errors.append(f"{bad_key} rows reference class keys that are not in the 46-class taxonomy")
    if bad_index:
        errors.append(f"{bad_index} rows have a class_index that disagrees with class_mapping.json")
    if missing_files:
        errors.append(f"{missing_files} processed files listed in the manifest are missing")
    if bad_size:
        errors.append(f"{bad_size} processed images are unreadable or not {image_size}x{image_size}")
    leaked_sha = sum(1 for v in sha_splits.values() if len(v) > 1)
    leaked_groups = sum(1 for v in group_splits.values() if len(v) > 1)
    if leaked_sha:
        errors.append(f"LEAKAGE: {leaked_sha} identical images appear in more than one split")
    if leaked_groups:
        errors.append(f"LEAKAGE: {leaked_groups} near-duplicate groups span more than one split")

    # deep decode of a random sample (catches truncated JPEGs that header-only checks miss)
    rng = random.Random(seed)
    sample = rng.sample(rows, min(deep_sample, len(rows))) if rows else []
    undecodable = 0
    for r in sample:
        try:
            with Image.open(processed_dir / r["processed_path"]) as im:
                im.load()
        except Exception:
            undecodable += 1
    if undecodable:
        errors.append(f"{undecodable} of {len(sample)} sampled images failed to decode")

    sizes = {s: sum(counts[s].values()) for s in SPLITS}
    if sizes["train"] == 0 or sizes["val"] == 0:
        errors.append("the training or validation split is empty")
    if sizes["test"] == 0:
        warnings.append("the test split is empty: no held-out evaluation is possible")

    per_class = []
    missing_classes, tiny = [], []
    for c in mapping.classes:
        tr, va, te = (counts[s].get(c.key, 0) for s in SPLITS)
        per_class.append({"index": c.index, "class_key": c.key, "train": tr, "val": va, "test": te})
        if tr + va + te == 0:
            missing_classes.append(c.key)
        elif tr < 30:
            tiny.append(f"{c.key} (train={tr})")
    if missing_classes:
        warnings.append(f"{len(missing_classes)} classes have NO images and cannot be learned: "
                        f"{', '.join(missing_classes)}")
    if tiny:
        warnings.append(f"classes with fewer than 30 training images: {'; '.join(tiny)}")
    if sizes["test"] and any(r["test"] < 10 for r in per_class if r["train"] + r["val"] + r["test"] > 0):
        warnings.append("some classes have fewer than 10 test images; their test metrics are unreliable")

    report = {"ok": not errors, "errors": errors, "warnings": warnings, "images": len(rows), "split_sizes": sizes,
              "classes_present": mapping.num_classes - len(missing_classes), "missing_classes": missing_classes,
              "per_class": per_class, "checked_deep_sample": len(sample)}
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.dataset_validation")
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    a = p.parse_args(argv)
    try:
        report = validate(a.processed_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    a.reports_dir.mkdir(parents=True, exist_ok=True)
    (a.reports_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"images: {report['images']}  splits: {report['split_sizes']}  "
          f"classes with data: {report['classes_present']}/{load_mapping().num_classes}")
    for w in report["warnings"]:
        print("WARNING:", w)
    for e in report["errors"]:
        print("ERROR:", e)
    print("RESULT:", "PASSED" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
