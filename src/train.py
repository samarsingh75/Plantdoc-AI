"""Train the 46-class CNN on the prepared dataset.

    python -m src.train                       # 'lightweight' preset (student laptop / CPU)
    python -m src.train --preset standard     # use every training image, more epochs
    python -m src.train --weights none        # no ImageNet download (much lower accuracy)

Two phases (transfer learning):
  1. head     - backbone frozen, only the new 46-way classification head is trained
  2. finetune - the last N backbone layers are unfrozen and trained with a tiny learning rate
The best weights by validation accuracy are saved to models/plant_disease_model.keras, together with
models/model_metadata.json (ordered class names, preprocessing, data counts, honest training facts).
The held-out TEST set is never touched here; use  python -m src.evaluate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .config import (MODEL_FILENAME, MODEL_METADATA_FILENAME, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED,
                     TRAIN_CONFIG_PATH)
from .dataset_validation import read_manifest, validate
from .labels import load_mapping


def load_preset(name: str, path: Path = TRAIN_CONFIG_PATH) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if name not in doc["presets"]:
        raise SystemExit(f"unknown preset {name!r}; choose from {list(doc['presets'])}")
    return dict(doc["presets"][name])


def select_training_rows(rows: List[dict], max_per_class: Optional[int]) -> List[dict]:
    """Deterministically cap the number of TRAIN images per class (never touches val/test).
    Ordering is by SHA-256, i.e. pseudo-random but independent of file names/order."""
    train = [r for r in rows if r["split"] == "train"]
    if not max_per_class:
        return sorted(train, key=lambda r: r["sha256"])
    by_class: Dict[str, List[dict]] = defaultdict(list)
    for r in train:
        by_class[r["class_key"]].append(r)
    picked: List[dict] = []
    for key in sorted(by_class):
        picked.extend(sorted(by_class[key], key=lambda r: r["sha256"])[:max_per_class])
    return picked


def compute_class_weights(labels: List[int], num_classes: int, mode: str) -> Optional[Dict[int, float]]:
    if mode != "balanced":
        return None
    counts = Counter(labels)
    present = [c for c in range(num_classes) if counts.get(c, 0) > 0]
    total = sum(counts.values())
    return {c: float(np.clip(total / (len(present) * counts[c]), 0.2, 10.0)) for c in present}


def make_dataset(paths: List[str], labels: List[int], image_size: int, batch_size: int, training: bool, seed: int):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((paths, np.asarray(labels, dtype=np.int32)))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3, dct_method="INTEGER_ACCURATE")
        img = tf.cast(img, tf.float32)
        img.set_shape([None, None, 3])
        img = tf.cond(tf.reduce_all(tf.shape(img)[:2] == image_size), lambda: img,
                      lambda: tf.image.resize(img, [image_size, image_size], method="bilinear"))
        img.set_shape([image_size, image_size, 3])
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        import keras
        from keras import layers
        augment = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.08, fill_mode="reflect"),
            layers.RandomZoom(0.15, fill_mode="reflect"),
            layers.RandomBrightness(0.15, value_range=(0, 255)),
            layers.RandomContrast(0.15),
        ], name="augmentation")
        # augmentation is applied to TRAINING batches only; val/test are never augmented
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _paths_labels(rows: List[dict], processed_dir: Path, mapping) -> Tuple[List[str], List[int]]:
    paths = [str(Path(processed_dir) / r["processed_path"]) for r in rows]
    labels = [mapping.index_of(r["class_key"]) for r in rows]
    return paths, labels


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(args: argparse.Namespace) -> int:
    mapping = load_mapping()
    cfg = load_preset(args.preset)
    for name in ("architecture", "batch_size", "head_epochs", "finetune_epochs", "dropout"):
        if getattr(args, name, None) is not None:
            cfg[name] = getattr(args, name)
    if args.max_train_per_class is not None:
        cfg["max_train_per_class"] = args.max_train_per_class or None
    image_size = int(cfg["image_size"])

    # ---- 1. data checks -------------------------------------------------------
    print("Validating the prepared dataset ...")
    try:
        report = validate(args.processed_dir, image_size)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for w in report["warnings"]:
        print("  WARNING:", w[:300])
    if not report["ok"]:
        for e in report["errors"]:
            print("  ERROR:", e, file=sys.stderr)
        print("Fix the dataset (python -m src.preprocess) before training.", file=sys.stderr)
        return 1
    if report["missing_classes"] and not args.allow_missing_classes:
        print("\nERROR: these classes have no images, so the model could not learn them:\n  "
              + "\n  ".join(report["missing_classes"]) +
              "\nObtain the data (README: 'Wheat and rice data') or, to train anyway on the classes that do have data,\n"
              "re-run with --allow-missing-classes. The app will then clearly state that those classes are untrained.",
              file=sys.stderr)
        return 1

    rows = read_manifest(args.processed_dir)
    train_rows = select_training_rows(rows, cfg.get("max_train_per_class"))
    val_rows = [r for r in rows if r["split"] == "val"]
    full_train = sum(1 for r in rows if r["split"] == "train")
    tr_paths, tr_labels = _paths_labels(train_rows, args.processed_dir, mapping)
    va_paths, va_labels = _paths_labels(val_rows, args.processed_dir, mapping)
    cw = compute_class_weights(tr_labels, mapping.num_classes, cfg.get("class_weight", "none"))
    print(f"Training images: {len(tr_paths)} (of {full_train} available), validation images: {len(va_paths)}")

    # ---- 2. model --------------------------------------------------------------
    import keras
    import tensorflow as tf
    from keras import callbacks
    from .model import WeightsUnavailableError, build_model, compile_model, set_finetune

    keras.utils.set_random_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Devices: {'GPU x' + str(len(gpus)) if gpus else 'CPU only'}")
    weights = None if args.weights == "none" else args.weights
    try:
        model = build_model(mapping.num_classes, image_size, cfg["architecture"], weights, cfg["dropout"])
    except WeightsUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    train_ds = make_dataset(tr_paths, tr_labels, image_size, cfg["batch_size"], True, args.seed)
    val_ds = make_dataset(va_paths, va_labels, image_size, cfg["batch_size"], False, args.seed)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    tmp_model = args.models_dir / ("training_in_progress_" + MODEL_FILENAME)
    history: Dict[str, List[float]] = defaultdict(list)
    phases: List[dict] = []
    best_val = -1.0

    def run_phase(name: str, epochs: int, lr: float) -> None:
        nonlocal best_val
        if epochs <= 0:
            return
        compile_model(model, lr)
        ckpt = callbacks.ModelCheckpoint(str(tmp_model), monitor="val_accuracy", mode="max", save_best_only=True,
                                         initial_value_threshold=best_val if best_val >= 0 else None, verbose=0)
        cbs = [ckpt,
               callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=cfg["early_stopping_patience"],
                                       restore_best_weights=True),
               callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7)]
        print(f"\n=== Phase '{name}': {epochs} epochs, lr={lr} ===")
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw, callbacks=cbs, shuffle=False, verbose=1)
        for k, v in h.history.items():
            history[k].extend(float(x) for x in v)
        phases.append({"name": name, "epochs_run": len(h.history["loss"]), "learning_rate": lr})
        best_val = max(best_val, max(h.history["val_accuracy"]))

    run_phase("head", int(cfg["head_epochs"]), float(cfg["head_learning_rate"]))
    if int(cfg["finetune_epochs"]) > 0:
        n_trainable = set_finetune(model, int(cfg["finetune_layers"]))
        print(f"Fine-tuning {n_trainable} backbone layers")
        run_phase("finetune", int(cfg["finetune_epochs"]), float(cfg["finetune_learning_rate"]))
    if not tmp_model.is_file():
        print("ERROR: training produced no checkpoint (zero epochs?).", file=sys.stderr)
        return 1

    # ---- 3. finalize: move best checkpoint into place, evaluate on VALIDATION only ---------
    final = args.models_dir / MODEL_FILENAME
    os.replace(tmp_model, final)
    best = keras.models.load_model(final, compile=False)
    compile_model(best, 1e-4)
    val_loss, val_acc, val_top3 = best.evaluate(val_ds, verbose=0)
    print(f"\nBest model: validation accuracy {val_acc:.4f}, top-3 {val_top3:.4f}, loss {val_loss:.4f}")

    train_counts = Counter(r["class_key"] for r in train_rows)
    untrained = [k for k in mapping.keys if train_counts.get(k, 0) == 0]
    few = [k for k in mapping.keys if 0 < train_counts.get(k, 0) < 30]
    (args.models_dir / "training_history.json").write_text(
        json.dumps({"phases": phases, "history": history}, indent=2), encoding="utf-8")
    manifest_path = args.processed_dir / "manifest.csv"
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_file": MODEL_FILENAME,
        "architecture": cfg["architecture"],
        "input_size": image_size,
        "preprocessing": {
            "input": "RGB, float32, pixel values 0-255, bilinear resize (aspect ratio not preserved)",
            "scaling_inside_model": "x/127.5 - 1" if cfg["architecture"] == "mobilenetv2" else "EfficientNet built-in",
            "shared_code": "src/image_utils.py"},
        "num_classes": mapping.num_classes,
        "class_names": mapping.keys,
        "class_mapping_version": mapping.version,
        "class_mapping_sha256": mapping.sha256(),
        "weights_initialisation": "imagenet" if weights else "random (no pretrained weights)",
        "training": {"preset": args.preset, "config": cfg, "seed": args.seed, "phases": phases,
                     "train_images_used": len(tr_paths), "train_images_available": full_train,
                     "train_subsampled": len(tr_paths) < full_train, "val_images": len(va_paths),
                     "class_weights": "balanced" if cw else "none", "device": "GPU" if gpus else "CPU"},
        "data": {"manifest_sha256": _sha256_file(manifest_path),
                 "train_images_per_class_used": {k: train_counts.get(k, 0) for k in mapping.keys}},
        "classes_without_training_data": untrained,
        "classes_with_fewer_than_30_training_images": few,
        "validation": {"accuracy": float(val_acc), "top3_accuracy": float(val_top3), "loss": float(val_loss),
                       "note": "validation set was used for checkpoint selection; use the test set for final numbers"},
        "test_metrics": None,
        "software": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
                     "platform": platform.platform()},
    }
    (args.models_dir / MODEL_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")
    print(f"Saved {final} and {args.models_dir / MODEL_METADATA_FILENAME}")
    if untrained:
        print("NOTE: no training images for:", ", ".join(untrained))
    print("Next: python -m src.evaluate")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.train", description="Train the 46-class plant disease CNN.")
    p.add_argument("--preset", default=None, help="lightweight (default) or standard")
    p.add_argument("--architecture", choices=["mobilenetv2", "efficientnetb0"], default=None)
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--head-epochs", dest="head_epochs", type=int, default=None)
    p.add_argument("--finetune-epochs", dest="finetune_epochs", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--max-train-per-class", type=int, default=None, help="0 = use all training images")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--allow-missing-classes", action="store_true",
                   help="train even though some of the 46 classes have no images")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preset is None:
        args.preset = json.loads(TRAIN_CONFIG_PATH.read_text(encoding="utf-8"))["default_preset"]
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
