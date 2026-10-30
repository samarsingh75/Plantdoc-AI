"""Evaluate the trained model on the held-out TEST split and write reports.

    python -m src.evaluate

Outputs (data/reports/): evaluation_report.md, evaluation_metrics.json, classification_report.csv,
confusion_matrix.csv / confusion_matrix.png, training_curves.png, class_distribution.png.
The test split was never used for training, checkpoint selection or early stopping.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .config import MODEL_METADATA_FILENAME, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .dataset_validation import read_manifest
from .labels import load_mapping
from .predict import ModelMismatchError, ModelNotFoundError, load_predictor

CAVEATS = [
    "These numbers describe performance on THIS test split only. They depend on dataset quality, label noise, "
    "class balance, image diversity and how similar the test images are to the training images.",
    "PlantVillage photos are single leaves on uniform backgrounds; images split randomly from the same source are "
    "much easier than real field photos. Expect lower accuracy on pictures taken in a field or with a phone.",
    "Softmax confidence is not a calibrated probability of being correct, and the model has no 'none of the above' "
    "class: a photo of something else will still be assigned one of the 46 classes.",
    "This is an educational model, not a production-grade or laboratory-confirmed diagnostic tool.",
]


def predict_split(predictor, rows: List[dict], processed_dir: Path, batch_size: int = 32) -> np.ndarray:
    """Probabilities for every manifest row, using the SAME preprocessing code as inference."""
    from PIL import Image
    from .image_utils import prepare_batch
    probs = []
    for i in range(0, len(rows), batch_size):
        batch = []
        for r in rows[i:i + batch_size]:
            with Image.open(processed_dir / r["processed_path"]) as im:
                batch.append(prepare_batch(im.convert("RGB"), predictor.image_size)[0])
        probs.append(np.asarray(predictor.model(np.stack(batch), training=False)))
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, predictor.mapping.num_classes))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mapping) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    n = mapping.num_classes
    present = sorted(set(y_true.tolist()))      # classes that actually occur in the test set
    names = [mapping.by_index(i).key for i in present]
    out = {"n_test": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
           "classes_in_test": len(present), "classes_absent_from_test": [mapping.by_index(i).key for i in range(n) if i not in present]}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average=avg, zero_division=0)
        out[avg] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    out["classification_report"] = classification_report(y_true, y_pred, labels=present, target_names=names,
                                                         output_dict=True, zero_division=0)
    out["classification_report_text"] = classification_report(y_true, y_pred, labels=present, target_names=names,
                                                              zero_division=0)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(n))).tolist()
    return out


def plot_confusion(cm: np.ndarray, mapping, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, sums, out=np.zeros_like(cm, dtype=float), where=sums > 0)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(mapping.num_classes))
    ax.set_yticks(range(mapping.num_classes))
    ax.set_xticklabels([str(i) for i in range(mapping.num_classes)], fontsize=6)
    ax.set_yticklabels([f"{i} {mapping.by_index(i).key[:30]}" for i in range(mapping.num_classes)], fontsize=6)
    ax.set_xlabel("predicted class index")
    ax.set_ylabel("true class")
    ax.set_title("Confusion matrix on the test set (row-normalised; empty rows = class absent from test set)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_training_curves(history_path: Path, out_dir: Path) -> bool:
    if not history_path.is_file():
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doc = json.loads(history_path.read_text(encoding="utf-8"))
    h = doc["history"]
    boundary = doc["phases"][0]["epochs_run"] if len(doc["phases"]) > 1 else None
    for key, title, fname in (("accuracy", "Accuracy", "training_accuracy.png"), ("loss", "Loss", "training_loss.png")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h[key], label=f"train {key}")
        ax.plot(h["val_" + key], label=f"validation {key}")
        if boundary:
            ax.axvline(boundary - 0.5, color="grey", linestyle="--", label="start of fine-tuning")
        ax.set_xlabel("epoch")
        ax.set_title(f"Training and validation {title.lower()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)
    return True


def evaluate(models_dir: Path, processed_dir: Path, reports_dir: Path) -> int:
    try:
        predictor = load_predictor(models_dir)
    except (ModelNotFoundError, ModelMismatchError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    mapping = predictor.mapping
    rows = [r for r in read_manifest(processed_dir) if r["split"] == "test"]
    if not rows:
        print("ERROR: the test split is empty; nothing to evaluate.", file=sys.stderr)
        return 1
    print(f"Evaluating on {len(rows)} held-out test images ...")
    probs = predict_split(predictor, rows, processed_dir)
    y_true = np.array([mapping.index_of(r["class_key"]) for r in rows])
    y_pred = probs.argmax(axis=1)
    m = compute_metrics(y_true, y_pred, mapping)

    reports_dir.mkdir(parents=True, exist_ok=True)
    # per-source accuracy: a big gap between sources hints at dataset-background shortcuts
    by_source = {}
    for s in sorted({r["source_id"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source_id"] == s]
        by_source[s] = {"n": len(idx), "accuracy": float((y_true[idx] == y_pred[idx]).mean())}
    m["accuracy_by_source"] = by_source
    warnings = []
    support = Counter(y_true.tolist())
    small = [mapping.by_index(c).key for c, n in support.items() if n < 10]
    if small:
        warnings.append(f"{len(small)} classes have fewer than 10 test images; their per-class scores are unreliable.")
    if m["classes_absent_from_test"]:
        warnings.append("Classes with no test images (not evaluated): " + ", ".join(m["classes_absent_from_test"]))
    if predictor.untrained_classes:
        warnings.append("Classes with no TRAINING data: " + ", ".join(predictor.untrained_classes))
    if len(rows) < 20 * m["classes_in_test"]:
        warnings.append("The test set is small relative to the number of classes; treat the headline accuracy as a rough estimate.")
    if predictor.metadata.get("weights_initialisation", "").startswith("random"):
        warnings.append("This model was trained WITHOUT pretrained weights.")
    m["warnings"], m["caveats"] = warnings, CAVEATS
    m["evaluated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cm = np.array(m["confusion_matrix"])
    (reports_dir / "evaluation_metrics.json").write_text(
        json.dumps({k: v for k, v in m.items() if k != "classification_report_text"}, indent=2), encoding="utf-8")
    with open(reports_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for k, v in m["classification_report"].items():
            if isinstance(v, dict):
                w.writerow([k, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1-score']:.4f}", int(v["support"])])
    np.savetxt(reports_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, mapping, reports_dir / "confusion_matrix.png")
    have_curves = plot_training_curves(models_dir / "training_history.json", reports_dir)
    report_json = reports_dir / "dataset_report.json"
    if report_json.is_file():
        from .preprocess import plot_class_distribution
        plot_class_distribution(json.loads(report_json.read_text(encoding="utf-8")), reports_dir / "class_distribution.png")

    md = ["# Evaluation report", "", f"Evaluated: {m['evaluated_utc']}  |  model: {predictor.metadata.get('architecture')} "
          f"({predictor.metadata.get('weights_initialisation')})", "",
          "## Headline (held-out test set)", "",
          f"- Test images: **{m['n_test']}** across {m['classes_in_test']} classes",
          f"- Accuracy: **{m['accuracy']:.4f}**",
          f"- Macro   precision / recall / F1: {m['macro']['precision']:.4f} / {m['macro']['recall']:.4f} / {m['macro']['f1']:.4f}",
          f"- Weighted precision / recall / F1: {m['weighted']['precision']:.4f} / {m['weighted']['recall']:.4f} / {m['weighted']['f1']:.4f}",
          "", "Accuracy by data source:", ""]
    md += [f"- `{s}`: {v['accuracy']:.4f} (n={v['n']})" for s, v in by_source.items()]
    md += ["", "## Warnings", ""] + ([f"- {w}" for w in warnings] or ["- none"])
    md += ["", "## Caveats", ""] + [f"- {c}" for c in CAVEATS]
    md += ["", "## Per-class report", "", "```", m["classification_report_text"], "```", "",
           "Files: `confusion_matrix.png`, `confusion_matrix.csv`, `classification_report.csv`, "
           + ("`training_accuracy.png`, `training_loss.png`, " if have_curves else "") + "`class_distribution.png`", ""]
    (reports_dir / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")

    # record the measured numbers in the model metadata (only real, measured values)
    meta_path = models_dir / MODEL_METADATA_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["test_metrics"] = {"n_test": m["n_test"], "accuracy": m["accuracy"], "macro_f1": m["macro"]["f1"],
                            "weighted_f1": m["weighted"]["f1"], "evaluated_utc": m["evaluated_utc"],
                            "warnings": warnings}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Test accuracy {m['accuracy']:.4f} | macro F1 {m['macro']['f1']:.4f} | weighted F1 {m['weighted']['f1']:.4f}")
    for w in warnings:
        print("WARNING:", w)
    print("Report written to", reports_dir / "evaluation_report.md")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.evaluate")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    a = p.parse_args(argv)
    return evaluate(a.models_dir, a.processed_dir, a.reports_dir)


if __name__ == "__main__":
    sys.exit(main())
