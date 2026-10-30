"""Dataset preparation: raw images -> clean, de-duplicated, split, resized dataset.

    python -m src.preprocess                       # defaults: seed 42, 70/15/15, 224 px

Stages
 1. discover   - find images in every available source, map source labels to the
                 canonical 46 classes (unmapped / ambiguous labels are excluded, never guessed)
 2. inspect    - fully decode every image (corrupt files are quarantined), compute a
                 SHA-256 and a perceptual difference-hash (dHash) robust to flips/rotations
 3. de-duplicate - exact duplicates dropped (same-label) or removed as label conflicts
                 (different labels); near-duplicates are *grouped*, not dropped
 4. split      - deterministic, stratified, GROUP-aware 70/15/15 split, so images of one
                 near-duplicate group can never straddle train/val/test
 5. write      - resized copies in data/processed/{train,val,test}/<class_key>/ + manifest.csv
 6. report     - data/reports/dataset_report.{json,md}, class_distribution.png, CSV lists

The original raw files are never modified or deleted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import __version__
from .config import (IMAGE_SIZE, INTERIM_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, SEED)
from .data_download import labels_confirmed, load_sources, source_dir
from .discovery import iter_images
from .image_utils import ImageValidationError, open_rgb, resize_for_model
from .labels import ClassMapping, load_mapping

SPLITS = ("train", "val", "test")
INVENTORY_NAME = "inventory.csv"
JPEG_QUALITY = 95


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class PrepConfig:
    raw_dir: Path = RAW_DIR
    interim_dir: Path = INTERIM_DIR
    processed_dir: Path = PROCESSED_DIR
    reports_dir: Path = REPORTS_DIR
    image_size: int = IMAGE_SIZE
    seed: int = SEED
    val_frac: float = 0.15
    test_frac: float = 0.15
    phash_threshold: int = 10         # max differing bits (of 256) for "near-duplicate"
    include_unconfirmed: bool = False  # include sources whose labels nobody confirmed yet
    workers: int = 4
    min_per_split_warn: int = 10      # warn when a class has fewer val/test images than this
    sources: Optional[Sequence[str]] = None
    write_images: bool = True


@dataclass
class Record:
    source_id: str
    path: Path
    rel_path: str
    source_label: str
    class_key: str
    size: int = 0
    mtime_ns: int = 0
    status: str = "pending"      # ok | corrupt
    reason: str = ""
    sha256: str = ""
    width: int = 0
    height: int = 0
    hashes: Tuple[int, ...] = field(default_factory=tuple)   # 8 dihedral 256-bit dHash variants
    group: int = -1
    split: str = ""


# ----------------------------------------------------------------------------
# perceptual hashing
# ----------------------------------------------------------------------------
HASH_BITS = 256          # 16 x 16 gradient bits; 64-bit hashes proved far too coarse on uniform-background
                         # leaf photos (they chained ~20% of PlantVillage into one false "duplicate" group)
THUMB = 48
MAX_GROUP_SIZE = 64      # a genuine near-duplicate group is small; refuse unions that would exceed this

_DIHEDRAL = (None, Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_180,
             Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270, Image.Transpose.TRANSPOSE,
             Image.Transpose.TRANSVERSE)


def _dhash_from_gray(gray: Image.Image) -> int:
    small = np.asarray(gray.resize((17, 16), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()                 # 16 x 16 = 256 bits
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def dhash_variants(img: Image.Image) -> Tuple[int, ...]:
    """256-bit dHash of the image and of its 7 flips/rotations (so a flipped or rotated copy
    of a photo is still recognised as a near-duplicate)."""
    thumb = img.convert("L").resize((THUMB, THUMB), Image.BILINEAR)
    return tuple(_dhash_from_gray(thumb if op is None else thumb.transpose(op)) for op in _DIHEDRAL)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _is_textureless(h: int) -> bool:
    c = h.bit_count()
    return c <= HASH_BITS * 0.06 or c >= HASH_BITS * 0.94   # almost constant image: hash is meaningless


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, max_size: Optional[int] = None) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if max_size and self.size[ra] + self.size[rb] > max_size:
            return False
        lo, hi = min(ra, rb), max(ra, rb)
        self.parent[hi] = lo
        self.size[lo] += self.size[hi]
        return True


def cluster_near_duplicates(hashes: Sequence[Tuple[int, ...]], threshold: int,
                            max_group: int = MAX_GROUP_SIZE) -> Tuple[List[int], int]:
    """Return ``(group_id_per_item, refused_unions)``. Items within ``threshold`` bits (under any
    flip/rotation) share a group. Banded lookup keeps it near-linear for ~50k images."""
    n = len(hashes)
    uf = UnionFind(n)
    n_bands = threshold + 1                       # pigeonhole: distance <= threshold => a band matches
    edges = [round(i * HASH_BITS / n_bands) for i in range(n_bands + 1)]
    bands = [(edges[i], edges[i + 1]) for i in range(n_bands)]

    def band_val(h: int, lo_hi: Tuple[int, int]) -> int:
        lo, hi = lo_hi
        return (h >> lo) & ((1 << (hi - lo)) - 1)

    index: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, hs in enumerate(hashes):
        if not hs or _is_textureless(hs[0]):
            continue
        for b, spec in enumerate(bands):
            index[(b, band_val(hs[0], spec))].append(i)
    refused = 0
    for j, hs in enumerate(hashes):
        if not hs or _is_textureless(hs[0]):
            continue
        for variant in hs:
            for b, spec in enumerate(bands):
                bucket = index.get((b, band_val(variant, spec)))
                if not bucket or len(bucket) > 2000:      # skip pathological buckets
                    continue
                for i in bucket:
                    if i != j and uf.find(i) != uf.find(j) and hamming(variant, hashes[i][0]) <= threshold:
                        if not uf.union(i, j, max_group):
                            refused += 1
    roots: Dict[int, int] = {}
    groups = []
    for i in range(n):
        groups.append(roots.setdefault(uf.find(i), len(roots)))
    return groups, refused


# ----------------------------------------------------------------------------
# stage 1: discovery
# ----------------------------------------------------------------------------
def discover(cfg: PrepConfig, mapping: ClassMapping) -> Tuple[List[Record], List[dict], List[dict]]:
    """Return (records, excluded_label_rows, source_summaries)."""
    sources = load_sources()
    wanted = list(cfg.sources) if cfg.sources else list(sources)
    records: List[Record] = []
    excluded: List[dict] = []
    summaries: List[dict] = []
    for sid in wanted:
        if sid not in sources:
            raise ValueError(f"unknown source {sid!r}")
        root = source_dir(sid, cfg.raw_dir, sources)
        if not root.is_dir():
            summaries.append({"source_id": sid, "found": False, "images": 0, "used": 0,
                              "note": "directory not found - source not available"})
            continue
        confirmed = labels_confirmed(sid, cfg.raw_dir, sources)
        use_source = confirmed or cfg.include_unconfirmed
        n_found = n_used = 0
        label_counts: Counter = Counter()
        label_status: Dict[str, Tuple[Optional[str], str]] = {}
        for path, label in iter_images(root):
            n_found += 1
            label_counts[label] += 1
            if label not in label_status:
                label_status[label] = mapping.map_source_label(sid, label)
            key, status = label_status[label]
            if key is None or not use_source:
                continue
            records.append(Record(source_id=sid, path=path, rel_path=path.relative_to(cfg.raw_dir).as_posix(),
                                  source_label=label, class_key=key))
            n_used += 1
        for label, n in sorted(label_counts.items()):
            key, status = label_status[label]
            if key is None:
                excluded.append({"source_id": sid, "source_label": label, "images": n, "reason": status})
            elif not use_source:
                excluded.append({"source_id": sid, "source_label": label, "images": n,
                                 "reason": "source labels not confirmed (run confirm-labels)"})
        summaries.append({"source_id": sid, "found": True, "images": n_found, "used": n_used,
                          "labels_confirmed": confirmed, "included": use_source,
                          "note": "" if use_source else
                          "labels not confirmed: excluded. Review with 'review-labels' then 'confirm-labels'."})
    records.sort(key=lambda r: r.rel_path)
    return records, excluded, summaries


# ----------------------------------------------------------------------------
# stage 2: inspection (validation + hashing) with an on-disk cache
# ----------------------------------------------------------------------------
def _inspect(rec: Record) -> Record:
    try:
        st = rec.path.stat()
        rec.size, rec.mtime_ns = st.st_size, st.st_mtime_ns
        data = rec.path.read_bytes()
        rec.sha256 = hashlib.sha256(data).hexdigest()
        img = open_rgb(data)
        rec.width, rec.height = img.size
        rec.hashes = dhash_variants(img)
        rec.status = "ok"
    except ImageValidationError as exc:
        rec.status, rec.reason = "corrupt", str(exc)
    except OSError as exc:
        rec.status, rec.reason = "corrupt", f"unreadable file: {exc}"
    return rec


def _load_cache(path: Path) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if path.is_file():
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if "dhash256" not in (reader.fieldnames or []):
                    return {}                      # cache from an older hash format: recompute
                for row in reader:
                    cache[row["rel_path"]] = row
        except (OSError, csv.Error, KeyError):
            return {}
    return cache


def _save_cache(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rel_path", "size", "mtime_ns", "status", "reason", "sha256", "width", "height", "dhash256"])
        for r in records:
            w.writerow([r.rel_path, r.size, r.mtime_ns, r.status, r.reason, r.sha256, r.width, r.height,
                        ";".join(f"{h:064x}" for h in r.hashes)])


def inspect_all(records: List[Record], cfg: PrepConfig, log=print) -> int:
    """Fill status/hashes for every record, reusing cached results for unchanged files."""
    cache_path = cfg.interim_dir / INVENTORY_NAME
    cache = _load_cache(cache_path)
    todo: List[Record] = []
    reused = 0
    for r in records:
        row = cache.get(r.rel_path)
        try:
            st = r.path.stat()
        except OSError:
            todo.append(r)
            continue
        if row and int(row["size"]) == st.st_size and int(row["mtime_ns"]) == st.st_mtime_ns and row["status"] in ("ok", "corrupt"):
            r.size, r.mtime_ns, r.status, r.reason = st.st_size, st.st_mtime_ns, row["status"], row["reason"]
            r.sha256, r.width, r.height = row["sha256"], int(row["width"] or 0), int(row["height"] or 0)
            r.hashes = tuple(int(h, 16) for h in row["dhash256"].split(";")) if row["dhash256"] else ()
            reused += 1
        else:
            todo.append(r)
    log(f"  inspecting {len(todo)} images ({reused} reused from cache)...")
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        for _ in pool.map(_inspect, todo, chunksize=16):
            done += 1
            if done % 2000 == 0:
                log(f"    {done}/{len(todo)}")
    _save_cache(cache_path, records)
    return reused


def quarantine(records: List[Record], cfg: PrepConfig) -> List[dict]:
    """Copy corrupt files into data/interim/quarantine/ (raw data stays untouched)."""
    rows = []
    for r in records:
        if r.status != "corrupt":
            continue
        dest = cfg.interim_dir / "quarantine" / r.source_id / re.sub(r"\W+", "_", r.reason)[:40]
        try:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.path, dest / f"{r.source_label}__{r.path.name}")
        except OSError:
            pass
        rows.append({"source_id": r.source_id, "path": r.rel_path, "reason": r.reason})
    return rows


# ----------------------------------------------------------------------------
# stage 3: de-duplication
# ----------------------------------------------------------------------------
def remove_exact_duplicates(records: List[Record]) -> Tuple[List[Record], List[dict]]:
    """Same bytes + same class: keep first. Same bytes + different classes: label conflict -> drop all."""
    by_hash: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        by_hash[r.sha256].append(r)
    kept: List[Record] = []
    dropped: List[dict] = []
    for r in records:
        group = by_hash[r.sha256]
        classes = {g.class_key for g in group}
        if len(classes) > 1:
            dropped.append({"path": r.rel_path, "class": r.class_key, "reason": "label_conflict",
                            "duplicate_of": group[0].rel_path})
        elif group[0] is not r:
            dropped.append({"path": r.rel_path, "class": r.class_key, "reason": "exact_duplicate",
                            "duplicate_of": group[0].rel_path})
        else:
            kept.append(r)
    return kept, dropped


# ----------------------------------------------------------------------------
# stage 4: stratified, group-aware split
# ----------------------------------------------------------------------------
def assign_splits(records: List[Record], val_frac: float, test_frac: float, seed: int) -> Dict[str, Dict[str, int]]:
    """Assign ``record.split`` for every record. Groups (near-duplicate clusters) are indivisible.
    Greedy stratified assignment: each group goes to the split whose per-class quota it fills best.
    Fully deterministic for a given seed and input ordering."""
    rng = random.Random(seed)
    per_class: Counter = Counter(r.class_key for r in records)
    fracs = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    target: Dict[str, Dict[str, float]] = {s: {} for s in SPLITS}
    for c, n in per_class.items():
        if n >= 3:
            t_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
            t_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
        else:
            t_val = t_test = 0           # too few images to evaluate: everything trains (reported)
        target["val"][c], target["test"][c] = float(t_val), float(t_test)
        target["train"][c] = float(n - t_val - t_test)
    need = {s: dict(target[s]) for s in SPLITS}

    groups: Dict[int, List[Record]] = defaultdict(list)
    for r in records:
        groups[r.group].append(r)
    order = sorted(groups, key=lambda g: (-len(groups[g]), rng.random(), g))
    for g in order:
        members = groups[g]
        counts = Counter(m.class_key for m in members)
        best, best_score = "train", None
        for s in SPLITS:
            score = sum(n * need[s][c] / max(target[s][c], 1.0) for c, n in counts.items())
            if best_score is None or score > best_score + 1e-12:
                best, best_score = s, score
        for m in members:
            m.split = best
        for c, n in counts.items():
            need[best][c] -= n
    result: Dict[str, Dict[str, int]] = {s: Counter() for s in SPLITS}
    for r in records:
        result[r.split][r.class_key] += 1
    return {s: dict(v) for s, v in result.items()}


# ----------------------------------------------------------------------------
# stage 5: writing the processed dataset
# ----------------------------------------------------------------------------
def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem)[:60] or "img"


def _write_one(args) -> Optional[str]:
    rec, out_path, size = args
    try:
        img = resize_for_model(open_rgb(rec.path), size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=JPEG_QUALITY)
        return None
    except (ImageValidationError, OSError) as exc:
        return f"{rec.rel_path}: {exc}"


def write_processed(records: List[Record], cfg: PrepConfig, mapping: ClassMapping, log=print) -> List[dict]:
    out_root = cfg.processed_dir
    for s in SPLITS:
        d = out_root / s
        if d.exists():
            shutil.rmtree(d)
    jobs, rows = [], []
    for r in records:
        fname = f"{_safe_stem(r.path.name)}_{r.sha256[:8]}.jpg"
        out = out_root / r.split / r.class_key / fname
        jobs.append((r, out, cfg.image_size))
        rows.append({"split": r.split, "class_key": r.class_key, "class_index": mapping.index_of(r.class_key),
                     "processed_path": out.relative_to(out_root).as_posix(), "source_id": r.source_id,
                     "source_path": r.rel_path, "sha256": r.sha256, "group_id": r.group})
    errors: List[str] = []
    if cfg.write_images:
        log(f"  writing {len(jobs)} resized images to {out_root} ...")
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
            for i, err in enumerate(pool.map(_write_one, jobs, chunksize=16), 1):
                if err:
                    errors.append(err)
                if i % 2000 == 0:
                    log(f"    {i}/{len(jobs)}")
    if errors:
        failed = set(e.split(":")[0] for e in errors)
        rows = [r for r in rows if r["source_path"] not in failed]
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "class_key", "class_index", "processed_path",
                                           "source_id", "source_path", "sha256", "group_id"])
        w.writeheader()
        w.writerows(rows)
    return [{"path": e.split(":")[0], "reason": e} for e in errors]


# ----------------------------------------------------------------------------
# stage 6: reporting
# ----------------------------------------------------------------------------
def build_report(cfg: PrepConfig, mapping: ClassMapping, records: List[Record], split_counts: dict,
                 excluded: List[dict], source_summaries: List[dict], corrupt: List[dict],
                 dropped: List[dict], write_failures: List[dict], groups_info: dict) -> dict:
    per_class = Counter(r.class_key for r in records)
    rows = []
    warnings: List[str] = []
    missing, small = [], []
    for c in mapping.classes:
        n = per_class.get(c.key, 0)
        tr, va, te = (split_counts[s].get(c.key, 0) for s in SPLITS)
        rows.append({"index": c.index, "class_key": c.key, "display_name": c.display_name, "total": n,
                     "train": tr, "val": va, "test": te})
        if n == 0:
            missing.append(c.key)
        elif va < cfg.min_per_split_warn or te < cfg.min_per_split_warn:
            small.append(f"{c.key} (val={va}, test={te})")
    if missing:
        warnings.append(f"MISSING CLASSES ({len(missing)}): {', '.join(missing)} have no usable images, so a model "
                        f"trained on this data cannot learn them. Provide valid, correctly labelled data "
                        f"(see README, 'Wheat and rice data').")
    if small:
        warnings.append(f"SMALL EVALUATION SETS ({len(small)} classes have < {cfg.min_per_split_warn} validation or "
                        f"test images, so their metrics are unreliable): {'; '.join(small)}")
    nonzero = [r["total"] for r in rows if r["total"] > 0]
    imbalance = round(max(nonzero) / min(nonzero), 2) if nonzero else None
    if imbalance and imbalance > 10:
        warnings.append(f"CLASS IMBALANCE: largest/smallest class ratio is {imbalance}x; "
                        f"class weighting is used in training, and macro metrics matter more than accuracy.")
    present_groups = {c.group for c in mapping.classes if per_class.get(c.key, 0) > 0}
    for g in ("wheat", "rice"):
        if g not in present_groups:
            warnings.append(f"NO {g.upper()} DATA: none of the 4 {g} classes has images (source not available "
                            f"or its labels are not confirmed).")
    if "plantvillage" in present_groups and present_groups & {"wheat", "rice"}:
        warnings.append("DOMAIN SHORTCUT RISK: PlantVillage photos are single leaves on uniform lab backgrounds, "
                        "while wheat/rice sources usually are not. A model can separate datasets by background "
                        "instead of by disease; test accuracy then overstates real-world performance.")
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software_version": __version__,
        "class_mapping_version": mapping.version, "class_mapping_sha256": mapping.sha256(),
        "config": {"seed": cfg.seed, "image_size": cfg.image_size, "val_frac": cfg.val_frac,
                   "test_frac": cfg.test_frac, "near_duplicate_threshold_bits_of_256": cfg.phash_threshold,
                   "include_unconfirmed_sources": cfg.include_unconfirmed, "jpeg_quality": JPEG_QUALITY},
        "sources": source_summaries,
        "totals": {"usable_images": len(records), "classes_with_images": len(nonzero),
                   "classes_required": mapping.num_classes,
                   "corrupt_or_unreadable": len(corrupt),
                   "exact_duplicates_removed": sum(1 for d in dropped if d["reason"] == "exact_duplicate"),
                   "label_conflicts_removed": sum(1 for d in dropped if d["reason"] == "label_conflict"),
                   "excluded_unmapped_images": sum(e["images"] for e in excluded),
                   "write_failures": len(write_failures),
                   "split_sizes": {s: sum(split_counts[s].values()) for s in SPLITS},
                   "imbalance_ratio_max_over_min": imbalance},
        "near_duplicate_groups": groups_info,
        "classes": rows,
        "excluded_labels": excluded,
        "warnings": warnings,
    }


def write_reports(cfg: PrepConfig, report: dict, corrupt: List[dict], dropped: List[dict],
                  write_failures: List[dict]) -> None:
    d = cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def dump_csv(name: str, rows: List[dict], fields: List[str]) -> None:
        with open(d / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    dump_csv("failed_images.csv", corrupt + write_failures, ["source_id", "path", "reason"])
    dump_csv("duplicates_removed.csv", dropped, ["path", "class", "reason", "duplicate_of"])
    dump_csv("excluded_labels.csv", report["excluded_labels"], ["source_id", "source_label", "images", "reason"])

    t = report["totals"]
    lines = ["# Dataset preparation report", "",
             f"Generated: {report['generated_utc']}  |  class mapping v{report['class_mapping_version']}", "",
             "## Summary", "",
             f"- Usable images: **{t['usable_images']}** in **{t['classes_with_images']} / {t['classes_required']}** classes",
             f"- Split sizes: train {t['split_sizes']['train']}, val {t['split_sizes']['val']}, test {t['split_sizes']['test']}",
             f"- Corrupt / unreadable: {t['corrupt_or_unreadable']}; exact duplicates removed: {t['exact_duplicates_removed']}; "
             f"label conflicts removed: {t['label_conflicts_removed']}; excluded (unmapped labels): {t['excluded_unmapped_images']}",
             f"- Class imbalance (largest / smallest): {t['imbalance_ratio_max_over_min']}",
             f"- Seed {report['config']['seed']}, image size {report['config']['image_size']}, "
             f"near-duplicate threshold {report['config']['near_duplicate_threshold_bits_of_256']}/256 bits", "",
             "## Sources", ""]
    for s in report["sources"]:
        lines.append(f"- `{s['source_id']}`: found={s['found']}, images={s['images']}, used={s['used']} {s.get('note', '')}")
    lines += ["", "## Warnings", ""] + ([f"- {w}" for w in report["warnings"]] or ["- none"])
    lines += ["", "## Per-class counts", "", "| # | class | total | train | val | test |", "|---|---|---|---|---|---|"]
    for r in report["classes"]:
        lines.append(f"| {r['index']} | {r['class_key']} | {r['total']} | {r['train']} | {r['val']} | {r['test']} |")
    lines += ["", "## Near-duplicate groups", "", f"```\n{json.dumps(report['near_duplicate_groups'], indent=2)}\n```", ""]
    (d / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")
    plot_class_distribution(report, d / "class_distribution.png")


def plot_class_distribution(report: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["classes"]
    names = [f"{r['index']:02d} {r['class_key'][:34]}" for r in rows]
    fig, ax = plt.subplots(figsize=(11, 12))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for s, color in zip(SPLITS, ("#2e7d32", "#f9a825", "#c62828")):
        vals = np.array([r[s] for r in rows], dtype=float)
        ax.barh(y, vals, left=left, label=s, color=color)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_title("Images per class and split (46-class taxonomy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------
def run(cfg: PrepConfig, log=print) -> dict:
    mapping = load_mapping()
    log("[1/6] discovering images and mapping labels ...")
    records, excluded, source_summaries = discover(cfg, mapping)
    log(f"      {len(records)} candidate images; {sum(e['images'] for e in excluded)} excluded by label")
    if not records:
        raise SystemExit("No usable images were found. Run:  python -m src.data_download status")

    log("[2/6] validating images and computing hashes ...")
    inspect_all(records, cfg, log)
    corrupt = quarantine(records, cfg)
    good = [r for r in records if r.status == "ok"]
    log(f"      {len(good)} valid, {len(corrupt)} corrupt/unreadable (quarantined copies in data/interim/quarantine)")

    log("[3/6] removing exact duplicates and clustering near-duplicates ...")
    good, dropped = remove_exact_duplicates(good)
    groups, refused_unions = cluster_near_duplicates([r.hashes for r in good], cfg.phash_threshold)
    for r, g in zip(good, groups):
        r.group = g
    sizes = Counter(groups)
    multi = [g for g, n in sizes.items() if n > 1]
    mixed = 0
    by_group: Dict[int, set] = defaultdict(set)
    for r in good:
        by_group[r.group].add(r.class_key)
    mixed = sum(1 for g in multi if len(by_group[g]) > 1)
    groups_info = {"groups": len(sizes), "groups_with_multiple_images": len(multi),
                   "images_in_multi_groups": sum(sizes[g] for g in multi),
                   "largest_group": max(sizes.values()) if sizes else 0,
                   "groups_spanning_multiple_classes": mixed,
                   "unions_refused_by_size_cap": refused_unions, "max_group_size_cap": MAX_GROUP_SIZE}
    log(f"      removed {len(dropped)} duplicates; {len(multi)} near-duplicate groups "
        f"({groups_info['images_in_multi_groups']} images); {mixed} span >1 class (possible label noise)")

    log("[4/6] stratified group split ...")
    split_counts = assign_splits(good, cfg.val_frac, cfg.test_frac, cfg.seed)

    log("[5/6] writing processed dataset ...")
    write_failures = write_processed(good, cfg, mapping, log)

    log("[6/6] writing reports ...")
    report = build_report(cfg, mapping, good, split_counts, excluded, source_summaries, corrupt,
                          dropped, write_failures, groups_info)
    write_reports(cfg, report, corrupt, dropped, write_failures)
    (cfg.processed_dir / "prep_config.json").write_text(json.dumps(report["config"], indent=2), encoding="utf-8")
    log("\nDone. Reports: " + str(cfg.reports_dir / "dataset_report.md"))
    for w in report["warnings"]:
        log("  WARNING: " + w)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.preprocess", description="Prepare the 46-class dataset.")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--phash-threshold", type=int, default=10,
                   help="max differing bits (of 256, range 0-32) to call two images near-duplicates")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--include-unconfirmed", action="store_true",
                   help="include sources whose labels you have not confirmed (NOT recommended)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if not 0 <= a.phash_threshold <= 32:
        print("--phash-threshold must be between 0 and 32", file=sys.stderr)
        return 2
    if not 0 < a.val_frac + a.test_frac < 0.8:
        print("--val-frac + --test-frac must be between 0 and 0.8", file=sys.stderr)
        return 2
    cfg = PrepConfig(raw_dir=a.raw_dir, interim_dir=a.interim_dir, processed_dir=a.processed_dir,
                     reports_dir=a.reports_dir, image_size=a.image_size, seed=a.seed, val_frac=a.val_frac,
                     test_frac=a.test_frac, phash_threshold=a.phash_threshold, workers=a.workers,
                     sources=a.sources, include_unconfirmed=a.include_unconfirmed)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
