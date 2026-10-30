"""Find image files in a raw dataset tree and derive their source label.

The source label of an image is the name of the folder that directly contains it
(the usual ``<label>/<image>.jpg`` layout). Folders such as ``grayscale`` and
``segmented`` (alternative renderings of the same PlantVillage photos) are skipped.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Tuple

from .config import SUPPORTED_EXTENSIONS

SKIP_DIR_NAMES = {"grayscale", "greyscale", "segmented", "__macosx", ".git", "_git_tmp"}


def iter_images(root: Path) -> Iterator[Tuple[Path, str]]:
    """Yield ``(path, source_label)`` for every supported image under ``root``."""
    root = Path(root)
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in SKIP_DIR_NAMES)
        parent = Path(dirpath)
        if parent == root:
            continue  # images lying directly in the root have no label
        for name in sorted(filenames):
            if name.startswith("._"):
                continue
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                yield parent / name, parent.name


def count_by_label(root: Path) -> Dict[str, int]:
    counts: Counter = Counter()
    for _, label in iter_images(root):
        counts[label] += 1
    return dict(sorted(counts.items()))
