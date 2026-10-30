"""The single authoritative class taxonomy (46 classes).

Everything - preprocessing, training, evaluation, prediction, the Gemini context
and the reports - must obtain class names/indices from this module, which reads
``configs/class_mapping.json``. Nothing else should hard-code class lists.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import CLASS_MAPPING_PATH

EXPECTED_NUM_CLASSES = 46
EXPECTED_GROUP_COUNTS = {"plantvillage": 38, "wheat": 4, "rice": 4}


@dataclass(frozen=True)
class ClassInfo:
    index: int
    key: str            # stable machine identifier, also used as folder name
    group: str          # plantvillage | wheat | rice
    plant: str
    condition: str
    display_name: str
    is_healthy: bool


class MappingError(ValueError):
    """The class mapping file is missing, malformed or inconsistent."""


def normalise_label(raw: str) -> str:
    """Lowercase; collapse every run of non-alphanumerics to '_'; trim '_'."""
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


class ClassMapping:
    def __init__(self, doc: dict, path: Optional[Path] = None):
        self.doc = doc
        self.path = path
        self.version: str = doc.get("version", "unknown")
        self.classes: List[ClassInfo] = [
            ClassInfo(
                index=c["index"], key=c["key"], group=c["group"], plant=c["plant"],
                condition=c["condition"], display_name=c["display_name"],
                is_healthy=bool(c["is_healthy"]),
            )
            for c in doc["classes"]
        ]
        self._by_key: Dict[str, ClassInfo] = {c.key: c for c in self.classes}
        self._source_labels: Dict[str, Dict[str, str]] = {}
        for c in doc["classes"]:
            for source_id, labels in c.get("source_labels", {}).items():
                table = self._source_labels.setdefault(source_id, {})
                for lab in labels:
                    # PlantVillage labels are matched verbatim; others after normalisation.
                    table[lab if source_id == "plantvillage" else normalise_label(lab)] = c["key"]
        self.ambiguous = {k: {normalise_label(x) for x in v}
                          for k, v in doc.get("ambiguous_source_labels", {}).items() if not k.startswith("_")}
        self.known_unsupported = {k: {normalise_label(x) for x in v}
                                  for k, v in doc.get("known_unsupported_source_labels", {}).items()
                                  if not k.startswith("_")}

    # ---- basic accessors ---------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def keys(self) -> List[str]:
        """Ordered class keys == model output order."""
        return [c.key for c in self.classes]

    def by_index(self, index: int) -> ClassInfo:
        if not 0 <= index < self.num_classes:
            raise IndexError(f"class index {index} outside 0..{self.num_classes - 1}")
        return self.classes[index]

    def by_key(self, key: str) -> ClassInfo:
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown class key: {key!r}") from exc

    def index_of(self, key: str) -> int:
        return self.by_key(key).index

    def sha256(self) -> str:
        """Fingerprint of the ordered keys; stored in model metadata so a model can
        never be paired with a different label order."""
        return hashlib.sha256("\n".join(self.keys).encode("utf-8")).hexdigest()

    # ---- source label resolution -------------------------------------------
    def map_source_label(self, source_id: str, raw_label: str) -> Tuple[Optional[str], str]:
        """Map a source dataset's folder label to a canonical class key.

        Returns ``(class_key_or_None, status)`` where status is one of
        ``mapped``, ``ambiguous``, ``unsupported`` or ``unknown``.
        Unmapped labels are excluded by the pipeline - never guessed.
        """
        table = self._source_labels.get(source_id, {})
        if source_id == "plantvillage":
            if raw_label in table:
                return table[raw_label], "mapped"
            return None, "unknown"
        norm = normalise_label(raw_label)
        if norm in table:
            return table[norm], "mapped"
        if norm in self.ambiguous.get(source_id, set()):
            return None, "ambiguous"
        if norm in self.known_unsupported.get(source_id, set()):
            return None, "unsupported"
        return None, "unknown"

    def validate(self) -> None:
        """Raise MappingError unless the mapping is internally consistent."""
        n = len(self.classes)
        if n != EXPECTED_NUM_CLASSES:
            raise MappingError(f"expected {EXPECTED_NUM_CLASSES} classes, found {n}")
        if self.doc.get("num_classes") != n:
            raise MappingError("'num_classes' field disagrees with the class list")
        if [c.index for c in self.classes] != list(range(n)):
            raise MappingError("class indices must be contiguous 0..N-1 in file order")
        if len({c.key for c in self.classes}) != n:
            raise MappingError("duplicate class keys")
        if len({c.display_name for c in self.classes}) != n:
            raise MappingError("duplicate display names")
        counts: Dict[str, int] = {}
        for c in self.classes:
            counts[c.group] = counts.get(c.group, 0) + 1
        if counts != EXPECTED_GROUP_COUNTS:
            raise MappingError(f"group counts {counts} != {EXPECTED_GROUP_COUNTS}")
        # An alias may belong to exactly one class within a source.
        for source_id in self._source_labels:
            seen: Dict[str, str] = {}
            for c in self.doc["classes"]:
                for lab in c.get("source_labels", {}).get(source_id, []):
                    k = lab if source_id == "plantvillage" else normalise_label(lab)
                    if k in seen and seen[k] != c["key"]:
                        raise MappingError(f"alias {k!r} of source {source_id!r} maps to two classes")
                    seen[k] = c["key"]
            overlap = set(seen) & self.ambiguous.get(source_id, set())
            if overlap:
                raise MappingError(f"labels both mapped and ambiguous in {source_id}: {sorted(overlap)}")


@lru_cache(maxsize=4)
def _load(path_str: str) -> ClassMapping:
    path = Path(path_str)
    if not path.is_file():
        raise MappingError(f"class mapping not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MappingError(f"class mapping is not valid JSON: {exc}") from exc
    mapping = ClassMapping(doc, path)
    mapping.validate()
    return mapping


def load_mapping(path: Optional[Path] = None) -> ClassMapping:
    """Load and validate the (cached) class mapping."""
    return _load(str(path or CLASS_MAPPING_PATH))
