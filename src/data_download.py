"""Dataset acquisition: check, download (where legitimately possible) and record.

Sub-commands (run from the project root)::

    python -m src.data_download status
    python -m src.data_download fetch [--source plantvillage] [--sample-per-class N]
    python -m src.data_download ingest --source SRC --archive PATH
    python -m src.data_download review-labels --source SRC
    python -m src.data_download confirm-labels --source SRC
    python -m src.data_download all

Principles
* Existing, verified local data is reused - nothing is downloaded twice.
* Only sources with a direct, unauthenticated download are fetched automatically.
  Sources that need registration or acceptance of terms are *never* bypassed; the
  tool prints exact manual instructions and can then ingest the archive you
  downloaded yourself.
* Every acquisition is recorded in ``data/raw/provenance.json`` (URL, access date,
  commit/version, licence note, verified image counts).
* A dataset is only called "downloaded" after its files were counted and checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from .config import DATASET_SOURCES_PATH, RAW_DIR
from .discovery import count_by_label
from .labels import load_mapping

PROVENANCE_FILE = "provenance.json"
SOURCE_MANIFEST = ".source_manifest.json"
LABELS_CONFIRMED_FILE = ".labels_confirmed.json"
MAX_EXTRACT_BYTES = 40 * 1024 ** 3        # refuse archives that expand beyond 40 GB
MAX_EXPANSION_RATIO = 300                 # ... or more than 300x their compressed size


class AcquisitionError(RuntimeError):
    """A dataset could not be fetched, extracted or verified."""


# ----------------------------------------------------------------------------
# registry helpers
# ----------------------------------------------------------------------------
def load_sources(path: Path = DATASET_SOURCES_PATH) -> Dict[str, dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc["sources"]


def source_dir(source_id: str, raw_dir: Path = RAW_DIR, sources: Optional[Dict[str, dict]] = None) -> Path:
    sources = sources or load_sources()
    if source_id not in sources:
        raise AcquisitionError(f"unknown source {source_id!r}; known: {', '.join(sources)}")
    return Path(raw_dir) / sources[source_id]["raw_subdir"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def record_provenance(source_id: str, entry: dict, raw_dir: Path = RAW_DIR) -> None:
    path = Path(raw_dir) / PROVENANCE_FILE
    doc = _read_json(path) or {"sources": {}}
    doc.setdefault("sources", {})[source_id] = entry
    _write_json_atomic(path, doc)


# ----------------------------------------------------------------------------
# status / verification
# ----------------------------------------------------------------------------
def labels_confirmed(source_id: str, raw_dir: Path = RAW_DIR, sources: Optional[Dict[str, dict]] = None) -> bool:
    """True when the source is pre-verified in the registry or a human confirmed it locally."""
    sources = sources or load_sources()
    if sources[source_id].get("labels_confirmed"):
        return True
    marker = source_dir(source_id, raw_dir, sources) / LABELS_CONFIRMED_FILE
    return bool(_read_json(marker).get("confirmed"))


def source_status(source_id: str, raw_dir: Path = RAW_DIR, sources: Optional[Dict[str, dict]] = None) -> dict:
    """Inspect what is actually on disk for a source (never trusts markers alone)."""
    sources = sources or load_sources()
    src = sources[source_id]
    root = source_dir(source_id, raw_dir, sources)
    counts = count_by_label(root) if root.is_dir() else {}
    total = sum(counts.values())
    manifest = _read_json(root / SOURCE_MANIFEST) if root.is_dir() else {}
    expected = src.get("expected_images")
    expected_dirs = src.get("expected_class_folders")
    if total == 0:
        state = "missing"
    elif manifest.get("partial_sample"):
        state = "partial-sample"
    elif expected and total < expected:
        state = "incomplete"
    else:
        state = "present"
    mapping = load_mapping()
    mapped = {lab: mapping.map_source_label(source_id, lab) for lab in counts}
    return {
        "source_id": source_id, "state": state, "path": str(root),
        "image_count": total, "label_folders": len(counts),
        "expected_images": expected, "expected_class_folders": expected_dirs,
        "labels_confirmed": labels_confirmed(source_id, raw_dir, sources),
        "label_counts": counts,
        "label_mapping": {lab: {"class_key": k, "status": st} for lab, (k, st) in mapped.items()},
        "method": src["method"],
    }


def print_status(raw_dir: Path = RAW_DIR) -> None:
    sources = load_sources()
    print(f"Raw data directory: {raw_dir}\n")
    for sid, src in sources.items():
        st = source_status(sid, raw_dir, sources)
        flag = "confirmed" if st["labels_confirmed"] else "NOT confirmed"
        print(f"[{sid}] {src['name']}")
        print(f"    state: {st['state']:<14} images: {st['image_count']}  label folders: {st['label_folders']}  "
              f"labels: {flag}")
        if st["state"] in ("missing", "partial-sample", "incomplete"):
            if src["method"] == "manual":
                print("    -> manual download required:")
                for line in src["manual_instructions"]:
                    print("       " + line)
            else:
                print(f"    -> run: python -m src.data_download fetch --source {sid}")
        print()


# ----------------------------------------------------------------------------
# safe archive extraction
# ----------------------------------------------------------------------------
def _is_unsafe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or (len(name) > 1 and name[1] == ":"):
        return True
    return ".." in p.parts


def safe_extract(archive: Path, dest: Path) -> int:
    """Extract a .zip / .tar / .tar.gz into ``dest`` refusing path traversal, links,
    zip bombs and corrupt archives. Returns the number of files written."""
    archive, dest = Path(archive), Path(dest)
    if not archive.is_file():
        raise AcquisitionError(f"archive not found: {archive}")
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    written = 0

    def target_for(name: str) -> Path:
        if _is_unsafe_member(name):
            raise AcquisitionError(f"unsafe path inside archive: {name!r}")
        target = (dest / name).resolve()
        if dest_resolved != target and dest_resolved not in target.parents:
            raise AcquisitionError(f"archive member escapes the target directory: {name!r}")
        return target

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise AcquisitionError(f"corrupt ZIP (first bad member: {bad}); download it again")
            total = sum(i.file_size for i in zf.infolist())
            if total > MAX_EXTRACT_BYTES or total > max(archive.stat().st_size, 1) * MAX_EXPANSION_RATIO:
                raise AcquisitionError("archive expands suspiciously large (possible zip bomb); refusing")
            for info in zf.infolist():
                if info.is_dir() or "__MACOSX" in info.filename or info.filename.endswith(".DS_Store"):
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise AcquisitionError(f"symbolic link inside archive: {info.filename!r}")
                target = target_for(info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                written += 1
        return written

    if tarfile.is_tarfile(archive):
        try:
            with tarfile.open(archive) as tf:
                members = tf.getmembers()
                total = sum(m.size for m in members if m.isfile())
                if total > MAX_EXTRACT_BYTES or total > max(archive.stat().st_size, 1) * MAX_EXPANSION_RATIO:
                    raise AcquisitionError("archive expands suspiciously large; refusing")
                for m in members:
                    if m.isdir() or "__MACOSX" in m.name:
                        continue
                    if not m.isfile():
                        raise AcquisitionError(f"link/device entry inside archive: {m.name!r}")
                    target = target_for(m.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        continue
                    with fobj, open(target, "wb") as out:
                        shutil.copyfileobj(fobj, out)
                    written += 1
        except (tarfile.TarError, EOFError, OSError) as exc:
            raise AcquisitionError(f"corrupt or truncated tar archive: {exc}") from exc
        return written

    raise AcquisitionError("unsupported archive type (expected .zip, .tar or .tar.gz)")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ingest_archive(source_id: str, archive: Path, raw_dir: Path = RAW_DIR,
                   sources: Optional[Dict[str, dict]] = None) -> dict:
    """Extract an archive that the user downloaded manually into the source folder."""
    sources = sources or load_sources()
    src = sources[source_id]
    root = source_dir(source_id, raw_dir, sources)
    archive = Path(archive)
    digest = sha256_file(archive) if archive.is_file() else None
    n_files = safe_extract(archive, root)
    status = source_status(source_id, raw_dir, sources)
    if status["image_count"] == 0:
        raise AcquisitionError("the archive extracted but no JPG/JPEG/PNG images were found inside "
                               "<label-folder>/ directories; check the archive layout")
    _write_json_atomic(root / SOURCE_MANIFEST, {
        "complete": True, "partial_sample": False, "image_count": status["image_count"],
        "label_folders": status["label_folders"], "method": "manual-archive",
        "archive_name": archive.name, "archive_sha256": digest, "ingested_utc": _utc_now()})
    record_provenance(source_id, {
        "name": src["name"], "url": src["landing_url"], "method": "manual-archive",
        "accessed_utc": _utc_now(), "archive_name": archive.name, "archive_sha256": digest,
        "files_extracted": n_files, "verified_image_count": status["image_count"],
        "label_folders": status["label_folders"], "license": src["license"], "citation": src.get("citation"),
        "labels_confirmed": status["labels_confirmed"]}, raw_dir)
    return status


# ----------------------------------------------------------------------------
# generic HTTP download (for sources that publish a direct, public file URL)
# ----------------------------------------------------------------------------
def download_file(url: str, dest: Path, expected_sha256: Optional[str] = None, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest`` via a ``.part`` file; verify length and checksum."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
            expected_len = resp.headers.get("Content-Length")
            shutil.copyfileobj(resp, out)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise AcquisitionError(f"download failed: {exc}") from exc
    if expected_len and part.stat().st_size != int(expected_len):
        part.unlink(missing_ok=True)
        raise AcquisitionError("incomplete download (size mismatch); try again")
    if expected_sha256 and sha256_file(part) != expected_sha256.lower():
        part.unlink(missing_ok=True)
        raise AcquisitionError("checksum mismatch; the file is corrupt or has changed")
    os.replace(part, dest)
    return dest


# ----------------------------------------------------------------------------
# PlantVillage via a pinned, partial git checkout
# ----------------------------------------------------------------------------
def _git(args: List[str], cwd: Optional[Path] = None, input_text: Optional[str] = None) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, input=input_text, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise AcquisitionError(
            "git is not installed or not on PATH. Install Git for Windows (https://git-scm.com/download/win) "
            "or download PlantVillage manually and use:  python -m src.data_download ingest ...") from exc
    if proc.returncode != 0:
        raise AcquisitionError(f"git {' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def _rmtree_force(path: Path) -> None:
    def onerror(func, p, _exc):  # git pack files are read-only on Windows
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if Path(path).exists():
        shutil.rmtree(path, onerror=onerror)


def _gitignore_escape(path: str) -> str:
    out = []
    for ch in path:
        if ch in "*?[]\\":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "/" + "".join(out).lstrip("/")


def fetch_plantvillage(raw_dir: Path = RAW_DIR, sample_per_class: Optional[int] = None,
                       force: bool = False, sources: Optional[Dict[str, dict]] = None) -> dict:
    """Fetch the PlantVillage colour images from the pinned GitHub commit (full set or a
    deterministic per-class sample). Reuses existing verified data unless ``force``."""
    sources = sources or load_sources()
    src = sources["plantvillage"]
    root = source_dir("plantvillage", raw_dir, sources)
    status = source_status("plantvillage", raw_dir, sources)
    if status["state"] == "present" and not force:
        print(f"[plantvillage] already present ({status['image_count']} images) - skipping download.")
        return status
    if status["state"] == "partial-sample" and sample_per_class and not force:
        print("[plantvillage] a sample is already present - skipping (use --force to refetch).")
        return status

    tmp = Path(raw_dir) / "_git_tmp"
    _rmtree_force(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print("[plantvillage] cloning repository metadata (no image data yet)...")
    try:
        _git(["clone", "--quiet", "--depth", "1", "--filter=blob:none", "--no-checkout",
              src["repo_url"], str(tmp)])
        head = _git(["rev-parse", "HEAD"], cwd=tmp).strip()
        pinned = src["pinned_commit"]
        if head != pinned:
            try:
                _git(["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", pinned], cwd=tmp)
                head = pinned  # object is now available locally; checked out after sparse setup
            except AcquisitionError:
                print(f"[plantvillage] WARNING: pinned commit {pinned[:10]} unavailable; "
                      f"using repository HEAD {head[:10]} instead (recorded in provenance).")
        sparse = src["sparse_path"]
        if sample_per_class:
            files = _git(["ls-tree", "-r", "--name-only", head, sparse], cwd=tmp).splitlines()
            seen: Dict[str, int] = {}
            picked = []
            for f in sorted(files):
                label = f.split("/")[-2]
                seen[label] = seen.get(label, 0) + 1
                if seen[label] <= sample_per_class:
                    picked.append(_gitignore_escape(f))
            _git(["sparse-checkout", "set", "--no-cone", "--stdin"], cwd=tmp, input_text="\n".join(picked) + "\n")
        else:
            _git(["sparse-checkout", "set", "--cone", sparse], cwd=tmp)
        print("[plantvillage] downloading image files" +
              (f" ({sample_per_class} per class)" if sample_per_class else " (~1 GB, this takes a while)") + "...")
        _git(["checkout", "--quiet", "--detach", head], cwd=tmp)
        checked_out = tmp / sparse
        if not checked_out.is_dir():
            raise AcquisitionError("git checkout finished but the image folder is missing")
        if root.exists():
            _rmtree_force(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(checked_out), str(root))
    finally:
        _rmtree_force(tmp)

    status = source_status("plantvillage", raw_dir, sources)
    if status["image_count"] == 0:
        raise AcquisitionError("no images found after download")
    partial = bool(sample_per_class)
    if not partial:
        if status["label_folders"] != src["expected_class_folders"]:
            raise AcquisitionError(f"expected {src['expected_class_folders']} class folders, "
                                   f"found {status['label_folders']}")
        if status["image_count"] < src["expected_images"]:
            raise AcquisitionError(f"incomplete download: {status['image_count']} of "
                                   f"{src['expected_images']} images; run again with --force")
    _write_json_atomic(root / SOURCE_MANIFEST, {
        "complete": not partial, "partial_sample": partial, "sample_per_class": sample_per_class,
        "image_count": status["image_count"], "label_folders": status["label_folders"],
        "method": "git_sparse", "commit": head, "fetched_utc": _utc_now()})
    record_provenance("plantvillage", {
        "name": src["name"], "url": src["landing_url"], "method": "git_sparse",
        "repo_url": src["repo_url"], "commit": head, "accessed_utc": _utc_now(),
        "partial_sample": partial, "sample_per_class": sample_per_class,
        "verified_image_count": status["image_count"], "label_folders": status["label_folders"],
        "license": src["license"], "citation": src["citation"], "labels_confirmed": True}, raw_dir)
    status = source_status("plantvillage", raw_dir, sources)   # re-read now that the manifest exists
    print(f"[plantvillage] verified {status['image_count']} images in {status['label_folders']} class folders"
          f"{' (partial sample)' if partial else ''}.")
    return status


# ----------------------------------------------------------------------------
# label review / confirmation (guards the wheat & rice mapping)
# ----------------------------------------------------------------------------
def review_labels(source_id: str, raw_dir: Path = RAW_DIR) -> dict:
    st = source_status(source_id, raw_dir)
    print(f"Label review for '{source_id}'  ({st['image_count']} images, {st['label_folders']} folders)\n")
    if not st["label_folders"]:
        print("No images found. Follow the manual download instructions first (python -m src.data_download status).")
        return st
    print(f"{'source folder':<40}{'images':>8}   {'-> canonical class':<34}{'status'}")
    for lab, n in st["label_counts"].items():
        info = st["label_mapping"][lab]
        print(f"{lab:<40}{n:>8}   {str(info['class_key'] or '-'):<34}{info['status']}")
    mapped_classes = {v["class_key"] for v in st["label_mapping"].values() if v["class_key"]}
    mapping = load_mapping()
    wanted = {c.key for c in mapping.classes
              if source_id in mapping.doc["classes"][c.index].get("source_labels", {})}
    missing = sorted(wanted - mapped_classes)
    print("\nCanonical classes this source should supply but does not yet map to:", missing or "none")
    print("Folders with status ambiguous/unsupported/unknown are EXCLUDED, never guessed.")
    print("If a mapping is wrong or missing, edit 'source_labels' in configs/class_mapping.json.")
    return st


def confirm_labels(source_id: str, raw_dir: Path = RAW_DIR) -> None:
    """Record that a human reviewed this source's labels against its documentation."""
    sources = load_sources()
    root = source_dir(source_id, raw_dir, sources)
    st = source_status(source_id, raw_dir, sources)
    if st["image_count"] == 0:
        raise AcquisitionError("nothing to confirm: no images found for this source")
    _write_json_atomic(root / LABELS_CONFIRMED_FILE, {
        "confirmed": True, "confirmed_utc": _utc_now(),
        "mapping": {k: v for k, v in st["label_mapping"].items()}, "note": "confirmed by the user via the CLI"})
    print(f"Recorded label confirmation for '{source_id}'. Preprocessing will now include its mapped folders.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def fetch_all(raw_dir: Path = RAW_DIR, sample_per_class: Optional[int] = None) -> None:
    sources = load_sources()
    for sid, src in sources.items():
        if src["method"] == "git_sparse":
            try:
                fetch_plantvillage(raw_dir, sample_per_class)
            except AcquisitionError as exc:
                print(f"[{sid}] FAILED: {exc}")
        else:
            st = source_status(sid, raw_dir, sources)
            if st["state"] == "missing":
                print(f"[{sid}] requires a manual download (terms/registration); instructions:")
                for line in src["manual_instructions"]:
                    print("    " + line)
            else:
                print(f"[{sid}] found locally: {st['image_count']} images in {st['label_folders']} folders.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.data_download", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show what is available locally")
    f = sub.add_parser("fetch", help="download accessible sources")
    f.add_argument("--source", default="plantvillage")
    f.add_argument("--sample-per-class", type=int, default=None,
                   help="download only N images per class (quick trial / low bandwidth)")
    f.add_argument("--force", action="store_true")
    i = sub.add_parser("ingest", help="extract an archive you downloaded manually")
    i.add_argument("--source", required=True)
    i.add_argument("--archive", required=True)
    r = sub.add_parser("review-labels", help="show how source folders map to canonical classes")
    r.add_argument("--source", required=True)
    c = sub.add_parser("confirm-labels", help="record that you verified the labels of a manual source")
    c.add_argument("--source", required=True)
    a = sub.add_parser("all", help="status + fetch accessible sources + manual instructions")
    a.add_argument("--sample-per-class", type=int, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "status":
            print_status()
        elif args.cmd == "fetch":
            sources = load_sources()
            if args.source not in sources:
                raise AcquisitionError(f"unknown source {args.source!r}")
            if sources[args.source]["method"] != "git_sparse":
                raise AcquisitionError(f"'{args.source}' cannot be downloaded automatically. Instructions:\n  " +
                                       "\n  ".join(sources[args.source]["manual_instructions"]))
            fetch_plantvillage(sample_per_class=args.sample_per_class, force=args.force)
        elif args.cmd == "ingest":
            st = ingest_archive(args.source, Path(args.archive))
            print(f"Ingested: {st['image_count']} images in {st['label_folders']} folders.")
            print("Next: python -m src.data_download review-labels --source " + args.source)
        elif args.cmd == "review-labels":
            review_labels(args.source)
        elif args.cmd == "confirm-labels":
            confirm_labels(args.source)
        elif args.cmd == "all":
            fetch_all(sample_per_class=args.sample_per_class)
            print()
            print_status()
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
