"""Shared, side-effect-free helpers for the mower vision command line tools."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ultralytics must never install packages as a side effect of these scripts.
os.environ["YOLO_AUTOINSTALL"] = "False"
CLASSES = ["stone", "tree", "person", "other_roadblock"]
SPLITS = ("train", "val", "test")


def resolve(path: str | Path, base: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def read_yaml(path: str | Path) -> dict:
    import yaml
    with resolve(path).open(encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层必须是映射: {path}")
    return value


def dataset_paths(data_yaml: str | Path) -> tuple[Path, dict]:
    cfg = read_yaml(data_yaml)
    yaml_path = resolve(data_yaml)
    root = resolve(cfg["path"], yaml_path.parent)
    return root, cfg


def class_names(cfg: dict) -> list[str]:
    raw = cfg.get("names")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw.get(i, raw.get(str(i))) for i in range(len(raw))]
    return []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(data_yaml: str | Path) -> dict:
    """Fingerprint immutable metadata and image/label contents, without changing them."""
    root, cfg = dataset_paths(data_yaml)
    h = hashlib.sha256()
    counts = {}
    for split in SPLITS:
        images = sorted((root / cfg[split]).glob("*"))
        counts[split] = len([p for p in images if p.is_file()])
        for directory in (root / cfg[split], root / "labels" / split):
            for p in sorted(x for x in directory.glob("*") if x.is_file()):
                h.update(str(p.relative_to(root)).replace("\\", "/").encode())
                h.update(bytes.fromhex(sha256(p)))
    for name in ("manifest.csv", "split_manifest.csv", "dataset_info.json"):
        p = root / name
        if p.exists():
            h.update(name.encode())
            h.update(bytes.fromhex(sha256(p)))
    return {"sha256": h.hexdigest(), "split_counts": counts,
            "split_manifest": str(root / "split_manifest.csv")}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    result = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "numpy", "PyYAML"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "missing"
    return result


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
