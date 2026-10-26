"""Make a checked, independent YOLO dataset snapshot. Never write to the export."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from common import CLASSES, ROOT, SPLITS, fingerprint, resolve, sha256, timestamp, write_json
from validate_dataset import IMAGE_EXTS, validate

SOURCE = Path(r"E:\workspace\mower-annotation-exports")
TARGET = ROOT / "datasets" / "mower_v1_snapshot"
METADATA = ("manifest.csv", "split_manifest.csv", "dataset_info.json", "export_report.txt", "data.yaml")


def expected_files(source: Path) -> list[Path]:
    files = []
    for split in SPLITS:
        files.extend(p.relative_to(source) for p in (source / "images" / split).iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        files.extend(p.relative_to(source) for p in (source / "labels" / split).glob("*.txt"))
    files.extend(Path(name) for name in METADATA)
    return sorted(files, key=lambda p: p.as_posix())


def snapshot_yaml(target: Path) -> str:
    return (f"path: {target.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "nc: 4\nnames:\n  0: stone\n  1: tree\n  2: person\n  3: other_roadblock\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(TARGET))
    args = ap.parse_args()
    source, target = resolve(args.source), resolve(args.out)
    # V1 has one authoritative export and one designated snapshot location.
    if source != SOURCE or target != TARGET or source == target:
        ap.error(f"V1 仅允许从 {SOURCE} 复制到 {TARGET}")
    if not source.is_dir():
        ap.error(f"源目录不存在: {source}")
    audit = validate(ROOT / "configs" / "datasets" / "mower_v1.yaml")
    errors = [x for x in audit["issues"] if x["level"] == "ERROR"]
    if errors:
        ap.error(f"源数据审查存在 {len(errors)} 个 ERROR；拒绝复制")
    files = expected_files(source)
    if any((source / rel).is_symlink() for rel in files):
        ap.error("源目录含符号链接；拒绝复制")
    source_hashes = {rel.as_posix(): sha256(source / rel) for rel in files}
    source_fingerprint = fingerprint(ROOT / "configs" / "datasets" / "mower_v1.yaml")["sha256"]
    # An existing snapshot is accepted only when every required file and its
    # generated YAML agree. Its cache files may differ and are never touched.
    if target.exists():
        if not target.is_dir():
            ap.error(f"目标已存在且不是目录: {target}")
        mismatch = []
        for name, digest in source_hashes.items():
            copied = target / ("metadata/original_data.yaml" if name == "data.yaml" else name)
            if not copied.is_file() or sha256(copied) != digest:
                mismatch.append(name)
        for split in SPLITS:
            for subdir, suffixes in (("images", IMAGE_EXTS), ("labels", {".txt"})):
                wanted = {name for name in source_hashes if name.startswith(f"{subdir}/{split}/")}
                actual = {p.relative_to(target).as_posix() for p in (target / subdir / split).iterdir()
                          if p.is_file() and p.suffix.lower() in suffixes}
                mismatch.extend(sorted(actual - wanted))
        yaml_path = target / "data.yaml"
        if not yaml_path.is_file() or yaml_path.read_text(encoding="utf-8") != snapshot_yaml(target):
            mismatch.append("data.yaml")
        report_path = target / "snapshot_report.json"
        if (not report_path.is_file() or
            json.loads(report_path.read_text(encoding="utf-8")).get("dataset_fingerprint_sha256") != source_fingerprint):
            mismatch.append("snapshot_report.json")
        if mismatch:
            ap.error(f"已存在的工作副本与源数据不一致，拒绝覆盖: {mismatch[:8]}")
        print(f"已存在且核验一致: {target}；未写入任何文件")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mower_v1_snapshot_", dir=target.parent))
    try:
        for rel in files:
            destination = stage / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, destination)  # independent bytes; no hardlinks
            if sha256(destination) != source_hashes[rel.as_posix()]:
                raise RuntimeError(f"复制后 SHA256 不一致: {rel}")
        # Preserve original data.yaml under its original name in metadata/;
        # only the generated root data.yaml is used for training.
        original_yaml = stage / "data.yaml"
        (stage / "metadata").mkdir()
        original_yaml.rename(stage / "metadata" / "original_data.yaml")
        (stage / "data.yaml").write_text(snapshot_yaml(target), encoding="utf-8")
        report = {"created_at": timestamp(), "source": str(source), "target": str(target),
                  "copy_method": "shutil.copy2; no hardlinks", "source_validation": "0 ERROR",
                  "class_names": CLASSES, "source_file_sha256": source_hashes,
                  "dataset_fingerprint_sha256": source_fingerprint,
                  "generated_data_yaml_sha256": sha256(stage / "data.yaml"),
                  "file_count": len(files), "status": "verified"}
        write_json(stage / "snapshot_report.json", report)
        # Rename only after every byte has passed verification. An existing
        # destination is rejected above and must never be replaced.
        if target.exists():
            raise FileExistsError(f"复制期间目标出现，拒绝覆盖: {target}")
        stage.rename(target)
    except Exception:
        print(f"准备失败；未创建正式工作副本。暂存目录保留供检查: {stage}")
        raise
    print(f"工作副本已创建并逐文件核验: {target} ({len(files)} 个源文件)")
    print(f"报告: {target / 'snapshot_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
