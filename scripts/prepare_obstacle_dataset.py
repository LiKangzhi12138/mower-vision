"""Create/verify an independent one-class dataset from the immutable four-class snapshot."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from common import CLASSES, ROOT, SPLITS, class_names, dataset_paths, fingerprint, read_csv, resolve, sha256, timestamp, write_json
from validate_dataset import IMAGE_EXTS, validate

SOURCE = ROOT / "datasets" / "mower_v1_snapshot"
TARGET = ROOT / "datasets" / "mower_v1_obstacle"
METADATA = ("manifest.csv", "split_manifest.csv", "dataset_info.json", "export_report.txt",
            "snapshot_report.json", "metadata/original_data.yaml")
VERSION = "mower_v1_obstacle_v1"


def mapped_label(text: str) -> tuple[str, int]:
    """Change only the class token. Keep every coordinate token and line ending."""
    lines = text.splitlines(keepends=True)
    mapped = []
    count = 0
    for line in lines:
        content = line.rstrip("\r\n")
        ending = line[len(content):]
        if not content.strip():
            mapped.append(line)
            continue
        parts = content.split()
        if len(parts) != 5 or parts[0] not in {"0", "1", "2", "3"}:
            raise ValueError(f"无效四类别 YOLO 标签行: {content!r}")
        # Preserve whitespace and all coordinate bytes after the first token.
        first_end = content.find(parts[0]) + len(parts[0])
        mapped.append(content[:content.find(parts[0])] + "0" + content[first_end:] + ending)
        count += 1
    return "".join(mapped), count


def expected_files(source: Path):
    files = []
    for split in SPLITS:
        files += [p.relative_to(source) for p in (source / "images" / split).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        files += [p.relative_to(source) for p in (source / "labels" / split).glob("*.txt")]
    files += [Path(p) for p in METADATA]
    return sorted(files, key=lambda p: p.as_posix())


def data_yaml(target: Path):
    return (f"path: {target.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "nc: 1\nnames:\n  0: obstacle\n")


def verify_obstacle_dataset(target: Path = TARGET, source: Path = SOURCE) -> list[str]:
    """Read-only verification used before training as well as after conversion."""
    problems = []
    report_path = target / "conversion_report.json"
    if not report_path.is_file():
        return [f"缺少 conversion_report.json: {report_path}"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("version") != VERSION or report.get("source") != str(source) or report.get("target") != str(target):
        problems.append("转换版本或源/目标路径与报告不一致")
    yaml_path = target / "data.yaml"
    if not yaml_path.is_file() or yaml_path.read_text(encoding="utf-8") != data_yaml(target):
        problems.append("单类别 data.yaml 不一致")
        return problems
    root, cfg = dataset_paths(yaml_path)
    if root != target or cfg.get("nc") != 1 or class_names(cfg) != ["obstacle"]:
        problems.append("data.yaml 路径或类别不一致")
    if report.get("source_fingerprint_sha256") != fingerprint(source / "data.yaml")["sha256"]:
        problems.append("四类别源数据 fingerprint 已变化")
    if report.get("target_fingerprint_sha256") != fingerprint(yaml_path)["sha256"]:
        problems.append("单类别数据 fingerprint 已变化")
    source_hashes = report.get("source_file_sha256", {})
    target_hashes = report.get("target_file_sha256", {})
    expected = expected_files(source)
    if set(source_hashes) != {p.as_posix() for p in expected} or set(target_hashes) != set(source_hashes):
        problems.append("文件清单不完整")
    image_counts, object_counts = Counter(), Counter()
    for rel in expected:
        key = rel.as_posix()
        a, b = source / rel, target / rel
        if not a.is_file() or not b.is_file():
            problems.append(f"缺失文件: {key}")
            continue
        if sha256(a) != source_hashes.get(key) or sha256(b) != target_hashes.get(key):
            problems.append(f"SHA256 不一致: {key}")
        if rel.parts[0] == "images":
            if sha256(a) != sha256(b):
                problems.append(f"图片内容变化: {key}")
            image_counts[rel.parts[1]] += 1
        elif rel.parts[0] == "labels":
            with a.open("r", encoding="utf-8-sig", newline="") as stream:
                original = stream.read()
            with b.open("r", encoding="utf-8-sig", newline="") as stream:
                converted = stream.read()
            mapped, count = mapped_label(original)
            if converted != mapped:
                problems.append(f"类别映射或 bbox 文本不一致: {key}")
            object_counts[rel.parts[1]] += count
        elif sha256(a) != sha256(b):
            problems.append(f"原始元数据变化: {key}")
    if dict(image_counts) != report.get("split_image_counts") or dict(object_counts) != report.get("split_object_counts"):
        problems.append("图片或目标数量与转换报告不一致")
    status = Counter(r["status"] for r in read_csv(target / "split_manifest.csv"))
    if dict(status) != report.get("status_counts"):
        problems.append("positive/negative/exclude 状态与报告不一致")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if args.verify_only:
        errors = verify_obstacle_dataset()
        for problem in errors:
            print("ERROR:", problem)
        if errors:
            return 2
        print(f"单类别数据逐文件校验通过: {TARGET}")
        return 0
    if TARGET.exists():
        ap.error(f"目标已存在，拒绝覆盖: {TARGET}。请使用 --verify-only 核验。")
    if not SOURCE.is_dir():
        ap.error(f"源工作副本不存在: {SOURCE}")
    audit = validate(SOURCE / "data.yaml")
    errors = [x for x in audit["issues"] if x["level"] == "ERROR"]
    if errors:
        ap.error(f"四类别源数据存在 {len(errors)} 个 ERROR，拒绝转换")
    snapshot = json.loads((SOURCE / "snapshot_report.json").read_text(encoding="utf-8"))
    source_fp = fingerprint(SOURCE / "data.yaml")["sha256"]
    if source_fp != snapshot.get("dataset_fingerprint_sha256"):
        ap.error("四类别源工作副本 fingerprint 与制作报告不一致")
    files = expected_files(SOURCE)
    if any((SOURCE / rel).is_symlink() for rel in files):
        ap.error("源文件包含符号链接，拒绝转换")
    source_hashes = {p.as_posix(): sha256(SOURCE / p) for p in files}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mower_v1_obstacle_", dir=TARGET.parent))
    split_images, split_objects = Counter(), Counter()
    target_hashes = {}
    try:
        for rel in files:
            src, dst = SOURCE / rel, stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if rel.parts[0] == "labels":
                with src.open("r", encoding="utf-8-sig", newline="") as stream:
                    mapped, count = mapped_label(stream.read())
                with dst.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(mapped)
                split_objects[rel.parts[1]] += count
            else:
                shutil.copy2(src, dst)  # Independent bytes; never hardlink.
                if rel.parts[0] == "images":
                    split_images[rel.parts[1]] += 1
            target_hashes[rel.as_posix()] = sha256(dst)
            if rel.parts[0] != "labels" and target_hashes[rel.as_posix()] != source_hashes[rel.as_posix()]:
                raise RuntimeError(f"复制后 SHA256 不一致: {rel}")
        (stage / "data.yaml").write_text(data_yaml(TARGET), encoding="utf-8")
        # fingerprint resolves the absolute path in generated data.yaml, so
        # compute the target fingerprint from stage files using a stage YAML.
        (stage / "_stage_data.yaml").write_text(data_yaml(stage), encoding="utf-8")
        target_fp = fingerprint(stage / "_stage_data.yaml")["sha256"]
        (stage / "_stage_data.yaml").unlink()
        status = Counter(r["status"] for r in read_csv(stage / "split_manifest.csv"))
        report = {"version": VERSION, "created_at": timestamp(), "source": str(SOURCE), "target": str(TARGET),
                  "conversion_rule": "0|1|2|3 -> 0 obstacle; all bbox coordinate text unchanged",
                  "copy_method": "shutil.copy2 for images/metadata; UTF-8 write for mapped labels; no hardlinks",
                  "source_class_names": CLASSES, "target_class_names": ["obstacle"],
                  "source_fingerprint_sha256": source_fp, "target_fingerprint_sha256": target_fp,
                  "source_file_sha256": source_hashes, "target_file_sha256": target_hashes,
                  "split_image_counts": dict(split_images), "split_object_counts": dict(split_objects),
                  "status_counts": dict(status), "split_manifest_sha256": target_hashes["split_manifest.csv"]}
        write_json(stage / "conversion_report.json", report)
        if TARGET.exists():
            raise FileExistsError(f"复制期间目标出现，拒绝覆盖: {TARGET}")
        stage.rename(TARGET)
        errors = verify_obstacle_dataset()
        if errors:
            raise RuntimeError(f"转换后核验失败: {errors[:5]}")
    except Exception:
        print(f"转换未成功；请检查暂存或目标目录: {stage} / {TARGET}")
        raise
    print(f"已制作单类别数据: {TARGET}; 图片 {dict(split_images)}; 框 {dict(split_objects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
