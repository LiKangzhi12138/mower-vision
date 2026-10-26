"""Read-only YOLO dataset audit. Exit 2 when training-blocking errors exist."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from common import CLASSES, SPLITS, class_names, dataset_paths, read_csv, resolve, sha256, timestamp, write_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate(data_yaml: str | Path, *, hash_images: bool = True) -> dict:
    issues: list[dict] = []
    def issue(level: str, code: str, detail: str) -> None:
        issues.append({"level": level, "code": code, "detail": detail})

    report = {"checked_at": timestamp(), "data_yaml": str(resolve(data_yaml)), "issues": issues}
    try:
        root, cfg = dataset_paths(data_yaml)
    except Exception as exc:
        issue("ERROR", "DATA_YAML", str(exc))
        return report
    report["dataset_root"] = str(root)
    if not root.is_dir():
        issue("ERROR", "ROOT_MISSING", str(root))
        return report
    if cfg.get("nc") != 4 or class_names(cfg) != CLASSES:
        issue("ERROR", "CLASSES", f"期望 {CLASSES}; 实际 nc={cfg.get('nc')}, names={class_names(cfg)}")
    try:
        manifest = read_csv(root / "manifest.csv")
        split_manifest = read_csv(root / "split_manifest.csv")
        info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    except Exception as exc:
        issue("ERROR", "METADATA", str(exc))
        return report
    expected_m = {"image_id", "image_sha256", "group_id", "status", "split", "image_path", "label_path", "object_count"}
    expected_s = {"image_id", "group_id", "status", "eligible", "split"}
    if manifest and not expected_m <= manifest[0].keys():
        issue("ERROR", "MANIFEST_SCHEMA", f"缺少字段: {sorted(expected_m - manifest[0].keys())}")
        return report
    if split_manifest and not expected_s <= split_manifest[0].keys():
        issue("ERROR", "SPLIT_SCHEMA", f"缺少字段: {sorted(expected_s - split_manifest[0].keys())}")
        return report
    by_id = {r["image_id"]: r for r in manifest}
    split_by_id = {r["image_id"]: r for r in split_manifest}
    if len(by_id) != len(manifest) or len(split_by_id) != len(split_manifest):
        issue("ERROR", "DUPLICATE_IMAGE_ID", "manifest 中 image_id 重复")
    if not set(by_id) <= set(split_by_id):
        issue("ERROR", "MANIFEST_IDS", "manifest 的导出 image_id 未全部出现在 split_manifest")
    for image_id, sr in split_by_id.items():
        if image_id not in by_id and (sr["status"] in ("positive", "negative") or
                                   sr["eligible"].lower() != "false" or sr["split"]):
            issue("ERROR", "SPLIT_NONEXPORTED", f"未导出的记录却被标记为可训练: {image_id}")
    counts = {s: {"images": 0, "labels": 0, "instances": [0] * 4,
                  "class_images": [0] * 4, "empty_labels": 0} for s in SPLITS}
    seen_names: dict[str, str] = {}
    seen_hash: dict[str, str] = {}
    all_files: set[str] = set()
    small = {"area_lt_0.001": 0, "area_lt_0.01": 0, "total": 0}
    for split in SPLITS:
        if split not in cfg:
            issue("ERROR", "SPLIT_CONFIG", f"缺少 {split}")
            continue
        image_dir = root / cfg[split]
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            issue("ERROR", "SPLIT_PATH", f"目录不存在: {image_dir} 或 {label_dir}")
            continue
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        labels = sorted(label_dir.glob("*.txt"))
        counts[split]["images"], counts[split]["labels"] = len(images), len(labels)
        stems = collections.Counter(p.stem for p in images)
        if any(v > 1 for v in stems.values()):
            issue("ERROR", "STEM_COLLISION", f"{split}: 不同扩展名共享文件名 stem")
        image_stems = set(stems)
        label_stems = {p.stem for p in labels}
        for stem in sorted(image_stems - label_stems):
            issue("ERROR", "MISSING_LABEL", f"{split}/{stem}")
        for stem in sorted(label_stems - image_stems):
            issue("ERROR", "ORPHAN_LABEL", f"{split}/{stem}")
        for p in images:
            rel = p.relative_to(root).as_posix()
            all_files.add(rel)
            if p.name in seen_names:
                issue("WARNING", "FILENAME_REUSED", f"{p.name}: {seen_names[p.name]}, {split}")
            seen_names[p.name] = split
            row = by_id.get(p.stem)
            sr = split_by_id.get(p.stem)
            if row is None or sr is None:
                issue("ERROR", "UNLISTED_IMAGE", rel)
            else:
                if row["status"] not in ("positive", "negative") or sr["eligible"].lower() != "true":
                    issue("ERROR", "INELIGIBLE_EXPORTED", rel)
                if row["split"] != split or sr["split"] != split or row["image_path"].replace("\\", "/") != rel:
                    issue("ERROR", "SPLIT_MISMATCH", rel)
                if row["label_path"].replace("\\", "/") != f"labels/{split}/{p.stem}.txt":
                    issue("ERROR", "LABEL_PATH_MISMATCH", rel)
            try:
                from PIL import Image
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im.load()
                    if row and (int(row["image_width"]) != im.width or int(row["image_height"]) != im.height):
                        issue("ERROR", "IMAGE_SIZE", rel)
            except Exception as exc:
                issue("ERROR", "IMAGE_DECODE", f"{rel}: {exc}")
            if hash_images:
                digest = sha256(p)
                if digest in seen_hash:
                    issue("ERROR" if seen_hash[digest].split("/")[1] != split else "WARNING",
                          "DUPLICATE_CONTENT", f"{rel} 与 {seen_hash[digest]}")
                seen_hash[digest] = rel
                if row and digest != row["exported_image_sha256"]:
                    issue("ERROR", "IMAGE_HASH", rel)
            label = label_dir / f"{p.stem}.txt"
            if not label.exists():
                continue
            present = set()
            lines = [x.strip() for x in label.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
            if not lines:
                counts[split]["empty_labels"] += 1
                if not row or row["status"] != "negative":
                    issue("ERROR", "EMPTY_WITHOUT_NEGATIVE", rel)
            elif row and row["status"] != "positive":
                issue("ERROR", "NONEMPTY_NOT_POSITIVE", rel)
            if row and row["object_count"] != str(len(lines)):
                issue("ERROR", "OBJECT_COUNT", rel)
            for n, line in enumerate(lines, 1):
                parts = line.split()
                where = f"{label.relative_to(root).as_posix()}:{n}"
                if len(parts) != 5:
                    issue("ERROR", "LABEL_COLUMNS", where)
                    continue
                try:
                    cls = int(parts[0])
                    if str(cls) != parts[0] or cls not in range(4):
                        raise ValueError("illegal class")
                    x, y, w, h = map(float, parts[1:])
                    if not all(math.isfinite(v) for v in (x, y, w, h)):
                        raise ValueError("non-finite")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("range")
                    if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                        raise ValueError("geometry")
                except ValueError as exc:
                    issue("ERROR", "LABEL_FORMAT", f"{where}: {exc}")
                    continue
                counts[split]["instances"][cls] += 1
                present.add(cls)
                area = w * h
                small["total"] += 1
                small["area_lt_0.001"] += area < 0.001
                small["area_lt_0.01"] += area < 0.01
            for cls in present:
                counts[split]["class_images"][cls] += 1
    for row in manifest:
        sr = split_by_id.get(row["image_id"])
        if not sr:
            continue
        status, split = row["status"], row["split"]
        eligible = sr["eligible"].lower() == "true"
        if status in ("positive", "negative"):
            if not eligible or split not in SPLITS or row["image_path"].replace("\\", "/") not in all_files:
                issue("ERROR", "ELIGIBLE_MISSING", row["image_id"])
        elif eligible or split or row["image_path"] or row["label_path"]:
            issue("ERROR", "EXCLUDED_EXPORTED", row["image_id"])
        if sr["status"] != status or sr["split"] != split or sr["group_id"] != row["group_id"]:
            issue("ERROR", "MANIFEST_CONFLICT", row["image_id"])
    groups = collections.defaultdict(set)
    for row in manifest:
        if row["group_id"] and row["split"] in SPLITS:
            groups[row["group_id"]].add(row["split"])
    for group, splits in groups.items():
        if len(splits) > 1:
            issue("ERROR", "GROUP_LEAKAGE", f"{group}: {sorted(splits)}")
    group_coverage = sum(bool(r["group_id"]) for r in manifest if r["split"] in SPLITS)
    if group_coverage < sum(counts[s]["images"] for s in SPLITS):
        issue("WARNING", "GROUP_UNVERIFIED", f"仅 {group_coverage} 张导出图片有 group_id；无法充分核验同源采集泄漏")
    if counts["test"]["images"] < 20:
        issue("WARNING", "SMALL_TEST", f"Test 仅 {counts['test']['images']} 张")
    for split in ("val", "test"):
        for cls, name in enumerate(CLASSES):
            if counts[split]["instances"][cls] == 0:
                issue("WARNING", "CLASS_ABSENT", f"{split} 无 {name} 实例，无法评估该类召回率")
            elif counts[split]["instances"][cls] < 5:
                issue("WARNING", "CLASS_SPARSE", f"{split} 的 {name} 仅 {counts[split]['instances'][cls]} 个实例")
    for cls, name in enumerate(CLASSES):
        n = sum(counts[s]["instances"][cls] for s in SPLITS)
        if n < 50:
            issue("WARNING", "RARE_CLASS", f"{name}: {n} 个实例")
    actual = {s: counts[s]["images"] for s in SPLITS}
    if info.get("actual_counts") != actual:
        issue("ERROR", "INFO_COUNTS", f"dataset_info.json={info.get('actual_counts')}, actual={actual}")
    actual_status = dict(collections.Counter(r["status"] for r in split_manifest))
    if info.get("status_counts") != actual_status:
        issue("ERROR", "STATUS_COUNTS", f"dataset_info.json={info.get('status_counts')}, split_manifest={actual_status}")
    if len(manifest) != sum(actual.values()) or info.get("exported_image_count") != len(manifest):
        issue("ERROR", "EXPORT_COUNTS", "导出 manifest、磁盘图片数和 dataset_info.json 不一致")
    if info.get("classes") != {str(i): n for i, n in enumerate(CLASSES)}:
        issue("ERROR", "INFO_CLASSES", "dataset_info.json 类别与固定类别不符")
    report.update({"counts": counts, "small_bbox": small, "metadata": {
        "master_image_count": info.get("master_image_count"), "exported_image_count": info.get("exported_image_count"),
        "status_counts": info.get("status_counts"), "group_coverage": group_coverage,
        "grouping_note": info.get("grouping_note"), "split_algorithm": info.get("split_algorithm")}})
    issue("INFO", "SUMMARY", f"Train/Val/Test={actual}; 小框面积<0.001: {small['area_lt_0.001']}/{small['total']}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/datasets/mower_v1.yaml")
    ap.add_argument("--out", default="outputs/reports/dataset_audit")
    ap.add_argument("--no-hash", action="store_true", help="跳过图片内容哈希；完整审查时不要使用")
    args = ap.parse_args()
    report = validate(args.data, hash_images=not args.no_hash)
    out = resolve(args.out)
    write_json(out.with_suffix(".json"), report)
    lines = ["# 数据集审查", "", f"数据集：{report.get('dataset_root', '未知')}", ""]
    for x in report["issues"]:
        lines.append(f"- **{x['level']} {x['code']}**：{x['detail']}")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for x in report["issues"]:
        print(f"{x['level']} {x['code']}: {x['detail']}")
    print(f"报告: {out.with_suffix('.json')} / {out.with_suffix('.md')}")
    return 2 if any(x["level"] == "ERROR" for x in report["issues"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
