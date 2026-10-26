"""Evaluate four-class YOLO predictions as one obstacle class on a chosen split.

Predictions are obtained once at a low confidence. Matching is class agnostic,
confidence ordered, and one to one. No additional NMS is applied here.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import CLASSES, SPLITS, class_names, dataset_paths, resolve, sha256, timestamp, write_json

THRESHOLDS = (0.05, 0.10, 0.25, 0.40, 0.50, 0.70)
MATCH_IOU = 0.5
INFERENCE_CONF = 0.001
INFERENCE_IOU = 0.7  # Used by Ultralytics only for heads requiring traditional NMS.
MAX_DET = 300
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(ground_truth, predictions, threshold=MATCH_IOU):
    """Return (matches, FP indices, FN indices), ignoring the original class IDs.

    Each match has prediction index, GT index and IoU. Ties use original index,
    making results deterministic for equal-confidence predictions.
    """
    used_gt = set()
    matches = []
    false_positives = []
    order = sorted(range(len(predictions)), key=lambda i: (-predictions[i]["confidence"], i))
    for pred_index in order:
        options = [(box_iou(predictions[pred_index]["box"], gt["box"]), gt_index)
                   for gt_index, gt in enumerate(ground_truth) if gt_index not in used_gt]
        overlap, gt_index = max(options, key=lambda item: (item[0], -item[1]), default=(0.0, -1))
        if overlap >= threshold:
            used_gt.add(gt_index)
            matches.append({"prediction_index": pred_index, "gt_index": gt_index, "iou": overlap})
        else:
            false_positives.append(pred_index)
    false_negatives = [i for i in range(len(ground_truth)) if i not in used_gt]
    return matches, false_positives, false_negatives


def score_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def read_ground_truth(label_path, width, height):
    result = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"YOLO 标签应有 5 列: {label_path}:{line_number}")
        cls_text, *coords = parts
        cls = int(cls_text)
        x, y, w, h = map(float, coords)
        if cls not in range(len(CLASSES)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"无效类别或 bbox: {label_path}:{line_number}")
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            raise ValueError(f"bbox 越界: {label_path}:{line_number}")
        result.append({"class_id": cls, "box": [(x - w / 2) * width, (y - h / 2) * height,
                                                  (x + w / 2) * width, (y + h / 2) * height]})
    return result


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_image(image_path, out_path, ground_truth, predictions, matches, fp_indices, fn_indices):
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)

    def box(rect, color, label):
        rect = tuple(round(v) for v in rect)
        draw.rectangle(rect, outline=color, width=3)
        top = max(0, rect[1] - 16)
        draw.rectangle((rect[0], top, rect[0] + 7 * len(label) + 8, top + 15), fill=color)
        draw.text((rect[0] + 3, top + 1), label, fill="black")

    for item in matches:
        pred = predictions[item["prediction_index"]]
        box(pred["box"], "#38d66b", f"obstacle TP {pred['confidence']:.2f} IoU {item['iou']:.2f}")
    for index in fp_indices:
        pred = predictions[index]
        box(pred["box"], "#f44336", f"obstacle FP {pred['confidence']:.2f}")
    for index in fn_indices:
        box(ground_truth[index]["box"], "#ffab2e", f"obstacle FN GT#{index}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def plot_metrics(out, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([m["recall"] or 0 for m in metrics], [m["precision"] for m in metrics], "o-")
    for row in metrics:
        ax.annotate(f"{row['threshold']:.2f}", (row["recall"] or 0, row["precision"]), xytext=(5, 4), textcoords="offset points")
    ax.set(xlabel="Recall", ylabel="Precision", title="Class-agnostic obstacle PR (sampled thresholds)", xlim=(0, 1.05), ylim=(0, 1.05))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "precision_recall.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(metrics))
    ax.plot(xs, [m["recall"] or 0 for m in metrics], "o-", label="Recall")
    ax.plot(xs, [m["precision"] for m in metrics], "o-", label="Precision")
    ax.plot(xs, [m["f1"] for m in metrics], "o-", label="F1")
    ax2 = ax.twinx()
    ax2.bar(xs, [m["fp_per_image"] for m in metrics], width=0.4, color="tab:red", alpha=0.2, label="FP/image")
    ax.set(xticks=list(xs), xticklabels=[f"{m['threshold']:.2f}" for m in metrics], xlabel="Confidence threshold", ylabel="Score", ylim=(0, 1.05))
    ax2.set_ylabel("FP / image")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "threshold_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/experiments/pilot_yolo26s_001/weights/best.pt")
    parser.add_argument("--data", default="datasets/mower_v1_snapshot/data.yaml")
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument("--allow-test", action="store_true", help="仅允许显式最终 Test 评估；不在 Test 上推荐阈值")
    parser.add_argument("--out", default="outputs/evaluation/obstacle_pilot_001")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Test 集冻结；需要 --split test --allow-test，且不能用 Test 调优阈值")
    model_path, data_path, out = resolve(args.model), resolve(args.data), resolve(args.out)
    if not model_path.is_file():
        parser.error(f"模型不存在，不能自动下载: {model_path}")
    if not data_path.is_file():
        parser.error(f"数据配置不存在: {data_path}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"输出目录非空或不是目录，拒绝覆盖: {out}")
    root, cfg = dataset_paths(data_path)
    model_classes = class_names(cfg)
    one_class = cfg.get("nc") == 1 and model_classes == ["obstacle"]
    if not one_class and (model_classes != CLASSES or cfg.get("nc") != 4):
        parser.error(f"数据配置必须是原四类别或明确转换的单类别 obstacle: {data_path}")
    if one_class:
        from prepare_obstacle_dataset import SOURCE, TARGET, verify_obstacle_dataset
        if root != TARGET:
            parser.error(f"单类别数据必须是受校验的工作副本: {TARGET}")
        errors = verify_obstacle_dataset()
        if errors:
            parser.error(f"单类别数据校验失败: {errors[:5]}")
    image_dir = root / cfg[args.split]
    label_dir = root / "labels" / args.split
    if not image_dir.is_dir() or not label_dir.is_dir():
        parser.error(f"图片或标签目录不存在: {image_dir}, {label_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"评估集合为空: {image_dir}")
    if len({p.stem.casefold() for p in images}) != len(images):
        parser.error("图片文件名 stem 冲突，无法可靠配对标签")
    if {p.stem.casefold() for p in images} != {p.stem.casefold() for p in label_dir.glob("*.txt")}:
        parser.error("图片与标签没有一一配对，请先执行 validate")

    # Preflight all annotations before opening the output directory or loading the model.
    from PIL import Image
    samples = []
    for image_path in images:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        gt = read_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        if one_class:
            source_image = SOURCE / "images" / args.split / image_path.name
            if not source_image.is_file() or sha256(source_image) != sha256(image_path):
                parser.error(f"单类别图片与原 Val 图片不一致: {image_path.name}")
            original = read_ground_truth(SOURCE / "labels" / args.split / f"{image_path.stem}.txt", width, height)
            if len(gt) != len(original) or any(
                any(abs(a - b) > 0.01 for a, b in zip(x["box"], y["box"])) for x, y in zip(gt, original)
            ):
                parser.error(f"单类别 GT 与原四类别 bbox 不一致: {image_path.name}")
            for item, original_item in zip(gt, original):
                item["true_class_id"] = original_item["class_id"]
        else:
            for item in gt:
                item["true_class_id"] = item["class_id"]
        samples.append({"image": image_path, "gt": gt})

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    if len(model.names) != len(model_classes) or [model.names[i] for i in range(len(model_classes))] != model_classes:
        parser.error("权重类别定义与数据配置不一致")
    out.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        result = model.predict(source=str(sample["image"]), imgsz=640, conf=INFERENCE_CONF,
                               iou=INFERENCE_IOU, agnostic_nms=False, max_det=MAX_DET,
                               device=0, save=False, verbose=False)[0]
        sample["pred"] = [{"class_id": int(cls), "confidence": float(conf), "box": [float(v) for v in box]}
                          for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                                    result.boxes.conf.cpu().tolist(),
                                                    result.boxes.cls.cpu().tolist())]
        sample["speed_ms"] = dict(result.speed)

    end2end = bool(getattr(model.predictor.model, "end2end", False))
    postprocess = ("Ultralytics end-to-end head top-k (agnostic_nms=False), then confidence + max_det; no traditional NMS; no added NMS"
                   if end2end else "Ultralytics class-aware NMS (IoU=0.7); no added class-agnostic NMS")
    threshold_rows, per_image_rows, fp_rows, fn_rows, details = [], [], [], [], {}
    for threshold in THRESHOLDS:
        tp = fp = fn = images_with_fn = 0
        total_gt = Counter()
        matched_gt = Counter()
        image_details = []
        for sample in samples:
            gt = sample["gt"]
            pred = [p for p in sample["pred"] if p["confidence"] >= threshold]
            matches, fp_indices, fn_indices = match_boxes(gt, pred)
            image_name = sample["image"].name
            tp += len(matches)
            fp += len(fp_indices)
            fn += len(fn_indices)
            images_with_fn += bool(fn_indices)
            total_gt.update(g["true_class_id"] for g in gt)
            matched_gt.update(gt[m["gt_index"]]["true_class_id"] for m in matches)
            per_image_rows.append({"threshold": threshold, "image": image_name, "gt_count": len(gt),
                                   "prediction_count": len(pred), "tp": len(matches), "fp": len(fp_indices),
                                   "fn": len(fn_indices), "inference_ms": sample["speed_ms"].get("inference")})
            for i in fp_indices:
                p = pred[i]
                fp_rows.append({"threshold": threshold, "image": image_name, "prediction_index": i,
                                "confidence": p["confidence"], "predicted_class": model_classes[p["class_id"]],
                                "box_xyxy": json.dumps(p["box"])})
            for i in fn_indices:
                g = gt[i]
                fn_rows.append({"threshold": threshold, "image": image_name, "gt_index": i,
                                "true_class": CLASSES[g["true_class_id"]], "box_xyxy": json.dumps(g["box"])})
            image_details.append({"image": image_name, "ground_truth": gt, "predictions": pred,
                                  "matches": matches, "false_positive_indices": fp_indices,
                                  "false_negative_indices": fn_indices})
            if threshold in (0.25, 0.50):
                folder = out / "visualizations" / f"conf_{threshold:.2f}"
                draw_image(sample["image"], folder / f"{sample['image'].stem}.png",
                           gt, pred, matches, fp_indices, fn_indices)
        metrics = score_counts(tp, fp, fn)
        row = {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, **metrics,
               "fp_per_image": fp / len(samples), "prediction_count": tp + fp,
               "predictions_per_image": (tp + fp) / len(samples), "images_with_fn": images_with_fn,
               "per_true_class": {CLASSES[i]: {"gt": total_gt[i], "tp": matched_gt[i],
                                                "fn": total_gt[i] - matched_gt[i],
                                                "recall": matched_gt[i] / total_gt[i] if total_gt[i] else None}
                                  for i in range(4)}}
        threshold_rows.append(row)
        details[f"{threshold:.2f}"] = image_details

    write_csv(out / "threshold_metrics.csv", threshold_rows,
              ["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "fp_per_image",
               "prediction_count", "predictions_per_image", "images_with_fn"])
    write_csv(out / "per_image_metrics.csv", per_image_rows,
              ["threshold", "image", "gt_count", "prediction_count", "tp", "fp", "fn", "inference_ms"])
    write_csv(out / "false_positives.csv", fp_rows,
              ["threshold", "image", "prediction_index", "confidence", "predicted_class", "box_xyxy"])
    write_csv(out / "false_negatives.csv", fn_rows,
              ["threshold", "image", "gt_index", "true_class", "box_xyxy"])
    plot_metrics(out, threshold_rows)
    candidate = None if args.split == "test" else max(threshold_rows, key=lambda r: (r["f1"], r["recall"] or 0, -r["fp"]))["threshold"]
    metadata = {"created_at": timestamp(), "model": str(model_path), "model_sha256": sha256(model_path),
                "data": str(data_path), "split": args.split, "image_count": len(samples),
                "ground_truth_count": sum(len(s["gt"]) for s in samples), "class_names": model_classes,
                "true_class_group_names": CLASSES,
                "inference": {"imgsz": 640, "device": "cuda:0", "conf": INFERENCE_CONF,
                              "iou": INFERENCE_IOU, "agnostic_nms": False, "max_det": MAX_DET,
                              "end2end": end2end, "postprocess": postprocess,
                              "no_extra_class_agnostic_nms": True}, "match_iou": MATCH_IOU,
                "candidate_threshold": candidate,
                "thresholds": threshold_rows, "per_image_details": details}
    write_json(out / "threshold_metrics.json", metadata)
    lines = ["# 类别无关障碍物检测评估", "", f"- 权重：`{model_path}`（SHA256 `{metadata['model_sha256']}`）",
             f"- 数据：`{data_path}`，split：`{args.split}`，图片：{len(samples)} 张，GT：{metadata['ground_truth_count']} 个",
             f"- 推理：CUDA:0，imgsz=640，初始 conf={INFERENCE_CONF}，max_det={MAX_DET}，{postprocess}",
             "- 匹配：所有类别统一为 obstacle；预测按置信度降序，与尚未匹配的 GT 取最大 IoU，一对一匹配，IoU≥0.5 为 TP；其余预测为 FP、GT 为 FN。",
             "- 未添加类别无关 NMS；额外 NMS 可能压掉相邻障碍物，重复框在此按 FP 计。",
             "", "| conf | TP | FP | FN | Precision | Recall | F1 | FP/image | 漏检图片 | stone 召回 | person 召回 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"
    for row in threshold_rows:
        lines.append(f"| {row['threshold']:.2f} | {row['tp']} | {row['fp']} | {row['fn']} | "
                     f"{fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} | "
                     f"{fmt(row['fp_per_image'])} | {row['images_with_fn']} | "
                     f"{fmt(row['per_true_class']['stone']['recall'])} | {fmt(row['per_true_class']['person']['recall'])} |")
    lines += ["", "## 以真实类别分组的类别无关召回", ""]
    for row in threshold_rows:
        classes = ", ".join(f"{name}: {v['tp']}/{v['gt']} ({fmt(v['recall'])})" for name, v in row["per_true_class"].items())
        lines.append(f"- conf={row['threshold']:.2f}：{classes}")
    if candidate is not None:
        lines += ["", f"探索性候选阈值：**{candidate:.2f}**（当前 Val 上 F1 优先、召回次之、FP 较少时优先）。"
                  "这不是实际割草机部署的最优阈值，须结合漏检代价、人工复核与独立采集会话验证。"]
    lines += ["", "## 解释限制", "", "- 当前 Val 仅 11 张，其中 person 仅 2 个实例，不能证明人员检测安全性。",
              "- 当前 Train/Val 存在采集会话交叉，Val 指标不能证明跨场景泛化。",
              "- FP 表示相对于现有标注的未匹配预测；需人工排查是否存在真实障碍物漏标。",
              "- 缺乏人工确认的纯背景 negative 样本，不能充分估计真实场景误报警水平。",
              "- 固定 max_det=300 和低置信度候选仍可能截断预测；本报告不是四类别 mAP。",
              "- Test 集不得用于阈值调优；若显式评估 Test，本报告不推荐阈值。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(out), "split": args.split, "candidate_threshold": candidate,
                      "metrics": threshold_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
