"""Independent validation/test evaluation with FP/FN records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import CLASSES, SPLITS, dataset_paths, resolve, timestamp, write_json


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / (aa + bb - intersection) if aa + bb > intersection else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="configs/datasets/mower_v1.yaml")
    ap.add_argument("--split", choices=SPLITS, default="val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", default="outputs/evaluations")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        ap.error("Test 冻结；正式最终评估需显式 --split test --allow-test")
    model_path = resolve(args.model)
    if not model_path.is_file():
        ap.error(f"本地模型不存在: {model_path}")
    data_path = resolve(args.data)
    root, cfg = dataset_paths(data_path)
    out = resolve(args.out)
    run = out / f"{model_path.stem}_{args.split}_{timestamp()[:19].replace(':', '').replace('-', '').replace('T', '_')}"
    run.mkdir(parents=True, exist_ok=False)
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_path), split=args.split, imgsz=args.imgsz, device=0,
                        plots=True, project=str(out), name=run.name, exist_ok=True)
    box = metrics.box
    per_class = {name: {"map50_95": None, "map50": None, "precision": None, "recall": None,
                        "ground_truth_count": int(metrics.nt_per_class[i])}
                 for i, name in enumerate(CLASSES)}
    for index, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(index)
        per_class[CLASSES[int(cls)]].update({"map50_95": float(ap), "map50": float(ap50),
                                            "precision": float(p), "recall": float(r)})
    result = {"datetime": timestamp(), "model": str(model_path), "data": str(data_path),
              "split": args.split, "precision": float(box.mp), "recall": float(box.mr),
              "map50": float(box.map50), "map50_95": float(box.map),
              "per_class": per_class,
              "speed_ms": dict(metrics.speed), "weight_bytes": model_path.stat().st_size,
              "parameters": sum(p.numel() for p in model.model.parameters()),
              "confusion_matrix": [[0] * 5 for _ in range(5)],
              "confusion_matrix_axis": CLASSES + ["background"],
              "pr_curve": str(run / "BoxPR_curve.png"), "confusion_matrix_plot": str(run / "confusion_matrix.png"),
              "fp_fn_thresholds": {"conf": args.conf, "match_iou": 0.5}, "errors": []}
    image_dir = root / cfg[args.split]
    for image in sorted(p for p in image_dir.iterdir() if p.is_file()):
        from PIL import Image
        with Image.open(image) as im:
            width, height = im.size
        gt = []
        label = root / "labels" / args.split / f"{image.stem}.txt"
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            c, x, y, w, h = map(float, line.split())
            gt.append((int(c), [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]))
        pred_result = model.predict(source=str(image), imgsz=args.imgsz, conf=args.conf,
                                    iou=args.iou, device=0, verbose=False)[0]
        predictions = [(int(b.cls.item()), float(b.conf.item()), b.xyxy[0].tolist()) for b in pred_result.boxes]
        # Matrix rows are predicted classes, columns are ground-truth classes.
        # Match by geometry first, so class confusion remains visible.
        matrix_used = set()
        for pcls, _, pbox in sorted(predictions, key=lambda x: -x[1]):
            options = [(iou(pbox, gbox), j) for j, (_, gbox) in enumerate(gt) if j not in matrix_used]
            match_iou, index = max(options, default=(0, -1))
            if match_iou >= 0.5:
                matrix_used.add(index)
                result["confusion_matrix"][pcls][gt[index][0]] += 1
            else:
                result["confusion_matrix"][pcls][4] += 1
        for j, (gcls, _) in enumerate(gt):
            if j not in matrix_used:
                result["confusion_matrix"][4][gcls] += 1
        matched = set()
        for cls, confidence, box_xyxy in sorted(predictions, key=lambda x: -x[1]):
            options = [(iou(box_xyxy, gbox), j) for j, (gc, gbox) in enumerate(gt) if gc == cls and j not in matched]
            overlap, index = max(options, default=(0, -1))
            if overlap >= 0.5:
                matched.add(index)
            else:
                result["errors"].append({"kind": "FP", "image": str(image), "class": CLASSES[cls],
                                         "confidence": confidence, "box_xyxy": box_xyxy})
        for j, (cls, gt_box) in enumerate(gt):
            if j not in matched:
                result["errors"].append({"kind": "FN", "image": str(image), "class": CLASSES[cls],
                                         "box_xyxy": gt_box, "priority": cls in (0, 2)})
    result["priority_fn_count"] = {name: sum(e["kind"] == "FN" and e["class"] == name for e in result["errors"])
                                   for name in ("stone", "person")}
    write_json(run / "evaluation.json", result)
    summary = out / "comparison.csv"
    fields = ["datetime", "model", "data", "split", "precision", "recall", "map50", "map50_95",
              "weight_bytes", "parameters", "priority_fn_count", "report"]
    with summary.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({**{k: result[k] for k in fields if k in result},
                         "priority_fn_count": json.dumps(result["priority_fn_count"]), "report": str(run / "evaluation.json")})
    print(json.dumps({k: result[k] for k in ("split", "precision", "recall", "map50", "map50_95", "priority_fn_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
