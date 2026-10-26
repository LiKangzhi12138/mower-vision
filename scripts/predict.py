"""Single-image or directory inference, with images and machine-readable detections."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import CLASSES, resolve, timestamp, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="outputs/predictions")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7, help="由模型后处理解释；端到端头可能不使用 NMS")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    model_path, source = resolve(args.model), resolve(args.source)
    if not model_path.is_file() or not source.exists():
        ap.error("模型或输入路径不存在")
    out = resolve(args.out) / timestamp()[:19].replace(":", "").replace("-", "").replace("T", "_")
    out.mkdir(parents=True, exist_ok=False)
    from ultralytics import YOLO
    from PIL import Image
    model = YOLO(str(model_path))
    images = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in
                                                       {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        ap.error("没有可推理图片")
    records = []
    flat = []
    for image in images:
        result = model.predict(source=str(image), conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                               device=0, verbose=False)[0]
        detections = []
        for box in result.boxes:
            cls = int(box.cls.item())
            item = {"class_id": cls, "class_name": CLASSES[cls] if cls < len(CLASSES) else str(cls),
                    "confidence": float(box.conf.item()), "xyxy": box.xyxy[0].tolist()}
            detections.append(item)
            flat.append({"image": str(image), **{k: v for k, v in item.items() if k != "xyxy"},
                         **dict(zip(("x1", "y1", "x2", "y2"), item["xyxy"]))})
        plotted = result.plot()
        import cv2
        cv2.imwrite(str(out / image.name), plotted)
        records.append({"image": str(image), "annotated": str(out / image.name),
                        "count": len(detections), "speed_ms": dict(result.speed), "detections": detections})
        print(f"{image.name}: {len(detections)} detections, {result.speed.get('inference')} ms inference")
    write_json(out / "predictions.json", {"model": str(model_path), "conf": args.conf, "iou": args.iou,
                                           "postprocess_note": "iou 由当前 Ultralytics 模型接口解释，端到端头可能不使用传统 NMS",
                                           "images": records})
    with (out / "detections.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(flat)
    print("输出:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
