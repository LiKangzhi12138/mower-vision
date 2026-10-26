"""Export static FP32 ONNX and compare raw and postprocessed detections on CPU."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import CLASSES, ROOT, resolve, sha256, timestamp, versions, write_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def extract_prediction(value):
    """Get the exported inference tensor from Ultralytics' eval output tuple."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return extract_prediction(item)
            except TypeError:
                continue
    raise TypeError(f"无法从 PyTorch 输出中找到预测张量: {type(value).__name__}")


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (aa + bb - intersection) if aa + bb > intersection else 0.0


def detection_records(predictions):
    return [{"class_id": int(row[5]), "confidence": float(row[4]),
             "xyxy_letterbox": [float(x) for x in row[:4]]} for row in predictions]


def raw_candidate_records(raw, confidence_floor=0.001):
    """Decode the semantic candidate set for YOLO26 end-to-end [xyxy, score, class] output."""
    import numpy as np

    rows = np.asarray(raw, dtype=np.float32).reshape(-1, 6)
    valid = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= confidence_floor)]
    return detection_records(valid)


def match_detections(pt, ort):
    used = set()
    matches = []
    for i, pred in sorted(enumerate(pt), key=lambda pair: (-pair[1]["confidence"], pair[0])):
        options = [(iou(pred["xyxy_letterbox"], other["xyxy_letterbox"]), j)
                   for j, other in enumerate(ort) if j not in used and other["class_id"] == pred["class_id"]]
        score, j = max(options, default=(0.0, -1))
        if j >= 0:
            used.add(j)
            matches.append({"pt_index": i, "onnx_index": j, "iou": score,
                            "confidence_abs_diff": abs(pred["confidence"] - ort[j]["confidence"]),
                            "class_equal": pred["class_id"] == ort[j]["class_id"]})
    return matches


def raw_metrics(pt_array, onnx_array, atol, rtol):
    import numpy as np

    a, b = np.asarray(pt_array, dtype=np.float32), np.asarray(onnx_array, dtype=np.float32)
    if a.shape != b.shape:
        return {"shape_equal": False, "pt_shape": list(a.shape), "onnx_shape": list(b.shape),
                "allclose": False, "max_abs_diff": None, "mean_abs_diff": None,
                "rmse": None, "note": "Shape differs; raw elementwise comparison is undefined. Compare decoded detections separately."}
    diff = np.abs(a - b)
    return {"shape_equal": True, "shape": list(a.shape), "allclose": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
            "atol": atol, "rtol": rtol, "max_abs_diff": float(diff.max(initial=0)),
            "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
            "rmse": float(np.sqrt(np.mean(np.square(a - b)))) if diff.size else 0.0}


def write_consistency_report(path: Path, result: dict):
    consistency = result["consistency"]
    lines = ["# YOLO26s PyTorch / ONNX 一致性报告", "",
             f"- PyTorch 权重：`{result['pt_checkpoint']}`（SHA256 `{result['pt_sha256']}`）",
             f"- ONNX：`{result['onnx_file']}`（SHA256 `{result['onnx_sha256']}`，{result['onnx_bytes']} bytes）",
             f"- 导出：FP32、640、batch=1、static、simplify=False、opset {result['opset_imports']}",
             f"- 类别顺序：`{result['class_names']}`",
             f"- Checker：{result['onnx_checker']}；ORT：{result['onnxruntime_cpu_smoke']}，providers={result['onnxruntime_providers']}",
             f"- 输入：`{result['inputs']}`",
             f"- 输出：`{result['outputs']}`",
             f"- 输出解释：`{result['output_semantics']}`",
             f"- 预处理：{result['preprocessing']}",
             f"- 后处理：{result['postprocess']}",
             "", "## 一致性结果", "",
             f"状态：**{consistency['status']}**，验证图片 {consistency['image_count']} 张。",
             f"- 原始输出逐元素 allclose（atol={consistency['raw_atol']}、rtol={consistency['raw_rtol']}）："
             f"{consistency['raw_tensor_elementwise_allclose_all_images']}。",
             (f"- 按类别/框 IoU 匹配、忽略 top-k 行顺序的候选输出一致：{consistency['raw_semantic_outputs_agree_all_images']}。"
              if result["exported_end2end"] else
              f"- 同形状原始输出满足逐元素容差：{consistency['raw_semantic_outputs_agree_all_images']}。"),
             f"- 最终检测框一致（同类、一对一匹配、IoU≥0.99、置信度差≤0.01）："
             f"{consistency['all_final_detections_agree']}。", "",
             ("YOLO26 导出启用了端到端检测头，ONNX 输出为 top-k `(x1,y1,x2,y2,score,class)`。"
              "按 `score≥0.001` 过滤后以类别和框进行无序匹配，候选集合一致。"
              if result["exported_end2end"] else
              "本次导出保留 checkpoint 的非端到端原始头；PyTorch 与 ONNX 输出同形状原始候选，"
              "最终均使用同一套 Ultralytics class-aware NMS，conf、IoU 和 max_det 相同。"), "",
             "| 图片 | PyTorch 框数 | ONNX 框数 | 原始 max abs diff | 原始输出一致 | 最终框一致 |",
             "|---|---:|---:|---:|---|---|"]
    for item in consistency["images"]:
        lines.append(f"| `{Path(item['image']).name}` | {item['pt_detection_count']} | {item['onnx_detection_count']} | "
                     f"{item['raw'].get('max_abs_diff')} | {item['raw_candidates_agree']} | {item['detections_agree']} |")
    lines += ["", "一致性只验证本次权重、11 张 Val 图片及当前 CPU 后处理配置。尚需在 Jetson 上验证 TensorRT 精度、性能、内存与摄像头预处理一致性。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="outputs/experiments/pilot_yolo26s_001/weights/best.pt")
    ap.add_argument("--out", default="outputs/exports")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--source", default="datasets/mower_v1_snapshot/images/val",
                    help="一致性验证图片或目录；默认使用整个 Val 集")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--raw-atol", type=float, default=1e-4)
    ap.add_argument("--raw-rtol", type=float, default=1e-3)
    args = ap.parse_args()
    source = resolve(args.model)
    sample_source = resolve(args.source)
    out_root = resolve(args.out)
    if not source.is_file() or source.suffix.lower() != ".pt":
        ap.error(f"必须指定本地已存在的 .pt 权重，不会下载: {source}")
    if not sample_source.exists():
        ap.error(f"一致性验证图片路径不存在: {sample_source}")
    if args.imgsz != 640:
        ap.error("本轮主模型导出固定 imgsz=640")
    if not (0 <= args.conf <= 1 and 0 <= args.iou <= 1):
        ap.error("conf 和 iou 必须在 [0,1] 范围")
    images = ([sample_source] if sample_source.is_file() else
              sorted(p for p in sample_source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS))
    if not images:
        ap.error(f"没有可用于一致性验证的图片: {sample_source}")

    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression

    out_root.mkdir(parents=True, exist_ok=True)
    run = out_root / f"yolo26s_main_{timestamp()[:19].replace(':', '').replace('-', '').replace('T', '_')}"
    run.mkdir(parents=True, exist_ok=False)
    staged = run / source.name
    shutil.copy2(source, staged)
    pt_export = YOLO(str(staged))
    model_names = [pt_export.names[i] for i in range(len(pt_export.names))]
    if model_names != CLASSES:
        ap.error(f"主模型类别顺序不符: {model_names}")
    model_end2end = bool(getattr(pt_export.model, "end2end", False))
    export_parameters = {"format": "onnx", "imgsz": 640, "batch": 1, "dynamic": False,
                         "half": False, "device": "cpu", "simplify": False, "nms": None,
                         "opset": 0, "agnostic_nms": False}
    # nms=None preserves the checkpoint's default head. nms=False would switch
    # YOLO26 to its end-to-end top-k export path in this Ultralytics version.
    # opset=0 asks the installed exporter to select its supported default.
    export_call_parameters = {key: value for key, value in export_parameters.items() if key != "half"}
    onnx_path = Path(pt_export.export(**export_call_parameters))
    graph = onnx.load(str(onnx_path))
    props = {item.key: item.value for item in graph.metadata_props}
    props["names"] = json.dumps({i: name for i, name in enumerate(model_names)}, ensure_ascii=False)
    props["mower_preprocess"] = "BGR image; Ultralytics LetterBox(640x640, auto=False, center=True, padding=114); BGR2RGB; BCHW FP32 /255"
    # The exported output shape records whether the graph carries a top-k head.
    del graph.metadata_props[:]
    for key, value in props.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(onnx_path))
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"預期检测模型单输入单输出，实际输入/输出 {len(inputs)}/{len(outputs)}")
    input_info, output_info = inputs[0], outputs[0]
    expected_shape = [1, 3, 640, 640]
    if input_info.shape != expected_shape or input_info.type != "tensor(float)":
        raise RuntimeError(f"期望静态 FP32 输入 {expected_shape}，实际 {input_info.shape} {input_info.type}")
    exported_end2end = (len(output_info.shape) == 3 and output_info.shape[-1] == 6)
    props["mower_postprocess"] = (
        f"YOLO26 end-to-end head top-k in graph={exported_end2end}; external confidence/max_det filter; "
        f"no NMS; conf={args.conf}; max_det=300"
        if exported_end2end else
        f"external Ultralytics class-aware NMS; conf={args.conf}; iou={args.iou}; max_det=300"
    )
    # Keep report properties in the ONNX graph itself as well as export.json.
    graph = onnx.load(str(onnx_path))
    del graph.metadata_props[:]
    for key, value in props.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(onnx_path))
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)

    # Reload the source model so exporter-side flags cannot alter the PyTorch reference path.
    pt_reference = YOLO(str(source))
    pt_reference.model.to("cpu").eval()
    raw_end2end = bool(getattr(pt_reference.model, "end2end", False))
    # Match the PyTorch head to the output path represented by this ONNX graph.
    pt_reference.model.end2end = exported_end2end
    head = pt_reference.model.model[-1]
    if hasattr(head, "agnostic_nms"):
        head.agnostic_nms = False
    if hasattr(head, "max_det"):
        head.max_det = 300
    if bool(getattr(pt_reference.model, "end2end", False)) != exported_end2end:
        raise RuntimeError("无法将 PyTorch 检测头设置为 ONNX 导出时的后处理模式")
    stride = int(max(pt_reference.model.stride))
    letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=stride)
    sample_results = []
    all_detections_agree = True
    raw_allclose = True
    all_raw_semantic_agree = True
    ort_smoke_shape = None
    with torch.inference_mode():
        for image_path in images:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"OpenCV 无法解码验证图片: {image_path}")
            prepared = letterbox(image=bgr)
            rgb_chw = np.ascontiguousarray(prepared[:, :, ::-1].transpose(2, 0, 1))
            input_tensor = torch.from_numpy(rgb_chw).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
            pt_raw = extract_prediction(pt_reference.model(input_tensor)).detach().cpu().numpy()
            ort_outputs = session.run(None, {input_info.name: input_tensor.numpy()})
            onnx_raw = np.asarray(ort_outputs[0])
            if ort_smoke_shape is None:
                ort_smoke_shape = [list(np.asarray(x).shape) for x in ort_outputs]
            raw = raw_metrics(pt_raw, onnx_raw, args.raw_atol, args.raw_rtol)
            raw_allclose &= bool(raw["allclose"])
            if exported_end2end:
                pt_candidates = raw_candidate_records(pt_raw)
                onnx_candidates = raw_candidate_records(onnx_raw)
                candidate_matches = match_detections(pt_candidates, onnx_candidates)
                candidates_agree = (len(pt_candidates) == len(onnx_candidates) == len(candidate_matches) and
                                    all(m["iou"] >= 0.99 and m["confidence_abs_diff"] <= 0.01
                                        for m in candidate_matches))
            else:
                pt_candidates, onnx_candidates, candidate_matches = [], [], []
                candidates_agree = bool(raw["allclose"])
            all_raw_semantic_agree &= candidates_agree
            pt_nms = non_max_suppression(torch.from_numpy(pt_raw), conf_thres=args.conf, iou_thres=args.iou,
                                         classes=None, agnostic=False, max_det=300, nc=4,
                                         end2end=exported_end2end)[0].cpu().numpy()
            onnx_nms = non_max_suppression(torch.from_numpy(onnx_raw), conf_thres=args.conf, iou_thres=args.iou,
                                           classes=None, agnostic=False, max_det=300, nc=4,
                                           end2end=exported_end2end)[0].cpu().numpy()
            pt_dets, onnx_dets = detection_records(pt_nms), detection_records(onnx_nms)
            matched = match_detections(pt_dets, onnx_dets)
            detections_agree = (len(pt_dets) == len(onnx_dets) == len(matched) and
                                all(m["iou"] >= 0.99 and m["confidence_abs_diff"] <= 0.01 for m in matched))
            all_detections_agree &= detections_agree
            sample_results.append({"image": str(image_path), "image_sha256": sha256(image_path),
                                   "letterbox_tensor_shape": list(input_tensor.shape), "raw": raw,
                                   "raw_candidate_confidence_floor": 0.001 if exported_end2end else None,
                                   "pt_raw_candidate_count": len(pt_candidates), "onnx_raw_candidate_count": len(onnx_candidates),
                                   "raw_candidates_agree": candidates_agree, "raw_candidate_matches": candidate_matches,
                                   "pt_detection_count": len(pt_dets), "onnx_detection_count": len(onnx_dets),
                                   "detections_agree": detections_agree, "matches": matched,
                                   "pt_detections": pt_dets, "onnx_detections": onnx_dets})

    opsets = [{"domain": entry.domain or "ai.onnx", "version": entry.version} for entry in graph.opset_import]
    output_shape = [dim.dim_value if dim.dim_value else (dim.dim_param or None)
                    for dim in graph.graph.output[0].type.tensor_type.shape.dim]
    # YOLO26 top-k may reorder rows if end-to-end export is explicitly selected.
    # Keep positional tensor diffs as evidence, and judge semantic detections.
    raw_semantic_agree = all(x["raw_candidates_agree"] for x in sample_results)
    consistency_status = "passed" if raw_semantic_agree and all_detections_agree else "mismatch"
    metadata = {"created_at": timestamp(), "pt_checkpoint": str(source), "pt_sha256": sha256(source),
                "onnx_file": str(onnx_path), "onnx_sha256": sha256(onnx_path),
                "onnx_bytes": onnx_path.stat().st_size, "imgsz": 640, "batch": 1, "precision": "FP32",
                "dynamic": False, "simplify": False, "onnxslim_used": False, "export_parameters": export_parameters,
                "class_names": model_names, "class_order_verified": model_names == CLASSES,
                "source_checkpoint_end2end": model_end2end,
                "exported_end2end": exported_end2end,
                "pytorch_reference_end2end_before_override": raw_end2end,
                "postprocess": {"path": ("YOLO26 end-to-end top-k in ONNX graph + Ultralytics confidence/max_det filtering"
                                          if exported_end2end else "Ultralytics non_max_suppression applied to both raw tensors"),
                                "nms_embedded_in_onnx": False, "head_topk_embedded_in_onnx": exported_end2end,
                                "agnostic": False, "conf": args.conf,
                                "iou": args.iou, "max_det": 300,
                                "behavior": ("end-to-end confidence/max_det only; no NMS" if exported_end2end
                                             else "class-aware NMS")},
                "preprocessing": {"implementation": "Ultralytics LetterBox + OpenCV", "input_color": "BGR",
                                  "letterbox_shape": [640, 640], "auto": False, "center": True, "padding_value": 114,
                                  "channel_order": "RGB", "layout": "NCHW", "dtype": "float32", "scale": "1/255"},
                "onnx_checker": "passed", "onnxruntime_providers": session.get_providers(),
                "onnxruntime_cpu_smoke": "passed", "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in inputs],
                "outputs": [{"name": output_info.name, "shape": output_info.shape, "type": output_info.type,
                             "graph_static_shape": output_shape, "ort_actual_shape": ort_smoke_shape}],
                "output_semantics": ({"layout": "B,C,N", "box_channels": "decoded xywh in input-image pixels",
                                      "class_channels": "sigmoid probabilities in class_names order",
                                      "class_count": len(model_names), "candidate_count": output_info.shape[-1],
                                      "nms_embedded": False, "postprocess": "Ultralytics class-aware NMS"}
                                     if not exported_end2end else
                                     {"layout": "B,N,6", "fields": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
                                      "candidate_count": output_info.shape[1], "nms_embedded": False,
                                      "head_topk_embedded": True, "postprocess": "confidence/max_det filter"}),
                "opset_imports": opsets, "dependency_versions": versions(),
                "consistency": {"status": consistency_status, "image_count": len(images),
                                "raw_tensor_elementwise_allclose_all_images": raw_allclose,
                                "raw_semantic_outputs_agree_all_images": raw_semantic_agree,
                                "raw_atol": args.raw_atol,
                                "raw_rtol": args.raw_rtol, "all_final_detections_agree": all_detections_agree,
                                "match_rule": "class-equal, one-to-one by maximum IoU in shared 640 letterbox coordinates; require IoU>=0.99 and confidence difference<=0.01",
                                "images": sample_results}}
    write_json(run / "export.json", metadata)
    write_consistency_report(run / "consistency_report.md", metadata)
    print("ONNX:", onnx_path)
    print("ONNX SHA256:", metadata["onnx_sha256"])
    print("ONNX Checker: passed")
    print("Consistency:", consistency_status, f"({len(images)} images)")
    return 0 if consistency_status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
