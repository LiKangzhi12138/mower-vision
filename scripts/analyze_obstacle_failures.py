"""CPU-only review of saved obstacle evaluation; never re-run model inference."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from common import CLASSES, ROOT, resolve, sha256, timestamp, write_json
from evaluate_obstacle import box_iou, draw_image, read_ground_truth

EVALUATION = ROOT / "outputs" / "evaluation" / "obstacle_pilot_001"
SNAPSHOT = ROOT / "datasets" / "mower_v1_snapshot"
REPORT = ROOT / "outputs" / "reports" / "obstacle_failure_analysis.md"
THRESHOLDS = ("0.05", "0.10", "0.25")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def size_bin(box, width, height):
    # Equivalent square side after 640 / max(original dimension) scaling.
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    return "small_<32px_equiv" if (area ** 0.5) * 640 / max(width, height) < 32 else "other"


def review(eval_dir=EVALUATION, snapshot=SNAPSHOT, report_path=REPORT):
    payload = json.loads((eval_dir / "threshold_metrics.json").read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("class_names") != CLASSES:
        raise ValueError("只能复核原四类别 Val 评估")
    if Path(payload["data"]).resolve() != (snapshot / "data.yaml").resolve():
        raise ValueError("评估数据来源与指定四类别 snapshot 不一致")
    if sha256(Path(payload["model"])) != payload["model_sha256"]:
        raise ValueError("评估权重 SHA256 已变化")
    csv_metrics = {f"{float(row['threshold']):.2f}": row for row in read_csv(eval_dir / "threshold_metrics.csv")}
    csv_fp, csv_fn = read_csv(eval_dir / "false_positives.csv"), read_csv(eval_dir / "false_negatives.csv")
    per_image = read_csv(eval_dir / "per_image_metrics.csv")
    summary = {}
    case_dir = report_path.parent / "obstacle_failure_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for threshold in THRESHOLDS:
        details = payload["per_image_details"][threshold]
        metric = csv_metrics[threshold]
        if len(details) != int(payload["image_count"]):
            raise ValueError(f"{threshold}: 图片数不一致")
        if len([r for r in per_image if f"{float(r['threshold']):.2f}" == threshold]) != len(details):
            raise ValueError(f"{threshold}: per_image_metrics.csv 行数不一致")
        category_fp, category_fn, class_fn, size_total, size_fn = (Counter() for _ in range(5))
        cases = []
        fp_case_rows, fn_case_rows = [], []
        low_matches = {d["image"]: {m["gt_index"] for m in d["matches"]}
                       for d in payload["per_image_details"]["0.05"]}
        for detail in details:
            image_name = detail["image"]
            image_path = snapshot / "images" / "val" / image_name
            with Image.open(image_path) as im:
                width, height = im.size
            gt = detail["ground_truth"]
            actual_gt = read_ground_truth(snapshot / "labels" / "val" / f"{Path(image_name).stem}.txt", width, height)
            if len(gt) != len(actual_gt) or any(
                a["class_id"] != b["class_id"] or any(abs(x - y) > 0.01 for x, y in zip(a["box"], b["box"]))
                for a, b in zip(gt, actual_gt)
            ):
                raise ValueError(f"已保存 GT 与当前原四类别标签不一致: {image_name}")
            pred, matches = detail["predictions"], detail["matches"]
            matched_gt = {m["gt_index"] for m in matches}
            for item in gt:
                size_total[size_bin(item["box"], width, height)] += 1
            for index in detail["false_positive_indices"]:
                p = pred[index]
                overlaps = [box_iou(p["box"], g["box"]) for g in gt]
                best_iou = max(overlaps, default=0)
                if any(overlap >= 0.5 and i in matched_gt for i, overlap in enumerate(overlaps)):
                    category = "overlap_with_matched_GT_possible_duplicate"
                elif best_iou >= 0.5:
                    category = "matching_competition"
                elif best_iou >= 0.1:
                    category = "partial_overlap_localization_or_annotation"
                else:
                    category = "little_overlap_background_or_unlabeled_review"
                category_fp[category] += 1
                fp_case_rows.append({"image": image_name, "prediction_index": index, "confidence": p["confidence"],
                                     "box": p["box"], "max_gt_iou": best_iou, "triage": category})
            for index in detail["false_negative_indices"]:
                item = gt[index]
                cls = CLASSES[item["class_id"]]
                class_fn[cls] += 1
                size = size_bin(item["box"], width, height)
                size_fn[size] += 1
                best_iou = max((box_iou(item["box"], p["box"]) for p in pred), default=0)
                if threshold != "0.05" and index in low_matches[image_name]:
                    category = "detected_at_0.05_but_below_threshold"
                elif best_iou >= 0.5:
                    category = "matching_competition"
                elif best_iou >= 0.1:
                    category = "partial_overlap_localization_or_annotation"
                else:
                    category = "no_matching_candidate_at_this_threshold"
                category_fn[category] += 1
                fn_case_rows.append({"image": image_name, "gt_index": index, "true_class": cls,
                                     "box": item["box"], "size_bin": size, "max_prediction_iou": best_iou,
                                     "triage": category})
            cases.append({"image": image_name, "gt": len(gt), "predictions": len(pred), "tp": len(matches),
                          "fp": len(detail["false_positive_indices"]), "fn": len(detail["false_negative_indices"])})
        if (sum(category_fp.values()), sum(category_fn.values())) != (int(metric["fp"]), int(metric["fn"])):
            raise ValueError(f"{threshold}: CSV 与 JSON 的 FP/FN 数量不一致")
        if len([r for r in csv_fp if f"{float(r['threshold']):.2f}" == threshold]) != int(metric["fp"]):
            raise ValueError(f"{threshold}: false_positives.csv 数量不一致")
        if len([r for r in csv_fn if f"{float(r['threshold']):.2f}" == threshold]) != int(metric["fn"]):
            raise ValueError(f"{threshold}: false_negatives.csv 数量不一致")
        summary[threshold] = {"metric": metric, "fp_triage": dict(category_fp), "fn_triage": dict(category_fn),
                              "fn_by_true_class": dict(class_fn), "size_total": dict(size_total),
                              "size_fn": dict(size_fn), "cases": sorted(cases, key=lambda c: (-c["fn"], -c["fp"])),
                              "fp_cases": fp_case_rows, "fn_cases": fn_case_rows}

    # Representative cases are selected by deterministic FP/FN burden, not by
    # subjective choice of an easy validation image.
    selected = []
    for threshold, key, limit in (("0.25", "fn", 3), ("0.25", "fp", 3), ("0.05", "fp", 3)):
        ranked = sorted(summary[threshold]["cases"], key=lambda c: (-c[key], c["image"]))
        for case in ranked[:limit]:
            if case["image"] not in selected:
                selected.append(case["image"])
    selected = selected[:5]
    for threshold in THRESHOLDS:
        detail_by_image = {d["image"]: d for d in payload["per_image_details"][threshold]}
        for image_name in selected:
            item = detail_by_image[image_name]
            draw_image(snapshot / "images" / "val" / image_name,
                       case_dir / f"conf_{threshold}_{Path(image_name).stem}.png",
                       item["ground_truth"], item["predictions"], item["matches"],
                       item["false_positive_indices"], item["false_negative_indices"])

    info = payload["inference"]
    lines = ["# YOLO26s 类别无关障碍物检测 FP/FN 复核", "",
             f"生成时间：{timestamp()}。来源：`{eval_dir}`；数据：`{snapshot}`；只读复核 Val 标签与已有预测，不重新推理。",
             "", "## 阈值与高风险图片", "",
             "| conf | TP | FP | FN | FP/image | 漏检图片 | FP 最多图片 | FN 最多图片 |",
             "|---:|---:|---:|---:|---:|---:|---|---|"]
    for threshold in THRESHOLDS:
        s = summary[threshold]
        m = s["metric"]
        fp_top = max(s["cases"], key=lambda x: x["fp"])
        fn_top = max(s["cases"], key=lambda x: x["fn"])
        lines.append(f"| {threshold} | {m['tp']} | {m['fp']} | {m['fn']} | {float(m['fp_per_image']):.2f} | "
                     f"{m['images_with_fn']} | `{fp_top['image']}` ({fp_top['fp']}) | `{fn_top['image']}` ({fn_top['fn']}) |")
    lines += ["", "### 各图片 FP/FN（按 0.25 漏检数排序）", "",
              "| 图片 | 0.05 FP/FN | 0.10 FP/FN | 0.25 FP/FN |", "|---|---:|---:|---:|"]
    case_by_t = {t: {c["image"]: c for c in summary[t]["cases"]} for t in THRESHOLDS}
    for case in summary["0.25"]["cases"]:
        name = case["image"]
        cells = [f"{case_by_t[t][name]['fp']}/{case_by_t[t][name]['fn']}" for t in THRESHOLDS]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines += ["", "## 按真实类别与尺寸统计的漏检", "",
              "小目标定义：将原图等比例缩放到长边 640 后，bbox 面积的等效正方形边长小于 32 像素。该口径仅用于探索性分层。", "",
              "| conf | stone FN | tree FN | person FN | other_roadblock FN | 小目标 FN/GT | 其余 FN/GT |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for t in THRESHOLDS:
        s = summary[t]
        counts = s["fn_by_true_class"]
        lines.append(f"| {t} | " + " | ".join(str(counts.get(c, 0)) for c in CLASSES) +
                     f" | {s['size_fn'].get('small_<32px_equiv', 0)}/{s['size_total'].get('small_<32px_equiv', 0)}" +
                     f" | {s['size_fn'].get('other', 0)}/{s['size_total'].get('other', 0)} |")
    lines += ["", "## 程序分诊（仅几何与阈值证据，待人工确认）", "",
              "FP 与已匹配 GT IoU≥0.5：疑似重复预测；IoU 0.1–0.5：定位或标注边界需复核；IoU<0.1：疑似背景或未标注目标，不能自动判为纯误检。",
              "FN 若在 0.05 已匹配、在更高阈值未匹配，归为阈值导致；其余按当前最高 IoU 分为匹配竞争、部分重叠或无合格候选。", ""]
    for t in THRESHOLDS:
        s = summary[t]
        lines.append(f"- conf={t}：FP {s['fp_triage']}；FN {s['fn_triage']}")
    lines += ["", "在 0.05 下，疑似重复框和部分重叠框合计占 150/194 个 FP；"
              "四类别的 class-aware NMS 可保留同一区域不同预测类别的框，可能贡献重复框，但每个红框仍需人工核实。"
              "在 0.25 下，23/46 个 FN 对应 0.05 曾匹配的 GT，阈值提高是可直接观察的原因。"
              "0.05 的小目标漏检率为 13/48（27.1%），其余为 10/72（13.9%）；"
              "0.10 也呈类似趋势。到 0.25 时两组分别为 18/48 和 28/72，"
              "小目标不再明显更差，因此不能笼统归因于尺寸。"
              "高负担样本 `fab6c019...` 为密集、暗光和局部遮挡场景；"
              "这些视觉条件可能影响检测，但需要更多独立会话样本才能判断因果。"]
    lines += ["", "## 人工复核清单", "",
              "以下图片按 FP/FN 总负担自动挑选；绿色 TP、红色 FP、橙色 FN。请核对红框内是否确有障碍物、橙框位置是否合理、相邻框是否指向同一实体。不得直接更改 GT。", ""]
    for name in selected:
        lines.append(f"- `{name}`：" + "；".join(
            f"[{t} 可视化](obstacle_failure_cases/conf_{t}_{Path(name).stem}.png)"
            f"（FP {case_by_t[t][name]['fp']} / FN {case_by_t[t][name]['fn']}）" for t in THRESHOLDS))
    lines += ["", "## 后处理路径核查", "",
              f"评估 JSON 记录 `end2end={info['end2end']}`、`agnostic_nms={info['agnostic_nms']}`、"
              f"`iou={info['iou']}`、`conf={info['conf']}`、`max_det={info['max_det']}`。",
              "本机 Ultralytics 8.4.174 的 `DetectionPredictor.postprocess` 将实际 `self.model.end2end` 传给 "
              "`non_max_suppression`；只有 `end2end=True` 或输出末维为 6 才跳过传统 NMS。"
              "对当前 `best.pt` 进行了不推理的 CPU 加载检查：检测头为 `Detect`、`nc=4`、"
              "`model.end2end=False`、`head.end2end=False`、`head.agnostic_nms=False`。"
              "本次运行元数据为 `end2end=False`，因此按 class-aware NMS 解释，评估脚本未追加类别无关 NMS。"
              "此前的端到端 top-k 描述是该版本支持的另一条条件路径，不能代替对本次权重运行结果的判定。",
              "", "## 候选截断与结论边界", "",
              "- 已保存结果最小阈值为 0.05；在 0.05 下单图最多 56 个预测，小于 max_det=300，"
              "但 JSON 未保存 0.001–0.05 的全部候选数或 NMS 前候选，不能排除某图曾触及 300 上限。"
              "低于 0.001 的预测也无法从现有输出恢复。若需定量排除截断，应另做只读诊断推理，记录每图 0.001 后候选数并提高 max_det 对照。",
              "- 现有 Val 仅 11 张、person 仅 2 个实例，Train/Val 采集会话交叉，且没有人工确认的纯背景 negative；"
              "本报告不能证明跨场景泛化、人员安全性或真实误报警水平。",
              "- 小目标关联、重复预测与定位问题是几何分诊结果，成因仍需结合原图、标注规则和人工复核；不自动修正标签。", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(report_path.with_suffix(".json"), {"source_evaluation": str(eval_dir), "selected_images": selected,
                                                  "thresholds": summary})
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation", default=str(EVALUATION))
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args()
    summary = review(resolve(args.evaluation), SNAPSHOT, resolve(args.out))
    print(f"FP/FN 复核报告: {resolve(args.out)}; 0.25 FN={summary['0.25']['metric']['fn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
