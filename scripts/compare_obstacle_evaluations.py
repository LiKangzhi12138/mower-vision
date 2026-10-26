"""Compare saved obstacle evaluations only after both Val runs are complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT, dataset_paths, resolve, sha256


def load(path):
    return json.loads((path / "threshold_metrics.json").read_text(encoding="utf-8"))


def original_class(gt):
    return gt.get("true_class_id", gt["class_id"])


def check_same_ground_truth(old, new):
    old_root, old_cfg = dataset_paths(old["data"])
    new_root, new_cfg = dataset_paths(new["data"])
    checked_images = set()
    for threshold in old["per_image_details"]:
        a = {r["image"]: r for r in old["per_image_details"][threshold]}
        b = {r["image"]: r for r in new["per_image_details"][threshold]}
        if set(a) != set(b):
            raise ValueError(f"{threshold}: Val 图片列表不同")
        for name in a:
            if name not in checked_images:
                if sha256(old_root / old_cfg["val"] / name) != sha256(new_root / new_cfg["val"] / name):
                    raise ValueError(f"Val 图片内容不同: {name}")
                checked_images.add(name)
            ga, gb = a[name]["ground_truth"], b[name]["ground_truth"]
            if len(ga) != len(gb) or any(
                original_class(x) != original_class(y) or
                any(abs(i - j) > 0.01 for i, j in zip(x["box"], y["box"]))
                for x, y in zip(ga, gb)
            ):
                raise ValueError(f"真实框或原类别分组不同: {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="outputs/evaluation/obstacle_pilot_001")
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="outputs/reports/obstacle_model_comparison.md")
    args = ap.parse_args()
    old, new, out = load(resolve(args.old)), load(resolve(args.new)), resolve(args.out)
    if out.exists():
        ap.error(f"比较报告已存在，拒绝覆盖: {out}")
    if old["split"] != "val" or new["split"] != "val":
        ap.error("对照只允许使用 Val；Test 不用于模型或阈值选择")
    for key in ("match_iou", "image_count", "ground_truth_count"):
        if old[key] != new[key]:
            ap.error(f"评估口径不同: {key}")
    for key in ("imgsz", "device", "conf", "iou", "agnostic_nms", "max_det"):
        if old["inference"][key] != new["inference"][key]:
            ap.error(f"推理设置不同: {key}")
    thresholds = [r["threshold"] for r in old["thresholds"]]
    if thresholds != [r["threshold"] for r in new["thresholds"]]:
        ap.error("置信度阈值集合不同")
    check_same_ground_truth(old, new)
    lines = ["# 四类别与单类别 YOLO26s 障碍物检测 Val 对照", "",
             f"旧权重：`{old['model']}`；新权重：`{new['model']}`。",
             "已核对两次评估的 Val 图片 SHA256、真实框坐标及原四类别归属一致；匹配 IoU 和推理参数一致。",
             "", "| conf | 旧 Recall | 新 Recall | 旧 FP/image | 新 FP/image | 旧 F1 | 新 F1 | 旧 FN | 新 FN |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a, b in zip(old["thresholds"], new["thresholds"]):
        lines.append(f"| {a['threshold']:.2f} | {a['recall']:.3f} | {b['recall']:.3f} | "
                     f"{a['fp_per_image']:.3f} | {b['fp_per_image']:.3f} | "
                     f"{a['f1']:.3f} | {b['f1']:.3f} | {a['fn']} | {b['fn']} |")
    lines += ["", "## 漏检图片变化（conf=0.25）", "",
              "| 图片 | 旧 FN | 新 FN |", "|---|---:|---:|"]
    a = {r["image"]: len(r["false_negative_indices"]) for r in old["per_image_details"]["0.25"]}
    b = {r["image"]: len(r["false_negative_indices"]) for r in new["per_image_details"]["0.25"]}
    for name in sorted(a, key=lambda n: (-(a[n] - b[n]), n)):
        lines.append(f"| `{name}` | {a[name]} | {b[name]} |")
    lines += ["", "## 解释边界", "",
              "四类别权重在 `agnostic_nms=False` 下的 NMS 按类别进行，单类别权重只存在一个类别，"
              "因此即使调用参数相同，重叠框抑制的实际效果也可能不同；这属于模型输出与后处理的整体对照，"
              "不能把全部差异归因于单类别训练。若需要分离该因素，应另做两模型共同参数的 NMS 敏感性实验。",
              "Val 仅 11 张且 Train/Val 采集会话交叉；不使用 Test 选择阈值，也不宣称此对照证明部署性能。", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"对照报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
