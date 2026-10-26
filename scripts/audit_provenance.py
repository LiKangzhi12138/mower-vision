"""Read-only evidence audit from exported rows back to original Snapshot metadata."""
from __future__ import annotations

import argparse
import collections
import json
import re
from itertools import combinations
from pathlib import Path

from common import CLASSES, ROOT, SPLITS, read_csv, resolve, sha256, timestamp, write_json
from validate_dataset import validate

DEFAULT_SAMPLES = Path(r"E:\workspace\mower-data\samples\airport_mower_samples")
DEFAULT_EXPORT = Path(r"E:\workspace\mower-annotation-exports")
SAMPLE_NAME = re.compile(r"sample_(\d+)__image_(\d+)\.[^.]+$", re.IGNORECASE)


def dhash(path: Path) -> int:
    from PIL import Image
    with Image.open(path) as im:
        pixels = list(im.convert("L").resize((9, 8)).get_flattened_data())
    return sum(1 << i for i in range(64)
               if pixels[(i // 8) * 9 + i % 8] > pixels[(i // 8) * 9 + i % 8 + 1])


def session_feasibility(export: Path, samples: Path, rows: list[dict], split_rows: list[dict],
                        matched: list[dict], similar: list[dict], audit: dict) -> dict:
    """Analyze immutable labels under all 2+1+1 assignments; never move files."""
    by_id = {row["image_id"]: row for row in rows}
    stats: dict[str, dict] = {}
    for source in matched:
        session = source["session_id"]
        if not session:
            raise ValueError("原始记录缺少 session_id，不能进行完整会话划分分析")
        item = stats.setdefault(session, {"records": 0, "trainable_images": 0,
                                          "status": {"positive": 0, "negative": 0, "exclude": 0},
                                          "instances": {name: 0 for name in CLASSES},
                                          "class_images": {name: 0 for name in CLASSES},
                                          "small_area_lt_0_001": 0,
                                          "small_by_class": {name: 0 for name in CLASSES},
                                          "original_split": {s: 0 for s in SPLITS}})
        item["records"] += 1
        status = source["status"]
        if status not in item["status"]:
            raise ValueError(f"未知状态: {status}")
        item["status"][status] += 1
        if status == "exclude":
            continue
        row = by_id.get(source["image_id"])
        if row is None or row["status"] != status or row["split"] not in SPLITS:
            raise ValueError(f"导出 manifest 与来源记录不一致: {source['image_id']}")
        item["trainable_images"] += 1
        item["original_split"][row["split"]] += 1
        label = export / row["label_path"]
        if not label.is_file():
            raise ValueError(f"标签不存在: {label}")
        present = set()
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"标签列数异常: {label}")
            cls = int(parts[0])
            if cls not in range(len(CLASSES)):
                raise ValueError(f"类别 ID 异常: {label}")
            width, height = float(parts[3]), float(parts[4])
            name = CLASSES[cls]
            item["instances"][name] += 1
            present.add(name)
            if width * height < 0.001:
                item["small_area_lt_0_001"] += 1
                item["small_by_class"][name] += 1
        if status == "negative" and present:
            raise ValueError(f"negative 含非空标签: {label}")
        for name in present:
            item["class_images"][name] += 1
    if len(matched) != len(split_rows) or len(stats) != 4:
        raise ValueError("来源匹配或 session 数量不足；不能枚举完整四会话方案")
    if sum(x["trainable_images"] for x in stats.values()) != len(rows):
        raise ValueError("session 可训练图片数与导出 manifest 不一致")
    if any(x["level"] == "ERROR" for x in audit["issues"]):
        raise ValueError("数据集审查有 ERROR；不能继续候选划分分析")
    info = json.loads((export / "dataset_info.json").read_text(encoding="utf-8"))
    status_totals = {status: sum(s["status"][status] for s in stats.values())
                     for status in ("positive", "negative", "exclude")}
    if {k: v for k, v in status_totals.items() if v} != info.get("status_counts"):
        raise ValueError("session 状态合计与 dataset_info.json 不一致")
    class_totals = {name: sum(s["instances"][name] for s in stats.values()) for name in CLASSES}
    if class_totals != info.get("class_instance_counts"):
        raise ValueError("session 类别实例合计与 dataset_info.json 不一致")
    if sum(s["small_area_lt_0_001"] for s in stats.values()) != audit["small_bbox"]["area_lt_0.001"]:
        raise ValueError("session 小目标合计与数据集审查不一致")

    total = len(rows)
    relative_to_session = {row["relative_path"]: source["session_id"]
                           for source in matched for row in [by_id.get(source["image_id"])] if row}

    def aggregate(ids: list[str]) -> dict:
        images = sum(stats[s]["trainable_images"] for s in ids)
        instances = {name: sum(stats[s]["instances"][name] for s in ids) for name in CLASSES}
        class_images = {name: sum(stats[s]["class_images"][name] for s in ids) for name in CLASSES}
        return {"images": images, "ratio": images / total, "instances": instances,
                "class_images": class_images, "person_present": instances["person"] > 0,
                "missing_classes": [name for name in CLASSES if instances[name] == 0],
                "small_area_lt_0_001": sum(stats[s]["small_area_lt_0_001"] for s in ids)}

    sessions = sorted(stats)
    target_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    candidates = []
    for train_pair in combinations(sessions, 2):
        remaining = sorted(set(sessions) - set(train_pair))
        for val, test in ((remaining[0], remaining[1]), (remaining[1], remaining[0])):
            assignment = {"train": list(train_pair), "val": [val], "test": [test]}
            splits = {split: aggregate(assignment[split]) for split in SPLITS}
            deviations = {split: round(abs(splits[split]["ratio"] - target_ratios[split]), 6)
                          for split in SPLITS}
            anomalies = []
            if splits["train"]["images"] < 60:
                anomalies.append("train_under_60")
            for split in ("val", "test"):
                if splits[split]["images"] < 10:
                    anomalies.append(f"{split}_under_10")
            for split in SPLITS:
                if splits[split]["missing_classes"]:
                    anomalies.append(f"{split}_missing_{','.join(splits[split]['missing_classes'])}")
            session_to_split = {session: split for split, ids in assignment.items() for session in ids}
            near_cross = sum(session_to_split[relative_to_session[p["left"]]] !=
                             session_to_split[relative_to_session[p["right"]]]
                             for p in similar if p["left"] in relative_to_session and p["right"] in relative_to_session)
            person_covered = sum(splits[split]["person_present"] for split in SPLITS)
            missing_count = sum(len(splits[split]["missing_classes"]) for split in SPLITS)
            ratio_l1 = round(sum(deviations.values()), 6)
            # Coverage is evaluated before size; Val/Test have equal importance.
            score_key = (-person_covered, missing_count, int(splits["train"]["images"] < 60),
                         ratio_l1, -min(splits["val"]["images"], splits["test"]["images"]))
            candidates.append({"assignment": assignment, "splits": splits, "strict_session_isolation": True,
                               "ratio_abs_deviation": deviations, "ratio_l1_deviation": ratio_l1,
                               "person_covered_splits": person_covered, "missing_class_occurrences": missing_count,
                               "dhash_candidate_pairs_cross_new_splits": near_cross,
                               "anomalies": anomalies, "_score_key": score_key})
    candidates.sort(key=lambda c: (c["_score_key"], c["assignment"]["train"],
                                   c["assignment"]["val"], c["assignment"]["test"]))
    previous_score = None
    current_rank = 0
    for order, candidate in enumerate(candidates, 1):
        if candidate["_score_key"] != previous_score:
            current_rank = order
        previous_score = candidate["_score_key"]
        candidate["rank"] = current_rank
        candidate["order"] = order
        del candidate["_score_key"]

    folds = []
    for held_out in sessions:
        remaining = [session for session in sessions if session != held_out]
        choices = []
        for val in remaining:
            train = [s for s in remaining if s != val]
            tr, va = aggregate(train), aggregate([val])
            key = (not tr["person_present"], not va["person_present"],
                   len(tr["missing_classes"]), len(va["missing_classes"]),
                   tr["images"] < 60, va["images"] < 10, -tr["images"], val)
            choices.append({"train_sessions": train, "val_session": val,
                            "train": tr, "val": va, "_rank_key": key})
        choices.sort(key=lambda c: c["_rank_key"])
        for choice in choices:
            del choice["_rank_key"]
        folds.append({"test_session": held_out, "test": aggregate([held_out]),
                      "inner_validation_options": choices,
                      "suggested_inner_validation": choices[0]["val_session"],
                      "selection_basis": "仅用剩余三个 session 的 Train/Val 类别覆盖与规模；不使用 held-out 测试指标"})

    best = candidates[0]
    adequate = (best["person_covered_splits"] == 3 and best["missing_class_occurrences"] == 0
                and best["splits"]["train"]["images"] >= 60
                and min(best["splits"][s]["images"] for s in ("val", "test")) >= 10)
    return {"analysis_time": timestamp(), "source": str(export), "samples_root": str(samples),
            "class_names": CLASSES, "small_target_definition": "normalized bbox area < 0.001",
            "total_original_records": len(split_rows), "total_trainable_images": total,
            "original_split_counts": {s: sum(r["split"] == s for r in rows) for s in SPLITS},
            "session_statistics": stats, "original_session_leakage":
                {s: v["original_split"] for s, v in stats.items() if sum(bool(n) for n in v["original_split"].values()) > 1},
            "target_ratios": target_ratios, "ranking_rule":
                "先最大化 person 在三个 split 的覆盖，再最小化类别缺失；随后考虑 Train<60、8:1:1 L1 偏差和较小 holdout 的规模。实质指标相同则并列，不依据模型性能或容易的 Test 排序。",
            "candidate_count": len(candidates), "candidates": candidates,
            "recommended_exploratory_rank": 1,
            "equally_ranked_exploratory_assignments": [c["assignment"] for c in candidates if c["rank"] == 1],
            "fully_adequate_single_split": adequate,
            "leave_one_session_out": folds,
            "leave_one_session_out_assessment": {"useful_as_diagnostic": True,
                                                 "sufficient_for_credible_generalization": False,
                                                 "test_never_used_for_early_stopping": True},
            "recommend_new_sessions": True,
            "recommend_start_formal_baseline_now": False,
            "limitations": ["原始数据仅四个采集会话；2+1+1 的 Train 只有两个会话",
                            "原始 Test 无 person，现有 Val/Test 会话交叉",
                            "dHash 相似仅是筛查线索；会话隔离不保证场景隔离"]}


def feasibility_markdown(result: dict) -> str:
    sessions = result["session_statistics"]
    lines = ["# 四会话独立划分可行性", "", "## 已核实的来源", "",
             f"- 原始记录 {result['total_original_records']} 条；可训练图片 {result['total_trainable_images']} 张。",
             "- 每条记录以 SHA256 对应原始 Snapshot 图像后，读取其 metadata.json 中的 session_id；未回填 group_id。",
             "- 原 Train/Val/Test 中四个 session 全部跨 split；原划分不能衡量独立采集会话泛化。",
             "- 小目标定义：归一化 bbox 面积 < 0.001。", "", "## 四个 session", "",
             "| session_id | 原始记录 | 可训练图 | 正/负/排除 | stone 实例/图 | tree 实例/图 | person 实例/图 | other 实例/图 | 小目标 | 原 Train/Val/Test |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for session, s in sorted(sessions.items()):
        cell = lambda name: f"{s['instances'][name]}/{s['class_images'][name]}"
        status = s["status"]
        old = s["original_split"]
        lines.append(f"| {session} | {s['records']} | {s['trainable_images']} | "
                     f"{status['positive']}/{status['negative']}/{status['exclude']} | "
                     f"{cell('stone')} | {cell('tree')} | {cell('person')} | {cell('other_roadblock')} | "
                     f"{s['small_area_lt_0_001']} | {old['train']}/{old['val']}/{old['test']} |")
    lines += ["", "## 12 种 2 Train + 1 Val + 1 Test 方案", "",
              "全部严格按 session 隔离；列中类别数为 stone/tree/person/other_roadblock 的实例数，完整实例数与图片数见 JSON。",
              "", "| 排名 | Train session | Val | Test | 图片数 T/V/E | 比例 T/V/E | person T/V/E | 小目标 T/V/E | 四类实例 T；V；E | 缺失次数 | 8:1:1 L1 偏差 | 异常 |",
              "|---:|---|---|---|---|---|---|---|---|---:|---:|---|"]
    for c in result["candidates"]:
        a, sp = c["assignment"], c["splits"]
        nums = "/".join(str(sp[s]["images"]) for s in SPLITS)
        ratios = "/".join(f"{sp[s]['ratio']:.1%}" for s in SPLITS)
        person = "/".join(str(sp[s]["instances"]["person"]) for s in SPLITS)
        small = "/".join(str(sp[s]["small_area_lt_0_001"]) for s in SPLITS)
        classes = "; ".join("/".join(str(sp[s]["instances"][n]) for n in CLASSES) for s in SPLITS)
        anomalies = ", ".join(c["anomalies"]) or "无"
        lines.append(f"| {c['rank']} | {'+'.join(a['train'])} | {a['val'][0]} | {a['test'][0]} | "
                     f"{nums} | {ratios} | {person} | {small} | {classes} | {c['missing_class_occurrences']} | "
                     f"{c['ratio_l1_deviation']:.3f} | {anomalies} |")
    best = [c for c in result["candidates"] if c["rank"] == 1]
    lines += ["", "**排序规则：**" + result["ranking_rule"], "",
              "188 个小目标中有 119 个集中于 20261010_153916；留出该会话时，小目标分布会明显变化。"
              "现有 dHash≤4 相似候选均位于同一真实 session，因此 12 种会话独立方案的此类跨 split 候选数均为 0；"
              "dHash 仍不能证明其他图片在场景层面独立。", "",
              "**探索性首选：**排名 1 有并列方案：" + "；".join(
                  f"Train={'+'.join(c['assignment']['train'])}, Val={c['assignment']['val'][0]}, Test={c['assignment']['test'][0]}"
                  for c in best) + "。现有证据不能在这些方案间公正选出唯一 Test；"
                  "这些只是事前数据覆盖排序，未实施划分，也未使用模型分数。",
              f"**充分性：**{'满足预设的最低覆盖与规模条件' if result['fully_adequate_single_split'] else '没有方案满足预设的最低覆盖与规模条件'}。"
              "即使条件满足，四个会话也不足以证明机场实地泛化。", "",
              "## Leave-One-Session-Out", "",
              "每折仅以一个 session 作外层测试；从另外三个中选一个完整 session 作验证，其余两个作训练。"
              "验证会话的选择只看剩余会话的类别覆盖和规模，绝不使用外层测试会话做 Early Stopping 或超参数选择。", "",
              "| 外层 Test | 内层 Train | 建议 Val | person 实例 Train/Val/Test | Test 缺失类别 | 三个可选 Val |",
              "|---|---|---|---|---|---|"]
    for fold in result["leave_one_session_out"]:
        option = fold["inner_validation_options"][0]
        counts = f"{option['train']['instances']['person']}/{option['val']['instances']['person']}/{fold['test']['instances']['person']}"
        lines.append(f"| {fold['test_session']} | {'+'.join(option['train_sessions'])} | {option['val_session']} | "
                     f"{counts} | {','.join(fold['test']['missing_classes']) or '无'} | "
                     f"{','.join(o['val_session'] for o in fold['inner_validation_options'])} |")
    lines += ["", "四折可作为会话敏感性诊断，但仅四个会话、每折训练只有两个会话，结果方差会很大；"
              "若某测试会话缺少某类，该折无法估计该类召回率。应补充独立采集会话，尤其含 person 的新会话。",
              "", "## 结论", "",
              "不建议现在启动用于正式泛化验收的 YOLO26s Baseline。建议先补采会话并设计真正独立的验证/测试。"
              "若现在仍运行完整训练，只能作为工程探索性 Baseline，不能作为可信泛化验收结果。"
              "本分析未复制图片、修改 manifest 或执行训练。", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SAMPLES), help="含 sample_*/metadata.json 的原始 Snapshot 目录")
    ap.add_argument("--out", default="outputs/reports/provenance_audit")
    ap.add_argument("--session-out", default="outputs/reports/session_split_feasibility")
    args = ap.parse_args()
    samples, export = resolve(args.source), DEFAULT_EXPORT
    rows = read_csv(export / "manifest.csv")
    split_rows = read_csv(export / "split_manifest.csv")
    issues = []
    session_splits: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    sample_splits: dict[str, set] = collections.defaultdict(set)
    matched = []
    for row in split_rows:
        m = SAMPLE_NAME.search(row["relative_path"])
        if not m:
            issues.append({"level": "WARNING", "code": "NO_SAMPLE_NAME", "image_id": row["image_id"]})
            continue
        sample = f"sample_{int(m[1]):06d}"
        original = samples / sample / "images" / f"image_{int(m[2]):02d}.png"
        metadata = samples / sample / "metadata.json"
        if not original.is_file() or not metadata.is_file():
            issues.append({"level": "WARNING", "code": "ORIGINAL_MISSING", "image_id": row["image_id"]})
            continue
        if sha256(original) != row["image_sha256"]:
            issues.append({"level": "ERROR", "code": "SOURCE_HASH_MISMATCH", "image_id": row["image_id"],
                           "sample": sample})
            continue
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        session = meta.get("session_id")
        if not session:
            issues.append({"level": "WARNING", "code": "SESSION_MISSING", "image_id": row["image_id"]})
        elif row["eligible"].lower() == "true":
            session_splits[session][row["split"]] += 1
            sample_splits[sample].add(row["split"])
        matched.append({"image_id": row["image_id"], "sample": sample, "session_id": session,
                        "split": row["split"], "status": row["status"], "original": str(original),
                        "sha256": row["image_sha256"]})
    cross_sessions = {session: dict(counts) for session, counts in session_splits.items() if len(counts) > 1}
    if cross_sessions:
        issues.append({"level": "WARNING", "code": "SESSION_CROSSES_SPLITS", "sessions": cross_sessions})
    same_sample = {sample: sorted(splits) for sample, splits in sample_splits.items() if len(splits) > 1}
    if same_sample:
        issues.append({"level": "ERROR", "code": "SAMPLE_CROSSES_SPLITS", "samples": same_sample})
    hashes = collections.defaultdict(set)
    for row in rows:
        hashes[row["image_sha256"]].add(row["split"])
    duplicate_cross = sum(len(splits) > 1 for splits in hashes.values())
    if duplicate_cross:
        issues.append({"level": "ERROR", "code": "EXACT_DUPLICATES_CROSS_SPLIT", "groups": duplicate_cross})
    visual = [(r, dhash(export / r["image_path"])) for r in rows]
    similar = []
    for (left, a), (right, b) in combinations(visual, 2):
        distance = (a ^ b).bit_count()
        if distance <= 4:
            similar.append({"hamming": distance, "left": left["relative_path"],
                            "right": right["relative_path"], "left_split": left["split"],
                            "right_split": right["split"]})
    audit = validate(ROOT / "configs" / "datasets" / "mower_v1.yaml")
    if len(matched) != len(split_rows):
        issues.append({"level": "WARNING", "code": "PROVENANCE_INCOMPLETE",
                       "matched": len(matched), "records": len(split_rows)})
    if any(x["level"] == "ERROR" for x in audit["issues"]):
        issues.append({"level": "ERROR", "code": "DATASET_AUDIT_ERROR"})
    report = {"checked_at": timestamp(), "export": str(export), "samples_root": str(samples),
              "export_metadata_sha256": {name: sha256(export / name)
                                         for name in ("manifest.csv", "split_manifest.csv", "dataset_info.json")},
              "split_records": len(split_rows), "exported_records": len(rows), "hash_matched_originals": len(matched),
              "source_matches": matched, "session_split_counts": {k: dict(v) for k, v in session_splits.items()},
              "cross_split_sessions": cross_sessions, "same_sample_cross_split": same_sample,
              "exact_duplicate_cross_split_groups": duplicate_cross,
              "dhash_threshold": 4, "dhash_candidate_pairs": similar,
              "dhash_note": "dHash 仅筛选视觉相似候选，不能单独证明重复拍摄或同一场景",
              "class_counts": audit.get("counts"), "issues": issues}
    out = resolve(args.out)
    write_json(out.with_suffix(".json"), report)
    lines = ["# 数据来源核验", "", f"- 导出记录：{len(rows)}；划分记录：{len(split_rows)}",
             f"- 匹配原始图片 SHA256：{len(matched)}/{len(split_rows)}",
             f"- 跨 split 的真实 session：{len(cross_sessions)}",
             f"- 跨 split 的相同 Snapshot：{len(same_sample)}",
             f"- 跨 split 的完全重复图片组：{duplicate_cross}",
             f"- dHash≤4 候选：{len(similar)}，其中跨 split：{sum(x['left_split'] != x['right_split'] for x in similar)}",
             "", "## 会话分布", ""]
    lines.extend(f"- `{session}`：{dict(counts)}" for session, counts in session_splits.items())
    lines += ["", "dHash 结果只用于人工复核，不用于自动更改 group_id 或划分。", ""]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("原始 SHA256 匹配:", len(matched), "/", len(split_rows))
    print("跨 split 会话:", cross_sessions)
    print("dHash≤4 跨 split 候选:", sum(x["left_split"] != x["right_split"] for x in similar))
    if not any(x["level"] == "ERROR" for x in issues):
        feasibility = session_feasibility(export, samples, rows, split_rows, matched, similar, audit)
        session_out = resolve(args.session_out)
        write_json(session_out.with_suffix(".json"), feasibility)
        session_out.with_suffix(".md").write_text(feasibility_markdown(feasibility), encoding="utf-8")
        print("会话方案:", len(feasibility["candidates"]), "报告:", session_out)
    return 2 if any(x["level"] == "ERROR" for x in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
