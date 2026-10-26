"""Plan a future group-level split; never move images or rewrite existing splits."""
from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path

from common import SPLITS, read_csv, resolve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="包含 image_id, group_id, status 及 class_ids 的新增数据 CSV")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.85, 0.10, 0.05])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = read_csv(resolve(args.manifest))
    required = {"image_id", "group_id", "status", "class_ids"}
    if not rows or not required <= rows[0].keys():
        ap.error(f"manifest 必须包含 {sorted(required)}；class_ids 用分号分隔类别 ID")
    if any(not r["group_id"].strip() for r in rows):
        ap.error("缺少可靠 group_id，拒绝按图片伪分组")
    if any(r["status"] not in ("positive", "negative") for r in rows):
        ap.error("输入只能包含已审核的 positive/negative 新增数据")
    if any(r.get("split", "").strip() for r in rows):
        ap.error("输入包含已有 split；拒绝重排冻结数据")
    if len({r["image_id"] for r in rows}) != len(rows):
        ap.error("image_id 重复")
    if any(x <= 0 for x in args.ratios) or abs(sum(args.ratios) - 1) > 1e-6:
        ap.error("三个比例必须为正且总和为 1")
    groups = collections.defaultdict(list)
    for row in rows:
        try:
            ids = {int(x) for x in row["class_ids"].split(";") if x.strip()}
            if not ids <= set(range(4)) or (row["status"] == "positive" and not ids):
                raise ValueError()
            if row["status"] == "negative" and ids:
                raise ValueError()
        except ValueError:
            ap.error(f"class_ids 非法: {row['image_id']}")
        groups[row["group_id"]].append((row, ids))
    if len(groups) < 3:
        ap.error("至少需要三个可靠来源组才能划分 Train/Val/Test")
    rng = random.Random(args.seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)
    total = len(rows)
    target = {s: total * ratio for s, ratio in zip(SPLITS, args.ratios)}
    class_total = collections.Counter(c for group in groups.values() for _, ids in group for c in ids)
    assigned = {s: [] for s in SPLITS}
    sizes = collections.Counter()
    classes = {s: collections.Counter() for s in SPLITS}
    # Seed each split with one complete group, then greedily minimize count/class deviation.
    for i, key in enumerate(keys):
        group = groups[key]
        gc = collections.Counter(c for _, ids in group for c in ids)
        candidates = [SPLITS[i]] if i < 3 else SPLITS
        def score(s):
            size_cost = ((sizes[s] + len(group) - target[s]) / max(1, target[s])) ** 2
            class_cost = sum(((classes[s][c] + gc[c] - class_total[c] * args.ratios[SPLITS.index(s)]) /
                              max(1, class_total[c])) ** 2 for c in range(4))
            return size_cost + class_cost
        chosen = min(candidates, key=score)
        assigned[chosen].append(key)
        sizes[chosen] += len(group)
        classes[chosen].update(gc)
    output = []
    for split, group_ids in assigned.items():
        for group_id in group_ids:
            for row, _ in groups[group_id]:
                output.append({"image_id": row["image_id"], "group_id": group_id,
                               "status": row["status"], "eligible": "true", "split": split})
    print({s: {"images": sizes[s], "class_images": dict(classes[s]), "groups": len(assigned[s])} for s in SPLITS})
    print("小数据集及不可拆分组可能无法同时满足所有比例和类别目标。")
    if args.dry_run:
        print("dry-run：未写入文件")
        return 0
    dest = resolve(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "group_id", "status", "eligible", "split"])
        writer.writeheader()
        writer.writerows(output)
    print(f"已写入 {dest}；未复制、移动或改写图片及现有划分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
