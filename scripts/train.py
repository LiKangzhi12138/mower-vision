"""Controlled Ultralytics training. Importing this module never starts training."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from common import CLASSES, ROOT, class_names, dataset_paths, fingerprint, read_yaml, resolve, sha256, timestamp, versions, write_json
from validate_dataset import validate
from prepare_obstacle_dataset import SOURCE as FOUR_CLASS_SNAPSHOT, TARGET as OBSTACLE_SNAPSHOT, verify_obstacle_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--experiment-id", help="唯一实验编号；只允许字母、数字、下划线和横线")
    args = ap.parse_args()
    cfg = read_yaml(args.config)
    required = {"data", "architecture", "pretrained_checkpoint", "epochs", "batch", "imgsz", "device",
                "workers", "seed", "optimizer", "lr0", "patience", "amp", "project", "name", "resume", "validate_data"}
    missing = required - cfg.keys()
    if missing:
        ap.error(f"配置缺失: {sorted(missing)}")
    if not cfg["resume"] and not args.experiment_id:
        ap.error("新训练必须通过 --experiment-id 指定独立实验编号")
    if str(cfg["device"]) != "0":
        ap.error("本工程训练设备必须明确设为 CUDA device=0")
    data = resolve(cfg["data"])
    if not data.is_file():
        ap.error(f"训练数据配置不存在: {data}。Baseline 请先手动运行 -Task prepare 制作工作副本。")
    data_cfg = read_yaml(data)
    names = class_names(data_cfg)
    obstacle_dataset = data_cfg.get("nc") == 1 and names == ["obstacle"]
    if not obstacle_dataset and (data_cfg.get("nc") != 4 or names != CLASSES):
        ap.error("数据集类别必须是原四类别或明确转换的单类别 obstacle")
    resume = cfg["resume"]
    checkpoint = resolve(cfg["pretrained_checkpoint"])
    if resume:
        if not isinstance(resume, str):
            ap.error("resume 必须是明确的 last.pt 路径，不能使用 true")
        checkpoint = resolve(resume)
        if checkpoint.name != "last.pt":
            ap.error("resume 必须明确指向已有实验的 last.pt")
    elif cfg["name"] == "baseline" and checkpoint != ROOT / "models" / "yolo26s.pt":
        ap.error("正式 Baseline 只能从 models/yolo26s.pt 官方预训练权重开始，不能使用 Smoke checkpoint")
    if not checkpoint.is_file():
        ap.error(f"本地权重不存在: {checkpoint}。请自行下载官方兼容权重或提供路径；脚本不会联网下载。")
    if cfg["validate_data"]:
        if obstacle_dataset:
            errors = verify_obstacle_dataset()
            if errors:
                ap.error(f"单类别数据校验失败: {errors[:5]}")
        else:
            audit = validate(data)
            errors = [x for x in audit["issues"] if x["level"] == "ERROR"]
            if errors:
                for x in errors:
                    print("ERROR", x["code"], x["detail"])
                ap.error("数据集检查存在阻断项")
    data_root, _ = dataset_paths(data)
    dataset_fp = fingerprint(data)
    expected_root = OBSTACLE_SNAPSHOT if obstacle_dataset else FOUR_CLASS_SNAPSHOT
    if data_root != expected_root:
        ap.error(f"训练数据集必须是工作副本: {expected_root}")
    report_path = data_root / ("conversion_report.json" if obstacle_dataset else "snapshot_report.json")
    if not report_path.is_file():
        ap.error(f"工作副本缺少 {report_path.name}，请先制作数据副本")
    snapshot = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_fp = snapshot.get("target_fingerprint_sha256" if obstacle_dataset else "dataset_fingerprint_sha256")
    if dataset_fp["sha256"] != recorded_fp:
        ap.error("工作副本 fingerprint 与制作报告不符，拒绝训练")
    if cfg["name"] == "baseline":
        provenance_path = ROOT / "outputs" / "reports" / "provenance_audit.json"
        if not provenance_path.is_file():
            ap.error("缺少来源审查报告，请先运行 -Task provenance")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        source_hashes = provenance.get("export_metadata_sha256", {})
        if set(source_hashes) != {"manifest.csv", "split_manifest.csv", "dataset_info.json"}:
            ap.error("来源审查报告缺少元数据哈希，请重新运行 -Task provenance")
        for name, expected in source_hashes.items():
            if sha256(data_root / name) != expected:
                ap.error(f"来源审查报告与工作副本 {name} 不一致")
        if provenance.get("hash_matched_originals") != provenance.get("split_records"):
            ap.error("来源审查未能将所有划分记录追溯到原始 Snapshot")
        if provenance.get("cross_split_sessions"):
            print("WARNING: 原始 session_id 已确认跨 Train/Val/Test；本轮指标不能证明独立会话泛化。")
    import torch
    if not torch.cuda.is_available():
        ap.error("CUDA 不可用，请先运行 -Task env -CudaConv")
    from ultralytics import YOLO
    if not resume and checkpoint.stem.lower() != str(cfg["architecture"]).lower():
        ap.error("architecture 与预训练权重文件名不符")
    model = YOLO(str(checkpoint))
    project = resolve(cfg["project"])
    if resume:
        run_dir = checkpoint.parent.parent
        if run_dir.parent != project or checkpoint.parent != run_dir / "weights":
            ap.error("resume checkpoint 必须位于本工程 outputs/experiments/<实验ID>/weights/last.pt")
        meta_path = run_dir / "experiment.json"
        if not meta_path.is_file():
            ap.error("resume 实验元信息不存在")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_fingerprint", {}).get("sha256") != dataset_fp["sha256"]:
            ap.error("resume 数据集 fingerprint 已变化")
        saved_cfg = metadata.get("hyperparameters", {})
        if {k: v for k, v in saved_cfg.items() if k != "resume"} != {k: v for k, v in cfg.items() if k != "resume"}:
            ap.error("resume 配置与原实验不一致；仅允许将 resume 设置为 last.pt 路径")
        if metadata.get("pretrained_checkpoint_sha256") != sha256(resolve(cfg["pretrained_checkpoint"])):
            ap.error("resume 对应的初始预训练权重 SHA256 已变化")
        metadata.setdefault("resume_history", []).append({"datetime": timestamp(), "checkpoint": str(checkpoint),
                                                               "checkpoint_sha256": sha256(checkpoint)})
        write_json(meta_path, metadata)
    else:
        experiment_id = args.experiment_id
        if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
            ap.error("experiment_id 只允许字母、数字、下划线和横线")
        run_dir = project / experiment_id
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(resolve(args.config), run_dir / "train_config.yaml")
        metadata = {"experiment_id": experiment_id, "datetime": timestamp(), "model": cfg["architecture"],
                    "pretrained_checkpoint": str(checkpoint), "pretrained_checkpoint_sha256": sha256(checkpoint),
                    "dataset_fingerprint": dataset_fp,
                    "class_names": names, "hyperparameters": cfg, "dependency_versions": versions(),
                    "gpu": torch.cuda.get_device_name(0), "checkpoint_paths": {}, "checkpoint_sha256": {},
                    "best_epoch": None,
                    "evaluation_metrics": None}
        write_json(run_dir / "experiment.json", metadata)
    try:
        if resume:
            # Ultralytics public resume interface uses the checkpoint path.
            model.train(resume=str(checkpoint))
        else:
            model.train(data=str(data), epochs=int(cfg["epochs"]), batch=int(cfg["batch"]),
                        imgsz=int(cfg["imgsz"]), device=0, workers=int(cfg["workers"]),
                        seed=int(cfg["seed"]), optimizer=cfg["optimizer"], lr0=float(cfg["lr0"]),
                        patience=int(cfg["patience"]), amp=bool(cfg["amp"]),
                        warmup_epochs=float(cfg.get("warmup_epochs", 3)),
                        close_mosaic=int(cfg.get("close_mosaic", 10)),
                        deterministic=bool(cfg.get("deterministic", True)), cache=bool(cfg.get("cache", False)),
                        pretrained=True,
                        project=str(project), name=run_dir.name,
                        # The empty run directory was exclusively reserved above with exist_ok=False.
                        # Ultralytics needs True here or it auto-increments the output name.
                        exist_ok=True, save=True)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA 显存不足；checkpoint 已保留。请手动调整 batch/imgsz 后新建实验或恢复训练。") from exc
    finally:
        meta_path = run_dir / "experiment.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata["checkpoint_paths"] = {k: str(run_dir / "weights" / f"{k}.pt")
                                            for k in ("best", "last") if (run_dir / "weights" / f"{k}.pt").exists()}
            metadata["checkpoint_sha256"] = {k: sha256(Path(path)) for k, path in metadata["checkpoint_paths"].items()}
            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                import csv
                with results_csv.open(newline="", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    cleaned = [{k.strip(): v for k, v in row.items()} for row in rows]
                    metric = "metrics/mAP50-95(B)"
                    if metric in cleaned[0]:
                        best = max(cleaned, key=lambda r: float(r.get(metric, 0) or 0))
                        metadata["best_epoch"] = int(best["epoch"])
                        metadata["evaluation_metrics"] = {k: v for k, v in best.items() if k.startswith("metrics/")}
            write_json(meta_path, metadata)
    print("实验目录:", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
