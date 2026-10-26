"""Load only a local YOLO checkpoint to confirm installed Ultralytics compatibility."""
import argparse
from common import resolve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    path = resolve(args.model)
    if not path.is_file() or path.suffix.lower() != ".pt":
        ap.error(f"本地 .pt 不存在: {path}。请自行取得与 Ultralytics 8.4.174 兼容的官方权重，放入 models 目录。")
    from ultralytics import YOLO
    model = YOLO(str(path))
    print("加载成功:", path, "task:", model.task, "parameters:", sum(p.numel() for p in model.model.parameters()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
