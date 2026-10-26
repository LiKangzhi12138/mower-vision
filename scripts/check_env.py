"""Explicit runtime diagnostics; no model loading or network calls."""
import argparse
import os
from pathlib import Path

from common import versions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-conv", action="store_true", help="小型 cuDNN 前向及反向检查")
    args = ap.parse_args()
    print("依赖版本:", versions())
    import torch
    print("Python:", os.sys.executable)
    print("CUDA available:", torch.cuda.is_available())
    print("cuDNN:", torch.backends.cudnn.version(), "enabled:", torch.backends.cudnn.enabled)
    if not torch.cuda.is_available():
        return 2
    print("GPU:", torch.cuda.get_device_name(0))
    if args.cuda_conv:
        layer = torch.nn.Conv2d(3, 4, 3).cuda()
        x = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
        layer(x).square().mean().backward()
        torch.cuda.synchronize()
        print("CUDA/cuDNN 卷积前向与反向通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
