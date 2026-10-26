# 智能割草机视觉障碍物检测｜Mower Vision

> YOLO26s · PyTorch 训练与评估 · ONNX 导出 · 面向 NVIDIA Jetson Orin NX 的部署准备  
> 当前状态：**V1.0 源码交付准备；Jetson 部署待完成**。当前结果不能作为已经完成安全验证的自主避障系统证据。

## 1. 项目简介

本项目面向智能割草机的前向视觉感知，目标是从摄像头图片中检测障碍物并输出边界框（bbox）、置信度和后续系统可使用的障碍物信息。**业务侧只关心是否检测到障碍物，不要求正确区分障碍物类型。**

开发训练时保留四个原始标注类别：

| 类别 ID | 类别名称 | 业务输出 |
| --- | --- | --- |
| 0 | `stone` | `obstacle` |
| 1 | `tree` | `obstacle` |
| 2 | `person` | `obstacle` |
| 3 | `other_roadblock` | `obstacle` |

因此，**主模型仍是四类别 YOLO26s，但最终向割草机业务模块统一输出 `obstacle`**。不要把类别误判与障碍物完全漏检混为一谈。

### 当前完成范围

- [x] 训练图片整理、标注导出、数据来源审查
- [x] 四类别 YOLO26s 模型训练及模型效果评估
- [x] 单类别 `obstacle` 模型对照训练及评估
- [x] 类别无关（class-agnostic）障碍物检测评估
- [x] 四类别主模型 ONNX 导出，ONNX Checker 与 ONNX Runtime CPU 验证
- [x] 11 张验证图上的 PyTorch/ONNX 一致性对照
- [ ] Jetson Orin NX 上构建 TensorRT Engine
- [ ] 目标设备 TensorRT 输出与 ONNX 结果一致性验证
- [ ] 摄像头实时视频流接入、坐标转换、检测接口集成
- [ ] 实际运行延迟、FPS、内存、功耗测试
- [ ] 独立采集场景、负样本、人员安全相关验证与实车验收

**Jetson 部署由接手开发者负责；本 README 第 7–8 节列出接口和后续验证工作。**

### 当前实际支持的输入与输出入口

| 入口 | 当前输入 | 当前输出 | 边界 |
| --- | --- | --- | --- |
| `scripts/predict.py`（`-Task predict`） | 本地单张图片，或图片目录下的 `.jpg/.jpeg/.png/.bmp/.webp` 文件；`-Model` 指向本地权重 | 绘制框的图片、`predictions.json`、`detections.csv`；框坐标是原图像素 `xyxy`，仍带原四类别 | 目录只读取当前层；未实现摄像头流或统一 `obstacle` 业务消息 |
| `scripts/train.py`（`-Task train`） | `configs/train.yaml`、YOLO 数据集、官方本地预训练 `yolo26s.pt` | 独立实验目录中的权重、曲线及实验元信息 | 数据与权重不随源码仓库提供 |
| `scripts/evaluate.py` / `scripts/evaluate_obstacle.py` | 模型权重、`data.yaml` 指定的图片和配对 GT 标签；默认 Val | 四类别指标，或类别无关 TP/FP/FN、阈值报告及可视化 | Test 评估需显式授权参数；当前 Val 不代表跨会话泛化 |
| `scripts/export.py`（`-Task export`） | 四类别 `.pt` 权重和一致性验证图片 | 静态 FP32 ONNX、`export.json`、`consistency_report.md` | 导出文件是模型与验证记录，不是实时推理服务 |

摄像头视频帧、ROS 2 消息、Mid-360 雷达输入及面向车辆的检测消息均**尚未接入**。接手者可把视频帧当作原始图像走第 7 节的预处理，但当前命令行脚本不读取视频流。

## 2. 选用的模型及交付文件

| 项目 | 主模型（V1） |
| --- | --- |
| 训练架构 | Ultralytics YOLO26s Detect |
| 来源 | 官方 `yolo26s.pt` 初始化后微调 |
| 权重 | `best.pt`（四类别主模型） |
| 部署格式 | `best.onnx`，ONNX opset 18 |
| 输入 | `images`，`float32 [1,3,640,640]` |
| 输出 | `output0`，`float32 [1,8,8400]` |
| 模型内部 NMS | **未内置**，需在部署端完成 |
| 推荐用途 | 工程联调、Jetson 部署原型与进一步验证 |

### 已在本地校验的文件哈希

| 文件 | SHA-256 |
| --- | --- |
| 四类别模型 `best.pt` | `f3c64cdfb74f58a4d991ec180fa5d83aab2a3b959e08612563bff0329669c700` |
| 四类别模型 `best.onnx` | `20de9e5602cd5eb64133a4f7a1e985f7bd747651cfcc1f0754da4db2ec948564` |

本地来源路径（仅作溯源；**接手者不应依赖此 Windows 绝对路径**）：

```text
outputs/experiments/pilot_yolo26s_001/weights/best.pt
outputs/exports/yolo26s_main_20261015_211448/best.onnx
outputs/exports/yolo26s_main_20261015_211448/export.json
outputs/exports/yolo26s_main_20261015_211448/consistency_report.md
```

**模型下载入口：**本仓库 [GitHub Releases](../../releases)。当前尚未发布 Release，链接内暂时没有模型；发布后应下载四类别 `best.onnx`，需要继续训练或复核时再下载 `best.pt`，并用上表 SHA256 核对。`outputs/` 整体不进入普通 Git 历史。原始 `export.json` 含本机绝对路径、验证图片名称/哈希及逐图预测坐标，对外公开前须审查或脱敏；部署接口以本 README 为准。

> 单类别备选模型：`outputs/experiments/pilot_yolo26s_obstacle_001/weights/best.pt`。它用于科研/工程对照，不是 V1 默认部署模型；若不需要两套模型，无须向部署接手者分发。

## 3. 仓库结构

以下是计划公开的文件结构，已对照当前源码核验。数据、权重、输出及本地 `docs/` 由 `.gitignore` 排除；本地文件不删除。

```text
mower-vision/
├── .gitignore
├── README.md
├── environment.yml               # Python 3.11 基础环境声明
├── requirements.txt              # 原开发环境的 pip 版本记录
├── run_vision.ps1                 # Windows 任务入口；含 cuDNN PATH 隔离
├── configs/
│   ├── smoke.yaml
│   ├── train.yaml
│   ├── train_obstacle.yaml
│   └── datasets/mower_v1.yaml    # 原导出目录的本机路径配置
├── scripts/
│   ├── common.py                  # 类别、路径、哈希及数据 fingerprint
│   ├── check_env.py               # 开发环境检查
│   ├── check_model.py             # 本地权重加载检查
│   ├── validate_dataset.py        # 数据集检查
│   ├── prepare_dataset.py         # 数据工作副本准备
│   ├── audit_provenance.py        # 原始会话与划分来源审查
│   ├── split_dataset.py           # 后续新增数据的分组划分
│   ├── train.py                   # YOLO26s 训练
│   ├── evaluate.py                # 标准四类别评估
│   ├── evaluate_obstacle.py       # 类别无关 TP/FP/FN/阈值评估
│   ├── predict.py                 # 图片预测
│   ├── export.py                  # ONNX 导出与一致性验证
│   ├── prepare_obstacle_dataset.py  # 四类标签转换为单类
│   ├── analyze_obstacle_failures.py  # FP/FN 复核
│   └── compare_obstacle_evaluations.py # 两模型对照
└── tests/                         # 两组 CPU 单元测试
```

## 4. 数据集与实验设计

本轮模型基于 **105 张有效图片**，按原始数据划分：Train 89、Val 11、Test 5。源标注为四类，包含 stone、tree、person、other_roadblock。

重要的数据局限：

- 图片来自 **4 个采集会话**；旧 Train/Val/Test 均跨越相同会话，**验证性能不能视为独立采集场景的泛化性能**。
- Val 只有 **2 个人员实例**，不能据此证明人员检测安全性。
- Test 只有 **5 张且没有 person**；未用 Test 选择阈值或模型，现有划分本身也不独立于训练会话。
- 当前数据缺少人工确认的纯背景 negative 图片，误报警率缺少真实场景证据。
- 目标中存在小目标、复杂草地背景等困难样本；后续应针对漏检和误检补采。
- FP 是相对于现有标注的未匹配框，部分可能对应漏标目标，需人工复核。
- 原始机场图片及标注文件**不随源码仓库分发**；如需复训，应经数据负责人确认权限并另行提供数据及来源说明。

### 训练和评估的数据格式

`configs/train.yaml` 指向工作副本 `datasets/mower_v1_snapshot/data.yaml`，并指定模型架构、`models/yolo26s.pt`、epoch 上限、batch、`imgsz=640`、设备、优化器、随机种子等。实际工作副本按如下结构组织；图片和标签以相同文件主名一一配对：

```text
datasets/mower_v1_snapshot/
├── data.yaml
├── images/{train,val,test}/<image-name>.<image-extension>
├── labels/{train,val,test}/<image-name>.txt
├── manifest.csv、split_manifest.csv、dataset_info.json
└── snapshot_report.json
```

`data.yaml` 中的 `path` 必须指向**本机实际存在**的数据集根目录；以下是格式示意，克隆仓库不会自动取得数据：

```yaml
path: /replace/with/local/mower_v1_snapshot
train: images/train
val: images/val
test: images/test
nc: 4
names: {0: stone, 1: tree, 2: person, 3: other_roadblock}
```

每个 YOLO TXT 目标占一行：`class_id x_center y_center width height`。类别 ID 为 0–3；四个 bbox 数值相对于**原始图片宽高归一化到 0–1**，不是像素，也不是 ONNX 输出坐标。例如 `2 0.50 0.50 0.10 0.20` 仅示意一个 `person` 框。数据校验脚本检查类别、坐标范围、几何边界、图片配对和来源状态；原始标注与工作副本均不在公开 Git 仓库中。

实际四类别训练目录 `outputs/experiments/pilot_yolo26s_001/` 含 `weights/best.pt`（按验证指标选出的权重）、`weights/last.pt`（最后训练状态）、逐 epoch 的 `results.csv`、曲线 `results.png` 和 `experiment.json`（配置、依赖、GPU、数据 fingerprint、权重哈希与最佳 epoch）。其他实验需用新的 `-ExperimentId` 建立独立目录。标准四类别评估输出 `evaluation.json`、PR/混淆矩阵图及汇总 CSV；类别无关评估输出 `threshold_metrics.csv/.json`、`per_image_metrics.csv`、`false_positives.csv`、`false_negatives.csv`、曲线、`report.md` 和可视化图片。评估输入是权重、`data.yaml`、对应 split 的图片/GT 标签及阈值；类别无关的 Precision、Recall、TP/FP/FN 和 FP/image 使用预测框与 GT 重新匹配计算，不由四类 mAP 推算。

## 5. 当前结果：必须按“检测到任意障碍物”理解

本项目对照过两条路线：

1. **主模型：**四类别训练；预测时四类统一计为 `obstacle`。
2. **备选模型：**标签转换后直接训练单类别 `obstacle`。

类别无关评估规则：所有 GT 和预测类别统一为 obstacle；预测框按置信度降序与尚未匹配的 GT 一对一匹配；**IoU ≥ 0.5 算 TP**。未匹配预测计 FP，未匹配 GT 计 FN。

在 **Val 11 张、GT 120 个**上的结果（匹配 IoU≥0.5；不是独立测试性能，来源为两个 `outputs/evaluation/*/report.md`）：

| 模型 | conf | TP/FP/FN | Precision | Recall | F1 | FP/图片 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 四类别主模型 | 0.05 | 97/194/23 | 33.3% | 80.8% | 0.472 | 17.64 |
| 四类别主模型 | 0.10 | 88/97/32 | 47.6% | 73.3% | 0.577 | 8.82 |
| 四类别主模型 | 0.25 | 74/19/46 | 79.6% | 61.7% | 0.695 | 1.73 |
| 四类别主模型 | 0.40 | 59/7/61 | 89.4% | 49.2% | 0.634 | 0.64 |
| 单类别备选模型 | 0.10 | 96/119/24 | 44.7% | 80.0% | 0.573 | 10.82 |
| 单类别备选模型 | 0.25 | 74/32/46 | 69.8% | 61.7% | 0.655 | 2.91 |

四类别主模型的 `experiment.json` 记录 best epoch 为 41；`results.csv` 最后一轮为 56，标准四类检测 mAP50 为 0.64492（它**不是**类别无关 Recall）。单类别模型在低阈值下召回较高，同时 FP/图片也较高；V1 选用四类别模型，单类别保留作备选。

**重要：**`conf=0.25` 是工程对照时使用的阈值，不是已验证的安全部署阈值。实际部署必须根据场景、漏检代价和人工核验调整。

## 6. 本地开发与实验复现（Windows）

已验证的原开发环境：Windows 11、RTX 5060 Laptop GPU、Python 3.11.17、PyTorch `2.13.0+cu130`、torchvision `0.28.0+cu130`、Ultralytics `8.4.174`、ONNX `1.23.2`、ONNX Runtime `1.31.0`（CPU Provider）。

> `environment.yml` 只声明 Python 3.11 基础环境；`requirements.txt` 是原 Windows 开发环境的 pip 版本记录，含 CUDA 专用 PyTorch 构建，不能当作 Jetson 的一键安装清单。此处只说明现有环境，不自动安装或改动环境。

```powershell
# 在已装好依赖、名称为 mower-vision 的 Conda 环境中执行
conda activate mower-vision
# 先在本机进入克隆后的 mower-vision 仓库根目录

# 基础检查
.\run_vision.ps1 -Task env -CudaConv

# 数据检查（需要另外取得授权的数据工作副本）
.\run_vision.ps1 -Task validate -Data datasets\mower_v1_snapshot\data.yaml

# 重新训练需要完整工作副本、官方预训练 yolo26s.pt、来源审查报告
.\run_vision.ps1 -Task check-model -Model models\yolo26s.pt
.\run_vision.ps1 -Task train -Config configs\train.yaml -ExperimentId my_new_run_001

# 类别无关评估，仅在已具备标签和数据集时进行
.\run_vision.ps1 -Task obstacle-eval `
  -Model outputs\experiments\pilot_yolo26s_001\weights\best.pt `
  -Data datasets\mower_v1_snapshot\data.yaml `
  -Split val `
  -Out outputs\evaluation\my_validation

# 单张图片推理：输入本机图片路径；输出仍保留原四类别
$imagePath = Read-Host '本地图片文件路径'
.\run_vision.ps1 -Task predict `
  -Model outputs\experiments\pilot_yolo26s_001\weights\best.pt `
  -Source $imagePath -Out outputs\predictions -Conf 0.25 -Iou 0.7 -ImageSize 640

# 目录批量推理：只读取该目录当前层的支持格式图片
.\run_vision.ps1 -Task predict `
  -Model outputs\experiments\pilot_yolo26s_001\weights\best.pt `
  -Source datasets\mower_v1_snapshot\images\val -Out outputs\predictions

# 标准四类别 Val 指标；Test 必须显式 -Split test -AllowTest
.\run_vision.ps1 -Task evaluate `
  -Model outputs\experiments\pilot_yolo26s_001\weights\best.pt `
  -Data datasets\mower_v1_snapshot\data.yaml -Split val -Out outputs\evaluations

# 如需从本地 .pt 再导出并比较 PyTorch/ONNX，必须提供验证图片
.\run_vision.ps1 -Task export `
  -Model outputs\experiments\pilot_yolo26s_001\weights\best.pt `
  -Source datasets\mower_v1_snapshot\images\val -Out outputs\exports -ImageSize 640

# 不加载模型的轻量单元测试
python -m unittest discover -s tests -v
```

训练副本及权重不在 Git 仓库中；上面的训练与评估命令仅在已获授权取得数据、权重和来源报告后可执行。`configs/datasets/mower_v1.yaml` 与 `prepare_dataset.py` 的默认源目录仍对应原开发机的导出位置，其他机器需先按实际数据位置适配并重新核验。`run_vision.ps1` 要求激活的 Conda 环境目录名为 `mower-vision`，只在子进程 PATH 中隔离系统 cuDNN 目录，并在退出时恢复。Jetson 不使用 PowerShell 入口。

`-Task predict` 由 Ultralytics 完成内部预处理和坐标恢复，`predictions.json` / `detections.csv` 中仍是四类别结果；它**不会**自动产生第 7.4 节建议的业务消息。`-Task export` 输入 `.pt` 与验证图片，创建独立目录，保存 `.onnx`、`export.json`、`consistency_report.md`（以及导出时复制的 `.pt`）；已有校验通过的主 ONNX 见第 2 节，无需为了部署重复导出。

## 7. ONNX 接口与 Jetson 对接规范（重点）

### 7.1 从图片到模型输入张量

```text
input name: images
shape:      [1, 3, 640, 640]
dtype:      float32
```

`[N,C,H,W]=[1,3,640,640]` 分别表示一次一张图、RGB 三通道、高 640、宽 640。ONNX 的 `images` 接口接收**连续内存中的 FP32 浮点张量**，不是 JPEG/PNG 文件字节，也不是 OpenCV 的 BGR `uint8 [H,W,3]` 数组。原图分辨率可以变化，须先变成固定的 640×640；像素除以 255 后通常在 `[0,1]`。

`scripts/export.py` 用于已验证的 PyTorch/ONNX 一致性路径，其实际预处理顺序如下：

1. `cv2.imread(..., cv2.IMREAD_COLOR)` 解码成原图大小 `H₀×W₀×3` 的 **BGR** 数组。
2. 使用 Ultralytics 8.4.174 的 `LetterBox(new_shape=(640,640), auto=False, stride=模型 stride)`。该版本在此调用中的默认值是 `scale_fill=False`、`scaleup=True`、`center=True`、`padding_value=114`、`interpolation=cv2.INTER_LINEAR`。它保持宽高比，不强行拉伸；`auto=False` 表示补足完整 640×640，而不是只补到 stride 的倍数。先缩放，再在左右/上下居中填充；奇数像素的两侧分配以该版本的 `round(half±0.1)` 为准。
3. 将填充后的 BGR 转为 **RGB**，由 HWC 转为 CHW 并复制成连续数组；转 `float32` 后逐元素除以 `255.0`，最后加 batch 维，得到 `[1,3,640,640]`。

例如原图 **宽 1280、高 720**：`r=min(640/1280,640/720)=0.5`，缩放后为 **宽 640、高 360**；竖向总空白 `640−360=280`，居中填充上 140、下 140，左右均为 0。填充区的原始 BGR 值为 `(114,114,114)`，归一化后各通道约为 `0.4471`。模型实际看见的是含上下灰边的 640×640 图片；第 7.5 节用同一组尺寸演示框坐标还原。

`scripts/predict.py` 调用 `YOLO.predict`，由 Ultralytics 内部管理预处理并把结果框映回原图；该接口在单图条件下可能采用最小矩形填充，**不能据此假定其内部张量总是 `[1,3,640,640]`**。Jetson 上要复现本次 ONNX 一致性验证，应以 `scripts/export.py` 的 `auto=False` 静态预处理为准，并保留实际缩放尺寸和各边 padding。

### 7.2 原始输出

```text
output name: output0
shape:       [1, 8, 8400]
layout:      [batch, channel, candidates]
channel 0-3: cx, cy, w, h（640×640 输入图坐标，像素单位）
channel 4:   stone score
channel 5:   tree score
channel 6:   person score
channel 7:   other_roadblock score
```

`output0` 是 **FP32**，共 8400 个候选；读取第 `k` 个候选要取 `output0[0, :, k]`，而不是 `output0[0, k, :]`。前四项是已解码的中心点和宽高，单位为 **640×640 模型输入图上的像素**，不是 YOLO TXT 的 0–1 归一化值，也不是原图坐标。后四项是经过 sigmoid 的类别分数，顺序固定为表中四类；**没有独立 objectness 通道**，因此不应再乘一个假想的 objectness 分数。原始输出没有内置 NMS，也不是最终检测框。当前 `export.json` 记录 `exported_end2end=false`，应按本节的原始候选和外部 NMS 处理，不能直接套用其他 YOLO26 端到端 top-k 输出格式。

例如下列 **纯示意数值，不是真实推理结果**：某个候选 `output0[0,:,k] = [95, 210, 70, 60, 0.91, 0.08, 0.02, 0.12]`。它在模型输入图上的框中心为 `(95,210)`，宽 70、高 60；最大分数 `0.91` 对应原模型类别 `stone`。即使最终业务名称写为 `obstacle`，也要先按四类别模型的分数和 NMS 规则处理它。

### 7.3 后处理

已验证的 Windows 路径在 `scripts/export.py` 中对 PyTorch 与 ONNX 的**同一原始张量**调用 Ultralytics 8.4.174 的 `non_max_suppression(..., conf_thres=0.25, iou_thres=0.7, agnostic=False, max_det=300, nc=4)`；本版 ONNX 无内置 NMS。部署端应按相同次序处理：

1. 对每个候选读取四个类别分数，选最大分数及其类别 ID；以置信度阈值筛掉低分候选。最终 `confidence` 就是被选中的**原模型类别分数**。
2. 将模型输入图中的中心框 `(cₓ,cᵧ,w,h)` 转成 `(x₁,y₁,x₂,y₂)=(cₓ−w/2,cᵧ−h/2,cₓ+w/2,cᵧ+h/2)`，按**原四类别**执行类别感知 NMS，IoU 阈值 0.7，最多保留 300 框。`conf=0.25` 是现有一致性测试条件，**不是部署最优或安全阈值**。
3. 把 NMS 保留框从 640×640 letterbox 坐标逆变换到**原始图片像素坐标**，再裁剪至 `[0,W₀]×[0,H₀]`。`scripts/predict.py` 的 `result.boxes.xyxy` 已由 Ultralytics 做了这一步，调用者不得再次逆变换。
4. 对外只把最终有效框统一标记为 `obstacle`；可选地保留原类别 ID/名称作诊断。**四类别模型并未因此变成单类别网络。**

类别感知 NMS 只抑制**同一原类别**中高度重叠的预测，不同类别的重叠框可能同时保留。类别无关 NMS 则可能跨类别抑制，改变相邻障碍物、重叠障碍物的召回和重复框数量。现有 `evaluate_obstacle.py` 是在模型预测框之后忽略类别进行 GT 一对一匹配，并非把推理 NMS 改为类别无关；Jetson 端不得未经逐图核验就改变 NMS 或再额外叠加一次 NMS。

#### 四种坐标表示与逆变换

| 表示 | 原点 / 方向 | 单位与含义 |
| --- | --- | --- |
| 原图坐标 `(xₒ,yₒ)` | 左上角 `(0,0)`，x 向右、y 向下 | 原始 `W₀×H₀` 图片中的像素；最终框应在此坐标系 |
| 模型输入坐标 `(xₘ,yₘ)` | 640×640 letterbox 图左上角 `(0,0)`，方向同上 | 含填充区域的像素坐标，不等于原图像素 |
| 原始输出 `xywh` | `(cₓ,cᵧ)` 为框中心 | `w,h` 是模型输入图中的框宽和高，不是右下角坐标 |
| 最终 `xyxy` | `(x₁,y₁)` 左上、`(x₂,y₂)` 右下 | 逆变换和裁剪后的原图像素；`width=x₂−x₁`、`height=y₂−y₁` |

对于第 7.1 节的静态 `auto=False` 预处理，令 `r=min(640/W₀,640/H₀)`，`Wᵣ=round(W₀r)`、`Hᵣ=round(H₀r)`。左填充 `L=round((640−Wᵣ)/2−0.1)`，上填充 `T=round((640−Hᵣ)/2−0.1)`；右、下填充由剩余像素确定。为匹配 Ultralytics `scale_boxes` 的实际计算，定义 `gₓ=Wᵣ/W₀`、`gᵧ=Hᵣ/H₀`。框的每个角点按以下方式逆变换并裁剪：

```text
x_original = clip((x_model - L) / gₓ, 0, W₀)
y_original = clip((y_model - T) / gᵧ, 0, H₀)
```

这样先**减去 padding**，再除缩放比。坐标保持浮点像素值；绘制或传给只接受整数的接口时，再由接入方约定取整方式。完整数值例子见第 7.5 节。

### 7.4 建议的对外障碍物消息（当前尚未实现）

现有 `predict.py` 的 JSON/CSV 保存的是原四类别图片预测，**不会**自动生成下面的消息。接手者可在视频帧接入后使用以下建议结构；示例数值仅供说明：

```json
{
  "frame_id": 1001,
  "timestamp": 0.0,
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    {
      "label": "obstacle",
      "confidence": 0.91,
      "bbox_format": "xyxy",
      "bbox": [120, 80, 260, 200],
      "model_class_id": 0
    }
  ]
}
```

`bbox` 是**原始图片像素**的左上、右下角，`confidence` 来自保留预测框的最大原类别分数；`model_class_id` 是可选诊断字段，业务 `label` 一律为 `obstacle`。无检测时用 `"detections": []`，仍保留图像尺寸、帧号和时间字段。`frame_id` 如何生成、`timestamp` 用什么时钟及什么单位、消息编码/传输、坐标取整和目标跟踪 ID 等尚需接入方约定；本仓库没有这些运行时接口。

本模型只给出**二维图像检测框**。框大小或位置不能直接推出距割草机的真实距离（米）、三维位置、地图坐标、运动速度或安全避障决策；这些需要相机标定、测距/雷达、其他感知与决策模块，且仍需系统级验证。

### 7.5 从一张图片到 obstacle 检测框：完整推理示例

以下数值为**可核算的示意**，不是对某张真实图片的推理结果；实际候选数、置信度与框由模型决定。

1. 输入一张 **1280×720** 的原图，OpenCV 解码为 `uint8 BGR [720,1280,3]`。
2. `LetterBox(640×640, auto=False)` 得到 `r=0.5`，先缩放为 **640×360**，再在上、下各填充 **140 像素**（填充值 114），左、右无需填充，保留长宽比。
3. BGR→RGB、HWC→CHW、转连续 FP32、除以 255、增加 batch 维，形成 `images [1,3,640,640]`。
4. ONNX 返回 `output0 [1,8,8400]`。假设某候选为第 7.2 节的示意值 `[95,210,70,60,0.91,0.08,0.02,0.12]`；选出的原类别为 `stone`，分数 `0.91`，超过 `conf=0.25`。
5. 中心框转换为模型输入图 `xyxy=[60,180,130,240]`；**假设**它通过类别感知 NMS（IoU 0.7、最多 300 框）。真实 NMS 是否保留它，取决于其他候选，不能只凭这一行判断。
6. 减去左填充 0 和上填充 140，再除以 `gₓ=gᵧ=0.5`：原图框 `[(60−0)/0.5,(180−140)/0.5,(130−0)/0.5,(240−140)/0.5]=[120,80,260,200]`，裁剪到原图范围后不变。
7. 最终向业务侧建议输出 `label="obstacle"`、`confidence=0.91`、`bbox_format="xyxy"`、`bbox=[120,80,260,200]`。可在原始 1280×720 画面上绘制此框，或封装成第 7.4 节的消息交给其他模块；若没有保留框，检测列表为空。

简化流程：`BGR 原图 → 固定 letterbox → RGB/NCHW/FP32/255 → ONNX 原始候选 → 四类分数选最大值 → 置信度过滤 → xywh→xyxy → 类别感知 NMS → 逆 letterbox / 裁剪 → obstacle 消息`。实际实现可参照 `scripts/export.py` 的预处理、NMS 和 `scripts/predict.py` 的原图坐标返回；不要把上面的示意候选当作固定测试向量。

### 7.6 ONNX 验证状态

已完成：11 张 Val 图 PyTorch/ONNX 对比，两端在 `conf=0.25, iou=0.7, max_det=300` 后均输出 **100 个检测框**，匹配框最低 IoU **0.999993**，最大置信度差约 **2.03e-6**；ONNX Checker 与 ONNX Runtime CPU 推理通过。

**未完成：**Jetson TensorRT FP16 的构建、数值一致性、性能、摄像头实时流及实车验证。不要将上述一致性结果解释为 Jetson 已部署成功。

## 8. 接手开发者的下一步

接手开发者按以下顺序开展 Jetson 工作：

1. 确认 Jetson 实际 JetPack / L4T、CUDA、TensorRT、设备存储与内存。
2. 从本仓库 [GitHub Releases](../../releases) 下载 ONNX 和可选的 `.pt`，核对本 README 的 SHA256；当前尚未发布 Release。逐图导出报告须经内容审查后另行提供。
3. 在目标 Jetson 上先验证 FP32 TensorRT 对照，再尝试 FP16、batch=1、640×640 静态 Engine。
4. 先使用固定测试图片验证 TensorRT 原始输出和后处理一致性。
5. 实现并测试摄像头输入、业务输出、延迟/FPS/功耗/内存监控。
6. 补采**独立会话、背景负样本、人员及极端场景**，完成部署前风险评估。

Jetson 联调时，应对**同一批原始图片**复用本 README 第 7.1 节的静态预处理，核对输入张量的 RGB 顺序、尺寸、填充、FP32 归一化以及模型/Engine 版本；再分别比对 ONNX 与 TensorRT 的原始输出、NMS 后四类别框、逆变换后的原图坐标和最终 `obstacle` 消息。若改用 FP16，须另行记录可接受的数值及检测差异，不能把 Windows CPU Provider 的一致性直接视为 Jetson 一致性。

**仍未实现：**Jetson TensorRT Engine 构建及板端一致性验证、摄像头实时采集、ROS 2 或其他系统消息接口、Mid-360 雷达与视觉联合感知、实车可靠性和安全性验证。接手者需在完成这些工作后，才可把第 7.4 节的建议格式落成正式接口。

## 9. 许可、数据安全及限制

- 项目使用 Ultralytics YOLO26。其[官方许可说明](https://www.ultralytics.com/license)指出，YOLO 代码及训练出的模型默认涉及 **AGPL-3.0**，闭源或商业部署需审查 Enterprise 授权；不能自行将第三方权重改标为 MIT/Apache。
- 本项目计划使用 **Public GitHub 仓库**。公开源码和 Release 权重前，须确认项目自有源码的授权、Ultralytics 许可、数据权属与隐私；机场原始图片和真实标注不进入公开仓库或 Release。
- 当前仓库**没有根目录 LICENSE**。要让其他开发者依法修改、分发并继续开发，应由项目权利人确认与 Ultralytics 授权兼容的项目许可并加入正式 LICENSE；在此之前不能把“Public”误称为已完成开源授权，也不要擅自添加 MIT/Apache LICENSE。
- 该项目是视觉目标检测原型，不承担完整安全避障、制动或路径规划保证。特别是人员与小目标漏检风险，必须有独立测试和保守的系统级安全策略。

## 10. 版本与交接记录

- **V1.0（2026-10）**：YOLO26s 四类别主模型、单类别对照、障碍物统一评估、ONNX 转换及一致性验证已完成；Jetson 部署待接手开发者完成。
- GitHub 仓库、Release 和 tag：**当前尚未创建/发布**；发布后补充实际链接与版本号。
- 交接负责人 / 接手人 / 联系方式 / 项目仓库 URL：**发布前补充**。

---

**本 README 对照当前源码、实验记录与模型文件编写。发布模型、原始报告或数据仍须由权利人完成授权与内容审查。**
