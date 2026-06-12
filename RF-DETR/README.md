# RF-DETR Crack Detection

基于 `rfdetr==1.7.1` 的裂纹 bbox 检测方案。当前仓库保留一套主数据集
`RF-DETR/data/full_valid_seed42` 和一套主实验
`RF-DETR/runs/v100_4gpu_seg_large_bboxval`。

本项目提交格式只需要 `predict_bboxes`，不提交 mask。训练仍使用
`RFDETRSegLarge`，但验证和评估走 bbox-only 路径，避免 RF-DETR-Seg 在大图上把
mask 上采样回原图导致显存溢出。

## 环境配置

推荐使用 Conda 创建独立环境。以下命令默认在仓库根目录执行：

```bash
git clone <repo-url>
cd cctech_q4

conda create -n crack python=3.10 -y
conda activate crack
python -m pip install --upgrade pip
python -m pip install -r RF-DETR/requirements.txt
```

`RF-DETR/requirements.txt` 是从可运行的 `crack` 环境导出的完整依赖快照，并在文件头部保留了
PyTorch CUDA 12.6 wheel 源：

```text
--extra-index-url https://download.pytorch.org/whl/cu126
```

关键依赖版本：

- `torch==2.10.0+cu126`
- `torchvision==0.25.0+cu126`
- `rfdetr==1.7.1`

CUDA 注意事项：

- 本项目训练配置在 CUDA GPU 上验证；CPU 不适合作为训练环境。
- 当前依赖锁定为 PyTorch CUDA 12.6 wheel。NVIDIA 驱动需要支持 CUDA 12.6 runtime。
- 如果机器驱动较旧，优先升级驱动；不要直接安装 CUDA 13 构建的 PyTorch，否则可能在模型迁移到 GPU 时失败。

安装后验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import rfdetr; print('rfdetr ok')"
```

期望第一条命令输出包含 `2.10.0+cu126 12.6`，且 `torch.cuda.is_available()` 为 `True`。

## 当前数据

原始官方数据：

```text
dataset/trainval/trainval.json  # 1285 张训练验证图，1652 个标注
dataset/trainval/images/
dataset/test/test.json          # 301 张测试图，无标注
dataset/test/images/
```

当前 RF-DETR 数据集：

```text
RF-DETR/data/full_valid_seed42/
├── train/
│   ├── images/                  # 1028 张，1287 个标注
│   └── _annotations.coco.json
├── valid/
│   ├── images/                  # 257 张，365 个标注
│   └── _annotations.coco.json
└── dataset_manifest.json
```

`full_valid_seed42` 是按 `seed=42` 和 `val-ratio=0.2` 从官方 `trainval` 导出的完整
train/valid 划分。图像默认硬链接优先，失败时软链接，再失败复制。

重新导出命令：

```bash
python RF-DETR/scripts/prepare_dataset.py \
  --raw-json dataset/trainval/trainval.json \
  --image-root dataset/trainval \
  --out-dir RF-DETR/data/full_valid_seed42 \
  --val-ratio 0.2 \
  --seed 42 \
  --link-mode auto
```

## 训练

当前主实验目录：

```text
RF-DETR/runs/v100_4gpu_seg_large_bboxval/
```

训练命令：

```bash
python RF-DETR/scripts/train.py \
  --model-size seg-large \
  --work-dir RF-DETR/runs/v100_4gpu_seg_large_bboxval \
  --data-dir RF-DETR/data/full_valid_seed42 \
  --raw-json dataset/trainval/trainval.json \
  --image-root dataset/trainval \
  --device cuda \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --resolution 384 \
  --max-epochs 120 \
  --num-workers 4 \
  --checkpoint-interval 5 \
  --eval-interval 1 \
  --large-thr 2048 \
  --map-score-thr 0.05 \
  --q4-score-thr 0.5 \
  --log-q4-metrics \
  --no-early-stopping \
  --precision fp32 \
  --val-bbox-only
```

注意：当前 `train.py` 已拆分阈值参数，不再支持旧的 `--score-thr`。训练和评估中：

- `--map-score-thr` 用于 mAP 预测保留阈值，低阈值保留 PR 曲线。
- `--q4-score-thr` 用于 precision/recall/IoU 工作点。

训练精度由 `--precision` 控制，可选 `auto`、`fp32`、`fp16`、`bf16`。代码默认值是
`auto`；当前主实验命令显式使用 `--precision fp32`，避免在 V100 等不适合 bf16 的
GPU 上触发低效或不稳定的自动精度选择。若更换到支持混合精度更好的 GPU，可按显存和速度需求改为
`fp16` 或 `bf16` 后重新验证指标。

当前主实验配置摘要：

- 模型：`RFDETRSegLarge`
- 预训练权重：`/root/.roboflow/models/rf-detr-seg-large.pt`
- 输入分辨率：`384`
- 训练设备：`cuda`，实际训练配置记录为 4 GPU DDP
- 训练集：`RF-DETR/data/full_valid_seed42/train`
- 内置验证：bbox-only，跳过 segm mask 插值
- 最佳权重：
  - `RF-DETR/runs/v100_4gpu_seg_large_bboxval/checkpoint_best_ema.pth`
  - `RF-DETR/runs/v100_4gpu_seg_large_bboxval/checkpoint_best_regular.pth`
  - `RF-DETR/runs/v100_4gpu_seg_large_bboxval/checkpoint_best_total.pth`

## 评估

完整 valid 评估命令：

```bash
CUDA_VISIBLE_DEVICES=0 python RF-DETR/scripts/eval.py \
  --model-size seg-large \
  --checkpoint RF-DETR/runs/v100_4gpu_seg_large_bboxval/checkpoint_best_ema.pth \
  --val-json RF-DETR/data/full_valid_seed42/valid/_annotations.coco.json \
  --image-root RF-DETR/data/full_valid_seed42/valid \
  --map-score-thr 0.05 \
  --q4-score-thr 0.5 \
  --large-thr 2048 \
  --official-iou-types bbox \
  --bbox-only \
  --out RF-DETR/runs/v100_4gpu_seg_large_bboxval/full_valid_q4_metrics.json
```

当前保留的主评估文件：

```text
RF-DETR/runs/v100_4gpu_seg_large_bboxval/full_valid_q4_metrics.json
```

其中包含：

- `metrics`：Q4 指标。
- `per_image_times`：每张图的尺寸分组和推理耗时。
- `mAP50_source`：当前为 `pycocotools_bbox`。

当前 `full_valid_q4_metrics.json` 中记录的指标：

```text
mAP50=0.8448309903759651
precision50=0.0067821266390139376
small_recall50=1.0
large_mean_bbox_iou=0.9376695611897636
large_recall50=1.0
small_mean_bbox_iou=0.7852259278297424
normal_avg_time_ms=55.86394783757959
large_avg_time_ms=590.0137849152088
```

## 可视化

验证集可视化，带 GT 对比：

```bash
CUDA_VISIBLE_DEVICES=0 python RF-DETR/scripts/visualize.py \
  --checkpoint RF-DETR/runs/v100_4gpu_seg_large_bboxval/checkpoint_best_ema.pth \
  --coco-json RF-DETR/data/full_valid_seed42/valid/_annotations.coco.json \
  --image-root RF-DETR/data/full_valid_seed42/valid \
  --out-dir RF-DETR/runs/v100_4gpu_seg_large_bboxval/vis_50 \
  --score-thr 0.5
```

测试集可视化，无 GT：

```bash
CUDA_VISIBLE_DEVICES=0 python RF-DETR/scripts/visualize.py \
  --checkpoint RF-DETR/runs/v100_4gpu_seg_large_bboxval/checkpoint_best_ema.pth \
  --coco-json <测试集>/_annotations.coco.json \
  --image-root <测试集图像目录> \
  --out-dir RF-DETR/runs/vis_test \
  --score-thr 0.3 \
  --no-draw-gt
```

当前仓库的官方测试集是 `dataset/test/test.json` 原始格式，不是 COCO
`_annotations.coco.json`；如需使用 `visualize.py` 可视化 test，需要先准备 COCO
格式测试集。

## 测试集推理与提交

最终推理使用官方原始 `test.json`：

```bash
CUDA_VISIBLE_DEVICES=0 python RF-DETR/scripts/infer_test.py \
  --test-json dataset/test/test.json \
  --image-root dataset/test \
  --checkpoint RF-DETR/runs/v100_4gpu_seg_large_bboxval/checkpoint_best_ema.pth \
  --model-size seg-large \
  --out RF-DETR/runs/v100_4gpu_seg_large_bboxval/results.json \
  --device cuda:0 \
  --score-thr 0.1 \
  --tile-size 1024 \
  --stride 896 \
  --large-thr 2048 \
  --fp16 \
  --optimize
```

校验提交格式：

```bash
python RF-DETR/tools/check_results.py \
  --test-json dataset/test/test.json \
  --results RF-DETR/runs/v100_4gpu_seg_large_bboxval/results.json \
  --image-root dataset/test
```

## 目录约定

```text
RF-DETR/
├── data/
│   └── full_valid_seed42/
├── runs/
│   ├── v100_4gpu_seg_large_bboxval/
│   └── v100_4gpu_seg_large_fullvalid/
├── scripts/
├── src/
└── tools/
```
