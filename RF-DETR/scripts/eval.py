from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coco_convert import coco_gt_boxes
from src.eval_utils import (
    build_per_image_times,
    evaluate_coco_official,
    evaluate_q4_metrics,
    filter_preds_by_score,
    predict_coco_images,
)
from src.io_utils import read_json, setup_logging, write_json
from src.metrics import compute_map50
from src.rfdetr_infer import MODEL_CLASSES, RFDETRWrapper

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR checkpoint with Q4 metrics.")
    parser.add_argument("--checkpoint", required=True, help="RF-DETR checkpoint 路径")
    parser.add_argument("--val-json", required=True, help="COCO 格式验证 JSON")
    parser.add_argument("--image-root", required=True, help="验证图像根目录")
    parser.add_argument("--model-size", choices=tuple(MODEL_CLASSES), default="seg-large", help="RF-DETR segmentation model size")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--map-score-thr", type=float, default=0.05, help="算 mAP 用的分数阈值（低阈值保完整 PR 曲线，默认 0.05）")
    parser.add_argument("--q4-score-thr", type=float, default=0.5, help="算 Q4 指标(precision/recall/IoU)用的分数阈值（工作点，默认 0.5）")
    parser.add_argument("--tile-size", type=int, default=1024, help="切片尺寸")
    parser.add_argument("--stride", type=int, default=896, help="切片步长")
    parser.add_argument("--large-thr", type=int, default=2048, help="超过该边长启用切片推理")
    parser.add_argument("--use-global", action="store_true", help="大图切片时额外融合一次全局缩放预测")
    parser.add_argument(
        "--bbox-only",
        action="store_true",
        help="bbox-only 推理：禁用 mask 分支并对大图整图推理（不 tiling 切碎裂纹），避免 seg 大图 mask 插值 OOM",
    )
    parser.add_argument("--fp16", action="store_true", help="推理优化为 fp16")
    parser.add_argument("--optimize", action="store_true", help="调用 RF-DETR optimize_for_inference")
    parser.add_argument("--skip-official", action="store_true", help="跳过 pycocotools 官方 mAP50，改用本地 mAP50 计算")
    parser.add_argument(
        "--official-iou-types",
        default="bbox",
        help="官方 COCOEval 类型，逗号分隔；Q4 指标只需要 bbox",
    )
    parser.add_argument("--out", default="RF-DETR/runs/metrics.json", help="指标输出 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    coco = read_json(args.val_json)
    predictor = RFDETRWrapper(
        checkpoint=args.checkpoint,
        model_size=args.model_size,
        device=args.device,
        score_thr=min(args.map_score_thr, args.q4_score_thr),  # 推理保留两套阈值都需要的最低分框
        fp16=args.fp16,
        optimize=args.optimize,
        bbox_only=args.bbox_only,
    )
    preds, times = predict_coco_images(
        predictor,
        coco,
        args.image_root,
        tile_size=args.tile_size,
        stride=args.stride,
        large_thr=args.large_thr,
        use_global=args.use_global,
        bbox_only=args.bbox_only,
    )
    gts = coco_gt_boxes(coco)
    # mAP 与 Q4 各按自己的阈值过滤：mAP 用低阈值保完整 PR，Q4 用工作点阈值
    map_preds = filter_preds_by_score(preds, args.map_score_thr)
    q4_preds = filter_preds_by_score(preds, args.q4_score_thr)

    official: dict[str, object] = {}
    map50: float | None = None
    map50_source = "local"
    if not args.skip_official:
        LOGGER.info("开始 pycocotools 官方评估（map 阈值 %.3f）", args.map_score_thr)
        iou_types = tuple(item.strip() for item in args.official_iou_types.split(",") if item.strip())
        official = evaluate_coco_official(
            args.val_json,
            map_preds,
            Path(args.out).parent / "official_eval",
            iou_types=iou_types,
        )
        bbox_metrics = official.get("bbox") if isinstance(official, dict) else None
        if isinstance(bbox_metrics, dict) and "mAP_50" in bbox_metrics:
            map50 = float(bbox_metrics["mAP_50"])
            map50_source = "pycocotools_bbox"
    if map50 is None:
        # 跳过官方评估时用本地 compute_map50，仍基于 map 阈值的预测，保持口径一致
        map50 = compute_map50(map_preds, gts)
    LOGGER.info("开始 Q4 指标评估（q4 阈值 %.3f）", args.q4_score_thr)
    output: dict[str, object] = {
        "metrics": evaluate_q4_metrics(
            q4_preds,
            gts,
            times,
            coco,
            large_thr=args.large_thr,
            map50=map50,
        ),
        "per_image_times": build_per_image_times(coco, times, large_thr=args.large_thr),
        "checkpoint": str(args.checkpoint),
        "map_score_thr": args.map_score_thr,
        "q4_score_thr": args.q4_score_thr,
        "large_thr": args.large_thr,
        "mAP50_source": map50_source,
    }
    write_json(output, args.out)
    LOGGER.info("评估完成: %s", args.out)


if __name__ == "__main__":
    main()
