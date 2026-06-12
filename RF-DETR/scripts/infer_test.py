from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import dataset_items, read_image_gray_or_color, read_json, resolve_image_path, sample_id, sample_image, setup_logging, write_json
from src.rfdetr_infer import MODEL_CLASSES, RFDETRWrapper

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on test.json and write results.json.")
    parser.add_argument("--test-json", required=True, help="原始 test.json 路径")
    parser.add_argument("--image-root", required=True, help="测试图像根目录")
    parser.add_argument("--checkpoint", required=True, help="RF-DETR checkpoint 路径")
    parser.add_argument("--model-size", choices=tuple(MODEL_CLASSES), default="seg-large", help="RF-DETR segmentation model size")
    parser.add_argument("--out", required=True, help="输出 results.json")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--score-thr", type=float, default=0.1, help="预测分数阈值")
    parser.add_argument("--tile-size", type=int, default=1024, help="切片尺寸")
    parser.add_argument("--stride", type=int, default=896, help="切片步长")
    parser.add_argument("--large-thr", type=int, default=2048, help="超过该边长启用切片推理")
    parser.add_argument("--fp16", action="store_true", help="推理优化为 fp16")
    parser.add_argument("--optimize", action="store_true", help="调用 RF-DETR optimize_for_inference")
    parser.add_argument("--use-global", action="store_true", help="大图切片时额外融合一次全局缩放预测")
    return parser.parse_args()


def _format_box(pred: dict[str, float | str]) -> dict[str, float | str]:
    return {
        "x1": round(float(pred["x1"]), 4),
        "y1": round(float(pred["y1"]), 4),
        "x2": round(float(pred["x2"]), 4),
        "y2": round(float(pred["y2"]), 4),
        "score": round(float(pred["score"]), 6),
        "label": "crack",
    }


def main() -> None:
    args = parse_args()
    setup_logging()
    data = read_json(args.test_json)
    samples = dataset_items(data)
    predictor = RFDETRWrapper(
        checkpoint=args.checkpoint,
        model_size=args.model_size,
        device=args.device,
        score_thr=args.score_thr,
        fp16=args.fp16,
        optimize=args.optimize,
    )

    results: list[dict[str, object]] = []
    for sample in tqdm(samples, desc="infer test"):
        image_name = sample_image(sample)
        image_path = resolve_image_path(args.image_root, image_name)
        img = read_image_gray_or_color(image_path)
        h, w = img.shape[:2]
        use_tiling = max(h, w) > args.large_thr
        preds, elapsed_ms = predictor.infer_image(
            img,
            tile_size=args.tile_size,
            stride=args.stride,
            use_tiling=use_tiling,
            use_global=args.use_global and use_tiling,
        )
        results.append(
            {
                "ID": int(sample["ID"]) if str(sample["ID"]).isdigit() else sample_id(sample),
                "image path": image_name,
                "inference_time_ms": round(float(elapsed_ms), 4),
                "groundtruth_bboxes": [],
                "predict_bboxes": [_format_box(pred) for pred in preds],
            }
        )

    write_json(results, args.out)
    LOGGER.info("推理完成: images=%d, out=%s", len(results), args.out)


if __name__ == "__main__":
    main()
