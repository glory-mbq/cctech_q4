from __future__ import annotations

"""可视化 RF-DETR 在 val/test 上的裂纹检测结果。

- 推理口径与 scripts/eval.py 完全一致：bbox-only + 大图整图（不 tiling 切碎裂纹）。
- 绿框=模型预测（标注置信度 score），红框=GT（可选，test 无标注时自动跳过）。
- `--score-thr` 控制可视化显示阈值：缺陷检测漏检代价高，默认 0.3（偏召回）；要画面更干净用 0.5。
  注意这是“显示阈值”，不改变模型本身；评估 mAP 仍应另用低阈值（见 eval.py）。
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coco_convert import coco_gt_boxes
from src.io_utils import read_image_bgr, read_json, resolve_image_path, setup_logging
from src.rfdetr_infer import MODEL_CLASSES, RFDETRWrapper

LOGGER = logging.getLogger(__name__)

# 颜色 (BGR)
COLOR_PRED = (0, 200, 0)   # 绿：预测框
COLOR_GT = (0, 0, 255)     # 红：GT 框


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="可视化 RF-DETR 裂纹检测结果（预测框 vs GT 框）。")
    p.add_argument("--checkpoint", required=True, help="RF-DETR checkpoint（建议 checkpoint_best_ema.pth）")
    p.add_argument("--model-size", choices=tuple(MODEL_CLASSES), default="seg-large")
    p.add_argument("--coco-json", required=True, help="COCO 标注 JSON（val/test 的 _annotations.coco.json）")
    p.add_argument("--image-root", required=True, help="图像根目录")
    p.add_argument("--out-dir", default="RF-DETR/runs/vis", help="可视化输出目录")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="可视化显示阈值；缺陷检测偏召回建议 0.25~0.3，要更干净用 0.5（仅影响显示，不改模型）",
    )
    p.add_argument("--large-thr", type=int, default=2048, help="超过该边长的图走整图 bbox-only（与 eval 一致）")
    p.add_argument(
        "--bbox-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bbox-only 推理：大图整图不切碎裂纹（与评估口径一致，强烈建议保持开启）",
    )
    p.add_argument("--draw-gt", action=argparse.BooleanOptionalAction, default=True, help="叠加 GT 框（红色）对比")
    p.add_argument("--max-images", type=int, default=None, help="最多可视化多少张（默认全部）")
    p.add_argument("--max-side", type=int, default=2048, help="保存图最长边上限（仅缩小、省磁盘；0=原尺寸）")
    return p.parse_args()


def _adaptive_style(h: int, w: int) -> tuple[int, float]:
    """按图尺寸自适应线宽与字体，保证 59×46 到 7460×9263 都清晰可读。"""
    s = max(h, w)
    thickness = max(2, int(round(s / 600)))
    font_scale = max(0.5, s / 2000.0)
    return thickness, font_scale


def _draw_box(img, box, color, thickness, label=None, font_scale=0.6) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        ft = max(1, thickness // 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, ft)
        y_top = max(th + 4, y1)
        cv2.rectangle(img, (x1, y_top - th - 4), (x1 + tw, y_top), color, -1)
        cv2.putText(img, label, (x1, y_top - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), ft, cv2.LINE_AA)


def visualize_one(predictor: RFDETRWrapper, image_path: Path, gts: list, args: argparse.Namespace):
    """对单张图推理并叠加预测/GT 框，返回 (可视化图, 显示的预测框数, GT 框数)。"""
    img = read_image_bgr(image_path)  # 统一 3 通道 BGR（灰度图自动转）
    h, w = img.shape[:2]
    use_tiling = (max(h, w) > args.large_thr) and not args.bbox_only
    preds, _ = predictor.infer_image(img, tile_size=1024, stride=896, use_tiling=use_tiling, use_global=False)
    thickness, font_scale = _adaptive_style(h, w)

    # 先画 GT（红，底层），再画预测（绿，上层）便于对比
    if args.draw_gt:
        for g in gts:
            _draw_box(img, (g["x1"], g["y1"], g["x2"], g["y2"]), COLOR_GT, thickness)

    kept = [p for p in preds if float(p.get("score", 0.0)) >= args.score_thr]
    for p in kept:
        _draw_box(img, (p["x1"], p["y1"], p["x2"], p["y2"]), COLOR_PRED, thickness, label=f"{p['score']:.2f}", font_scale=font_scale)

    if args.max_side and max(h, w) > args.max_side:
        scale = args.max_side / max(h, w)
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img, len(kept), len(gts)


def main() -> None:
    args = parse_args()
    setup_logging()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = RFDETRWrapper(
        checkpoint=args.checkpoint,
        model_size=args.model_size,
        device=args.device,
        score_thr=args.score_thr,
        bbox_only=args.bbox_only,
    )
    coco = read_json(args.coco_json)
    gts_by_image = coco_gt_boxes(coco) if args.draw_gt else {}
    images = coco.get("images", [])
    if args.max_images:
        images = images[: args.max_images]

    total_pred = 0
    total_gt = 0
    for i, info in enumerate(images, start=1):
        image_id = str(info["id"])
        image_path = resolve_image_path(args.image_root, info["file_name"])
        gts = gts_by_image.get(image_id, [])
        vis, n_pred, n_gt = visualize_one(predictor, image_path, gts, args)
        total_pred += n_pred
        total_gt += n_gt
        out_path = out_dir / f"{Path(info['file_name']).stem}_vis.jpg"
        cv2.imwrite(str(out_path), vis)
        if i % 20 == 0 or i == len(images):
            LOGGER.info("已处理 %d/%d", i, len(images))

    LOGGER.info("完成：%d 张可视化已保存到 %s", len(images), out_dir)
    LOGGER.info(
        "score>=%.3f 下：预测框 %d 个（平均 %.1f 框/图），GT 框 %d 个",
        args.score_thr,
        total_pred,
        total_pred / max(len(images), 1),
        total_gt,
    )
    LOGGER.info("图例：绿框=预测(数字为置信度)  红框=GT")


if __name__ == "__main__":
    main()
