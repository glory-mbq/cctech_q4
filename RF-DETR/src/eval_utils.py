from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from src.coco_convert import coco_gt_boxes
from src.io_utils import read_image_gray_or_color, read_json, resolve_image_path, write_json
from src.metrics import (
    DEFAULT_IOU_THR,
    compute_large_mean_bbox_iou,
    compute_large_recall,
    compute_map50,
    compute_precision,
    compute_small_mean_bbox_iou,
    compute_small_recall,
)
from src.rle_utils import normalize_rle


def predict_coco_images(
    predictor: Any,
    coco: dict[str, Any],
    image_root: str | Path,
    tile_size: int = 1024,
    stride: int = 896,
    large_thr: int = 2048,
    use_global: bool = False,
    bbox_only: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    preds_by_image: dict[str, list[dict[str, Any]]] = {}
    times_by_image: dict[str, float] = {}
    for img_info in tqdm(coco.get("images", []), desc="predict coco"):
        image_id = str(img_info["id"])
        image_path = resolve_image_path(image_root, img_info["file_name"])
        img = read_image_gray_or_color(image_path)
        h, w = img.shape[:2]
        use_tiling = (max(h, w) > large_thr) and not bbox_only
        preds, elapsed_ms = predictor.infer_image(
            img,
            tile_size=tile_size,
            stride=stride,
            use_tiling=use_tiling,
            use_global=use_global and use_tiling,
        )
        preds_by_image[image_id] = preds
        times_by_image[image_id] = elapsed_ms
    return preds_by_image, times_by_image


def filter_preds_by_score(
    preds_by_image: dict[str, list[dict[str, Any]]],
    score_thr: float,
) -> dict[str, list[dict[str, Any]]]:
    """按置信度阈值过滤每图预测框（供 mAP / Q4 各自独立阈值使用）。"""
    return {
        image_id: [p for p in preds if float(p.get("score", 0.0)) >= score_thr]
        for image_id, preds in preds_by_image.items()
    }


def evaluate_q4_metrics(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    times_by_image: dict[str, float],
    coco: dict[str, Any],
    large_thr: int = 2048,
    map50: float | None = None,
) -> dict[str, float | int]:
    time_summary = summarize_time_by_size(coco, times_by_image, large_thr=large_thr)
    return {
        "mAP50": float(map50) if map50 is not None else compute_map50(preds_by_image, gts_by_image),
        "precision50": compute_precision(preds_by_image, gts_by_image, iou_thr=DEFAULT_IOU_THR),
        "small_recall50": compute_small_recall(preds_by_image, gts_by_image, iou_thr=DEFAULT_IOU_THR),
        "large_mean_bbox_iou": compute_large_mean_bbox_iou(preds_by_image, gts_by_image),
        "large_recall50": compute_large_recall(preds_by_image, gts_by_image, iou_thr=DEFAULT_IOU_THR),
        "small_mean_bbox_iou": compute_small_mean_bbox_iou(preds_by_image, gts_by_image),
        "normal_avg_time_ms": time_summary["normal_avg_time_ms"],
        "large_avg_time_ms": time_summary["large_avg_time_ms"],
    }


def build_per_image_times(
    coco: dict[str, Any],
    times_by_image: dict[str, float],
    large_thr: int = 2048,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for img_info in coco.get("images", []):
        image_id = str(img_info["id"])
        width = int(img_info.get("width", 0))
        height = int(img_info.get("height", 0))
        size_group = "large" if max(width, height) > large_thr else "normal"
        records.append(
            {
                "image_id": int(image_id) if image_id.isdigit() else image_id,
                "file_name": img_info.get("file_name", ""),
                "width": width,
                "height": height,
                "size_group": size_group,
                "elapsed_ms": float(times_by_image.get(image_id, 0.0)),
            }
        )
    return records


def summarize_time_by_size(
    coco: dict[str, Any],
    times_by_image: dict[str, float],
    large_thr: int = 2048,
) -> dict[str, float]:
    records = build_per_image_times(coco, times_by_image, large_thr=large_thr)
    normal_times = [record["elapsed_ms"] for record in records if record["size_group"] == "normal"]
    large_times = [record["elapsed_ms"] for record in records if record["size_group"] == "large"]
    return {
        "normal_avg_time_ms": mean(normal_times) if normal_times else 0.0,
        "large_avg_time_ms": mean(large_times) if large_times else 0.0,
    }


def evaluate_coco_official(
    coco_gt_path: str | Path,
    preds_by_image: dict[str, list[dict[str, Any]]],
    out_dir: str | Path,
    iou_types: tuple[str, ...] = ("bbox",),
) -> dict[str, Any]:
    from pycocotools.coco import COCO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_gt = COCO(str(coco_gt_path))
    output: dict[str, Any] = {}
    if "bbox" in iou_types:
        bbox_results = _to_bbox_results(preds_by_image)
        bbox_path = out_dir / "bbox_predictions.json"
        write_json(bbox_results, bbox_path)
        output["bbox"] = _run_cocoeval(coco_gt, bbox_path, "bbox") if bbox_results else {}
    if "segm" in iou_types:
        segm_results = _to_segm_results(preds_by_image)
        segm_path = out_dir / "segm_predictions.json"
        write_json(segm_results, segm_path)
        output["segm"] = _run_cocoeval(coco_gt, segm_path, "segm") if segm_results else {}
    return output


def coco_gt_boxes_from_path(coco_path: str | Path) -> dict[str, list[dict[str, float | str]]]:
    return coco_gt_boxes(read_json(coco_path))


def _run_cocoeval(coco_gt: Any, result_path: Path, iou_type: str) -> dict[str, float]:
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(result_path))
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = [
        "mAP",
        "mAP_50",
        "mAP_75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {key: float(value) for key, value in zip(keys, evaluator.stats.tolist())}


def _to_bbox_results(preds_by_image: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for image_id, preds in preds_by_image.items():
        for pred in preds:
            x1 = float(pred["x1"])
            y1 = float(pred["y1"])
            x2 = float(pred["x2"])
            y2 = float(pred["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                {
                    "image_id": int(image_id) if str(image_id).isdigit() else image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(pred.get("score", 0.0)),
                }
            )
    return results


def _to_segm_results(preds_by_image: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for image_id, preds in preds_by_image.items():
        for pred in preds:
            segmentation = pred.get("segmentation")
            if not segmentation:
                continue
            results.append(
                {
                    "image_id": int(image_id) if str(image_id).isdigit() else image_id,
                    "category_id": 1,
                    "segmentation": normalize_rle(segmentation, for_json=True),
                    "score": float(pred.get("score", 0.0)),
                }
            )
    return results
