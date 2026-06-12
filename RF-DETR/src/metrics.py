from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.postprocess import box_iou

DEFAULT_IOU_THR = 0.5
SMALL_MAX_WIDTH = 5.0
SMALL_MAX_AREA = 50.0
LARGE_MIN_AREA = 300.0 * 300.0


def compute_iou(box_a: list[float] | np.ndarray, box_b: list[float] | np.ndarray) -> float:
    a = np.asarray(box_a, dtype=np.float32).reshape(1, 4)
    b = np.asarray(box_b, dtype=np.float32).reshape(1, 4)
    return float(box_iou(a, b)[0, 0])


def match_predictions(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    iou_thr: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    pred_boxes = np.array([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in preds], dtype=np.float32)
    gt_boxes = np.array([[g["x1"], g["y1"], g["x2"], g["y2"]] for g in gts], dtype=np.float32)
    ious = box_iou(pred_boxes, gt_boxes)
    order = sorted(range(len(preds)), key=lambda i: float(preds[i].get("score", 1.0)), reverse=True)
    used_gts: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    unmatched_preds: list[int] = []

    for pred_idx in order:
        gt_idx = int(ious[pred_idx].argmax())
        iou = float(ious[pred_idx, gt_idx])
        if iou >= iou_thr and gt_idx not in used_gts:
            used_gts.add(gt_idx)
            matches.append((pred_idx, gt_idx, iou))
        else:
            unmatched_preds.append(pred_idx)
    unmatched_gts = [i for i in range(len(gts)) if i not in used_gts]
    return matches, unmatched_preds, unmatched_gts


def compute_map50(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
) -> float:
    records: list[tuple[float, int, str, int]] = []
    total_gts = sum(len(v) for v in gts_by_image.values())
    if total_gts == 0:
        return 0.0
    for image_id, preds in preds_by_image.items():
        for idx, pred in enumerate(preds):
            records.append((float(pred.get("score", 0.0)), idx, image_id, 0))
    records.sort(key=lambda x: x[0], reverse=True)

    matched: dict[str, set[int]] = {image_id: set() for image_id in gts_by_image}
    tp: list[float] = []
    fp: list[float] = []
    for _, pred_idx, image_id, _ in records:
        pred = preds_by_image[image_id][pred_idx]
        gts = gts_by_image.get(image_id, [])
        if not gts:
            tp.append(0.0)
            fp.append(1.0)
            continue
        pred_box = np.array([[pred["x1"], pred["y1"], pred["x2"], pred["y2"]]], dtype=np.float32)
        gt_boxes = np.array([[g["x1"], g["y1"], g["x2"], g["y2"]] for g in gts], dtype=np.float32)
        ious = box_iou(pred_box, gt_boxes)[0]
        best_idx = int(ious.argmax())
        if float(ious[best_idx]) >= 0.5 and best_idx not in matched.setdefault(image_id, set()):
            matched[image_id].add(best_idx)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not tp:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gts, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

    ap = 0.0
    for recall_thr in np.linspace(0, 1, 101):
        valid = precisions[recalls >= recall_thr]
        ap += float(valid.max()) if valid.size else 0.0
    return ap / 101.0


def compute_precision(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    iou_thr: float = DEFAULT_IOU_THR,
) -> float:
    total_tp = 0
    total_fp = 0
    image_ids = set(preds_by_image) | set(gts_by_image)
    for image_id in image_ids:
        preds = preds_by_image.get(image_id, [])
        gts = gts_by_image.get(image_id, [])
        matches, unmatched_preds, _ = match_predictions(preds, gts, iou_thr=iou_thr)
        total_tp += len(matches)
        total_fp += len(unmatched_preds)
    denom = total_tp + total_fp
    return float(total_tp / denom) if denom else 0.0


def compute_recall(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    iou_thr: float = DEFAULT_IOU_THR,
) -> float:
    total = 0
    hit = 0
    for image_id, gts in gts_by_image.items():
        preds = preds_by_image.get(image_id, [])
        matches, _, _ = match_predictions(preds, gts, iou_thr=iou_thr)
        total += len(gts)
        hit += len(matches)
    return float(hit / total) if total else 0.0


def _box_area_xyxy(box: dict[str, Any]) -> float:
    return max(0.0, float(box["x2"]) - float(box["x1"])) * max(0.0, float(box["y2"]) - float(box["y1"]))


def _is_small_crack(box: dict[str, Any]) -> bool:
    width = float(box["x2"]) - float(box["x1"])
    return width <= SMALL_MAX_WIDTH or _box_area_xyxy(box) <= SMALL_MAX_AREA


def _is_large_crack(box: dict[str, Any]) -> bool:
    return _box_area_xyxy(box) >= LARGE_MIN_AREA


def _filter_gts(
    gts_by_image: dict[str, list[dict[str, Any]]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, list[dict[str, Any]]]:
    return {image_id: [gt for gt in gts if predicate(gt)] for image_id, gts in gts_by_image.items()}


def compute_small_recall(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    iou_thr: float = DEFAULT_IOU_THR,
) -> float:
    return compute_recall(preds_by_image, _filter_gts(gts_by_image, _is_small_crack), iou_thr=iou_thr)


def compute_large_recall(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    iou_thr: float = DEFAULT_IOU_THR,
) -> float:
    return compute_recall(preds_by_image, _filter_gts(gts_by_image, _is_large_crack), iou_thr=iou_thr)


def compute_mean_bbox_iou(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    predicate: Callable[[dict[str, Any]], bool],
) -> float:
    values: list[float] = []
    for image_id, gts in gts_by_image.items():
        selected_gts = [gt for gt in gts if predicate(gt)]
        preds = preds_by_image.get(image_id, [])
        if not selected_gts:
            continue
        if not preds:
            values.extend(0.0 for _ in selected_gts)
            continue
        pred_boxes = np.array([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in preds], dtype=np.float32)
        gt_boxes = np.array([[g["x1"], g["y1"], g["x2"], g["y2"]] for g in selected_gts], dtype=np.float32)
        ious = box_iou(gt_boxes, pred_boxes)
        values.extend(float(v) for v in ious.max(axis=1))
    return float(np.mean(values)) if values else 0.0


def compute_large_mean_bbox_iou(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
) -> float:
    return compute_mean_bbox_iou(preds_by_image, gts_by_image, _is_large_crack)


def compute_small_mean_bbox_iou(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
) -> float:
    return compute_mean_bbox_iou(preds_by_image, gts_by_image, _is_small_crack)
