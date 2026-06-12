from __future__ import annotations

from typing import Any

import numpy as np


Prediction = dict[str, Any]


def _to_array(preds: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    if not preds:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes = np.array([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in preds], dtype=np.float32)
    scores = np.array([p["score"] for p in preds], dtype=np.float32)
    return boxes, scores


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0, None)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def nms(preds: list[Prediction], iou_thr: float = 0.5) -> list[Prediction]:
    if not preds:
        return []
    boxes, scores = _to_array(preds)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou(boxes[current : current + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return [preds[i] for i in keep]


def soft_nms(
    preds: list[Prediction],
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    score_thr: float = 0.001,
) -> list[Prediction]:
    if not preds:
        return []
    work = [dict(p) for p in preds]
    boxes, _ = _to_array(work)
    scores = np.array([p["score"] for p in work], dtype=np.float32)
    keep: list[Prediction] = []

    while len(work) > 0:
        idx = int(scores.argmax())
        best = work.pop(idx)
        best_box = boxes[idx : idx + 1]
        keep.append(best)
        boxes = np.delete(boxes, idx, axis=0)
        scores = np.delete(scores, idx, axis=0)
        if not work:
            break
        ious = box_iou(best_box, boxes)[0]
        decay = np.where(ious > iou_thr, np.exp(-((ious * ious) / sigma)), 1.0)
        scores = scores * decay.astype(np.float32)
        next_work: list[Prediction] = []
        next_boxes: list[np.ndarray] = []
        next_scores: list[float] = []
        for pred, box, score in zip(work, boxes, scores):
            if float(score) >= score_thr:
                pred = dict(pred)
                pred["score"] = float(score)
                next_work.append(pred)
                next_boxes.append(box)
                next_scores.append(float(score))
        work = next_work
        boxes = np.asarray(next_boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(next_scores, dtype=np.float32)
    return sorted(keep, key=lambda x: x["score"], reverse=True)


def clip_boxes(preds: list[Prediction], w: int, h: int, min_size: float = 1.0) -> list[Prediction]:
    clipped: list[Prediction] = []
    for pred in preds:
        x1 = float(np.clip(pred["x1"], 0, w))
        y1 = float(np.clip(pred["y1"], 0, h))
        x2 = float(np.clip(pred["x2"], 0, w))
        y2 = float(np.clip(pred["y2"], 0, h))
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        item = dict(pred)
        item.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        clipped.append(item)
    return clipped


def merge_tile_predictions(
    preds: list[Prediction],
    image_w: int,
    image_h: int,
    score_thr: float = 0.1,
    nms_iou: float = 0.5,
    method: str = "nms",
) -> list[Prediction]:
    filtered = [dict(p) for p in preds if float(p.get("score", 0.0)) >= score_thr]
    filtered = clip_boxes(filtered, image_w, image_h)
    if method == "soft_nms":
        merged = soft_nms(filtered, iou_thr=nms_iou, score_thr=score_thr)
    elif method == "nms":
        merged = nms(filtered, iou_thr=nms_iou)
    else:
        raise ValueError(f"未知后处理方法: {method}")
    return sorted(merged, key=lambda x: x["score"], reverse=True)
