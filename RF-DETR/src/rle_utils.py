from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def normalize_rle(rle: dict[str, Any], for_json: bool = False) -> dict[str, Any]:
    if not isinstance(rle, dict):
        raise ValueError("segmentation 必须是 COCO-RLE dict")
    if "size" not in rle or "counts" not in rle:
        raise ValueError(f"RLE 缺少 size/counts 字段: {rle}")

    size = [int(rle["size"][0]), int(rle["size"][1])]
    counts = rle["counts"]
    if isinstance(counts, str):
        normalized_counts: str | bytes | list[int] = counts if for_json else counts.encode("utf-8")
    elif isinstance(counts, bytes):
        normalized_counts = counts.decode("utf-8") if for_json else counts
    elif isinstance(counts, list):
        normalized_counts = [int(x) for x in counts]
    else:
        raise ValueError(f"不支持的 RLE counts 类型: {type(counts).__name__}")
    return {"size": size, "counts": normalized_counts}


def _compressed_rle(rle: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_rle(rle, for_json=False)
    counts = normalized["counts"]
    if isinstance(counts, list):
        h, w = normalized["size"]
        encoded = mask_utils.frPyObjects(normalized, int(h), int(w))
        return normalize_rle(encoded, for_json=False)
    return normalized


def rle_area(rle: dict[str, Any]) -> float:
    compressed = _compressed_rle(rle)
    return float(mask_utils.area(compressed))


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    compressed = _compressed_rle(rle)
    mask = mask_utils.decode(compressed)
    return mask.astype(np.uint8)


def mask_to_rle(mask: np.ndarray, for_json: bool = True) -> dict[str, Any]:
    if mask.ndim != 2:
        raise ValueError(f"mask_to_rle 仅支持二维 mask，当前 shape={mask.shape}")
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return normalize_rle(encoded, for_json=for_json)


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    if mask.ndim != 2:
        raise ValueError(f"mask_to_bbox 仅支持二维 mask，当前 shape={mask.shape}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return [x1, y1, x2 - x1, y2 - y1]
