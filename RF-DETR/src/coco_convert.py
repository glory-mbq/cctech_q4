from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.io_utils import dataset_items, image_hw, resolve_image_path, sample_id, sample_image
from src.rle_utils import normalize_rle, rle_area

LOGGER = logging.getLogger(__name__)


def clip_bbox_xywh(bbox: list[float], width: int, height: int) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None
    x1 = max(0.0, min(x, float(width)))
    y1 = max(0.0, min(y, float(height)))
    x2 = max(0.0, min(x + w, float(width)))
    y2 = max(0.0, min(y + h, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def infer_hw(sample: dict[str, Any], image_root: Path) -> tuple[int, int]:
    path = resolve_image_path(image_root, sample_image(sample))
    hw = image_hw(path)
    if hw is not None:
        return hw

    annotations = sample.get("Annotations") or []
    for ann in annotations:
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, dict) and "size" in segmentation:
            h, w = segmentation["size"]
            return int(h), int(w)
    raise FileNotFoundError(f"无法读取图像尺寸，也无法从 RLE 推断: ID={sample.get('ID')}, path={path}")


def convert_dataset_to_coco(
    data: dict[str, Any],
    image_root: str | Path,
    keep_ids: set[str] | None = None,
) -> dict[str, Any]:
    image_root = Path(image_root)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for img_index, sample in enumerate(dataset_items(data), start=1):
        sid = sample_id(sample)
        if keep_ids is not None and sid not in keep_ids:
            continue

        h, w = infer_hw(sample, image_root)
        image_id = int(sample["ID"]) if str(sample["ID"]).isdigit() else img_index
        images.append(
            {
                "id": image_id,
                "file_name": sample_image(sample),
                "width": int(w),
                "height": int(h),
            }
        )

        for ann in sample.get("Annotations") or []:
            if ann.get("ObjectCategory", "crack") != "crack":
                continue
            bbox = clip_bbox_xywh(ann.get("bbox", []), w, h)
            if bbox is None:
                LOGGER.warning("过滤非法 bbox: image_id=%s, bbox=%s", sample.get("ID"), ann.get("bbox"))
                continue
            segmentation = ann.get("segmentation")
            try:
                rle = normalize_rle(segmentation, for_json=True)
                area = rle_area(segmentation)
            except Exception as exc:
                LOGGER.warning("过滤非法 mask: image_id=%s, error=%s", sample.get("ID"), exc)
                continue
            if area <= 0:
                LOGGER.warning("过滤空 mask: image_id=%s", sample.get("ID"))
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [round(float(v), 4) for v in bbox],
                    "area": float(area),
                    # 原始 iscrowd 无业务含义；COCO 训练中 1 会被视作 ignore。
                    "iscrowd": 0,
                    "segmentation": rle,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "crack", "supercategory": "none"}],
    }


def coco_gt_boxes(coco_data: dict[str, Any]) -> dict[str, list[dict[str, float | str]]]:
    image_id_to_key = {img["id"]: str(img["id"]) for img in coco_data.get("images", [])}
    result: dict[str, list[dict[str, float | str]]] = {key: [] for key in image_id_to_key.values()}
    for ann in coco_data.get("annotations", []):
        key = image_id_to_key.get(ann["image_id"], str(ann["image_id"]))
        x, y, w, h = [float(v) for v in ann["bbox"]]
        result.setdefault(key, []).append(
            {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "label": "crack",
            }
        )
    return result
