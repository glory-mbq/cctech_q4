from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import dataset_items, image_hw, read_json, resolve_image_path, sample_image, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate competition results.json format.")
    parser.add_argument("--test-json", required=True, help="原始 test.json 路径")
    parser.add_argument("--results", required=True, help="待校验 results.json")
    parser.add_argument("--image-root", default=None, help="可选图像根目录，用于 bbox 越界检查")
    parser.add_argument("--clip-out", default=None, help="可选：输出 clip 后的 results.json")
    return parser.parse_args()


def _id_value(value: Any) -> str:
    return str(value)


def _validate_box(box: dict[str, Any], image_hw_value: tuple[int, int] | None, clip: bool) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    required = ["x1", "y1", "x2", "y2", "score", "label"]
    for key in required:
        if key not in box:
            errors.append(f"预测框缺少字段 {key}")
    if errors:
        return box, errors

    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
        score = float(box["score"])
    except (TypeError, ValueError):
        errors.append(f"预测框坐标或 score 非数字: {box}")
        return box, errors

    if x2 <= x1 or y2 <= y1:
        errors.append(f"预测框坐标非法: {box}")
    if not 0.0 <= score <= 1.0:
        errors.append(f"score 超出 [0,1]: {box}")
    if box["label"] != "crack":
        errors.append(f"label 必须为 crack: {box}")

    clipped = dict(box)
    if image_hw_value is not None:
        h, w = image_hw_value
        out_of_bounds = x1 < 0 or y1 < 0 or x2 > w or y2 > h
        if out_of_bounds:
            LOGGER.warning("bbox 越界: box=%s, image_hw=%s", box, image_hw_value)
            if clip:
                x1 = max(0.0, min(x1, float(w)))
                y1 = max(0.0, min(y1, float(h)))
                x2 = max(0.0, min(x2, float(w)))
                y2 = max(0.0, min(y2, float(h)))
                clipped.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                if x2 <= x1 or y2 <= y1:
                    errors.append(f"clip 后预测框坐标非法: {box}")
    return clipped, errors


def validate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    test_data = read_json(args.test_json)
    results = read_json(args.results)
    if not isinstance(results, list):
        return [], ["results.json 顶层必须是 list"]

    test_samples = dataset_items(test_data)
    expected_by_id = {_id_value(sample["ID"]): sample for sample in test_samples}
    result_by_id = {_id_value(item.get("ID")): item for item in results if isinstance(item, dict)}
    errors: list[str] = []
    clipped_results: list[dict[str, Any]] = []

    if len(result_by_id) != len([item for item in results if isinstance(item, dict)]):
        errors.append("results.json 存在重复 ID")
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            errors.append(f"results[{idx}] 必须是 dict")

    if set(expected_by_id) != set(result_by_id):
        missing = sorted(set(expected_by_id) - set(result_by_id))
        extra = sorted(set(result_by_id) - set(expected_by_id))
        if missing:
            errors.append(f"缺少 ID: {missing[:20]}")
        if extra:
            errors.append(f"多余 ID: {extra[:20]}")

    for sample in test_samples:
        sid = _id_value(sample["ID"])
        item = result_by_id.get(sid)
        if item is None:
            continue
        expected_image = sample_image(sample)
        item_errors = []

        if item.get("image path") != expected_image:
            item_errors.append(f"ID={sid} image path 不匹配: got={item.get('image path')}, expected={expected_image}")
        if not isinstance(item.get("groundtruth_bboxes"), list):
            item_errors.append(f"ID={sid} groundtruth_bboxes 必须是 list")
        if not isinstance(item.get("predict_bboxes"), list):
            item_errors.append(f"ID={sid} predict_bboxes 必须是 list")
        try:
            elapsed = float(item.get("inference_time_ms"))
            if elapsed <= 0:
                item_errors.append(f"ID={sid} inference_time_ms 必须为正数")
        except (TypeError, ValueError):
            item_errors.append(f"ID={sid} inference_time_ms 必须是数字")

        image_hw_value = None
        if args.image_root:
            image_hw_value = image_hw(resolve_image_path(args.image_root, expected_image))
            if image_hw_value is None:
                item_errors.append(f"ID={sid} 图像无法读取，不能检查越界: {expected_image}")

        new_item = dict(item)
        new_boxes = []
        if isinstance(item.get("predict_bboxes"), list):
            for box in item["predict_bboxes"]:
                if not isinstance(box, dict):
                    item_errors.append(f"ID={sid} 预测框必须是 dict: {box}")
                    continue
                clipped_box, box_errors = _validate_box(box, image_hw_value, clip=bool(args.clip_out))
                item_errors.extend(f"ID={sid} {err}" for err in box_errors)
                new_boxes.append(clipped_box)
        new_item["predict_bboxes"] = new_boxes
        clipped_results.append(new_item)
        errors.extend(item_errors)

    return clipped_results, errors


def main() -> None:
    args = parse_args()
    setup_logging()
    clipped_results, errors = validate(args)
    if args.clip_out:
        write_json(clipped_results, args.clip_out)
        LOGGER.info("已输出 clip 后结果: %s", args.clip_out)
    if errors:
        for err in errors[:100]:
            LOGGER.error(err)
        raise SystemExit(f"校验失败，共 {len(errors)} 个错误")
    LOGGER.info("校验通过: %s", args.results)


if __name__ == "__main__":
    main()
