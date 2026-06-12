from __future__ import annotations

import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.coco_convert import convert_dataset_to_coco
from src.io_utils import dataset_items, ensure_dir, image_hw, resolve_image_path, sample_id, sample_image, write_json

LOGGER = logging.getLogger(__name__)

RFD_SPLITS = ("train", "valid", "test")


def sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value)


def split_ids(
    samples: list[dict[str, Any]],
    image_root: str | Path,
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    ids = [sample_id(sample) for sample in samples]
    strata = [_sample_stratum(sample, image_root) for sample in samples]
    try:
        train_ids, test_ids = train_test_split(
            ids,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=strata,
        )
    except ValueError as exc:
        LOGGER.warning("分层划分失败，退回随机划分: %s", exc)
        train_ids, test_ids = _random_split(ids, test_size, seed)
    return sorted(train_ids, key=sort_key), sorted(test_ids, key=sort_key)


def kfold_ids(
    samples: list[dict[str, Any]],
    image_root: str | Path,
    folds: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    if folds < 2:
        raise ValueError("--cv-folds 必须大于等于 2")
    if folds > len(samples):
        raise ValueError(f"--cv-folds 不能超过样本数: folds={folds}, samples={len(samples)}")

    ids = [sample_id(sample) for sample in samples]
    strata = _choose_strata(samples, image_root, min_count=folds)
    if strata is None:
        LOGGER.warning("没有满足每折最小样本数的分层，退回普通 KFold")
        splits = KFold(n_splits=folds, shuffle=True, random_state=seed).split(ids)
    else:
        splits = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(ids, strata)

    result: list[tuple[list[str], list[str]]] = []
    for train_idx, val_idx in splits:
        train_ids = [ids[idx] for idx in train_idx]
        val_ids = [ids[idx] for idx in val_idx]
        result.append((sorted(train_ids, key=sort_key), sorted(val_ids, key=sort_key)))
    return result


def subset_samples(samples: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample_id(sample) in ids]


def write_rfdetr_dataset(
    raw_data: dict[str, Any],
    image_root: str | Path,
    out_dir: str | Path,
    split_to_ids: dict[str, list[str]],
    link_mode: str = "auto",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    image_root = Path(image_root)
    manifest: dict[str, Any] = {"out_dir": out_dir.as_posix(), "splits": {}}

    for split_name, ids in split_to_ids.items():
        if split_name not in RFD_SPLITS:
            raise ValueError(f"RF-DETR split 只能是 {RFD_SPLITS}: {split_name}")
        split_dir = ensure_dir(out_dir / split_name)
        images_dir = ensure_dir(split_dir / "images")
        coco = convert_dataset_to_coco(raw_data, image_root, keep_ids=set(ids))
        linked = link_coco_images(coco, image_root, images_dir, mode=link_mode)
        write_json(coco, split_dir / "_annotations.coco.json")
        manifest["splits"][split_name] = {
            "num_ids": len(ids),
            "num_images": len(coco.get("images", [])),
            "num_annotations": len(coco.get("annotations", [])),
            "link_counts": dict(Counter(linked.values())),
        }

    write_json(manifest, out_dir / "dataset_manifest.json")
    return manifest


def link_coco_images(coco: dict[str, Any], image_root: str | Path, images_dir: str | Path, mode: str = "auto") -> dict[str, str]:
    if mode not in {"auto", "hardlink", "symlink", "copy"}:
        raise ValueError("--link-mode 必须是 auto/hardlink/symlink/copy")
    image_root = Path(image_root)
    images_dir = ensure_dir(images_dir)
    linked: dict[str, str] = {}

    for image in coco.get("images", []):
        file_name = str(image["file_name"])
        src = resolve_image_path(image_root, file_name)
        dst = images_dir / Path(file_name).name
        if not src.exists():
            raise FileNotFoundError(f"图像不存在: {src}")
        if dst.exists():
            linked[file_name] = "exists"
            image["file_name"] = f"images/{dst.name}"
            continue
        linked[file_name] = _place_file(src, dst, mode)
        image["file_name"] = f"images/{dst.name}"
    return linked


def _place_file(src: Path, dst: Path, mode: str) -> str:
    ensure_dir(dst.parent)
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlink"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def _random_split(ids: list[str], test_size: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * test_size)))
    return shuffled[val_count:], shuffled[:val_count]


def _bbox_area_bin(area: float) -> str:
    if area <= 50:
        return "tiny"
    if area <= 32 * 32:
        return "small"
    if area <= 300 * 300:
        return "medium"
    return "large"


def _image_size_bin(sample: dict[str, Any], image_root: str | Path) -> str:
    path = resolve_image_path(image_root, sample_image(sample))
    hw = image_hw(path)
    if hw is None:
        return "unknown"
    side = max(hw)
    if side <= 1024:
        return "short"
    if side <= 2048:
        return "mid"
    return "long"


def _sample_stratum(sample: dict[str, Any], image_root: str | Path, level: str = "detailed") -> str:
    max_area = 0.0
    has_tiny = False
    has_large = False
    for ann in sample.get("Annotations") or []:
        bbox = ann.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            continue
        width = max(float(bbox[2]), 0.0)
        height = max(float(bbox[3]), 0.0)
        area = width * height
        max_area = max(max_area, area)
        has_tiny = has_tiny or width <= 5 or area <= 50
        has_large = has_large or area >= 300 * 300

    size_bin = _image_size_bin(sample, image_root)
    if level == "detailed":
        return f"area={_bbox_area_bin(max_area)}|tiny={int(has_tiny)}|large={int(has_large)}|size={size_bin}"
    if level == "coarse":
        return f"tiny={int(has_tiny)}|large={int(has_large)}|long={int(size_bin == 'long')}"
    return f"tiny={int(has_tiny)}|large={int(has_large)}"


def _choose_strata(samples: list[dict[str, Any]], image_root: str | Path, min_count: int) -> list[str] | None:
    for level in ("detailed", "coarse", "binary"):
        strata = [_sample_stratum(sample, image_root, level=level) for sample in samples]
        counts = Counter(strata)
        if counts and min(counts.values()) >= min_count:
            return strata
    return None
