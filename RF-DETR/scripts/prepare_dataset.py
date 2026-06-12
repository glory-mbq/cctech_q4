from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_prepare import split_ids, write_rfdetr_dataset
from src.io_utils import dataset_items, image_hw, read_json, resolve_image_path, sample_image, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RF-DETR COCO-style dataset from CCTech Q4 trainval.json.")
    parser.add_argument("--raw-json", required=True, help="原始 trainval.json")
    parser.add_argument("--image-root", required=True, help="原始图像根目录")
    parser.add_argument("--out-dir", default="RF-DETR/data/plain", help="RF-DETR 数据输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="可选独立 test split 比例")
    parser.add_argument("--max-valid-side", type=int, default=None, help="验证集最大长边；超出则移回 train，避免内置验证显存爆")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "symlink", "copy"),
        default="auto",
        help="图像放置方式，auto=硬链接优先，失败软链接，再失败复制",
    )
    return parser.parse_args()


def _move_large_valid_to_train(
    samples: list[dict[str, object]],
    image_root: str | Path,
    train_ids: list[str],
    valid_ids: list[str],
    max_side: int,
) -> tuple[list[str], list[str]]:
    if max_side <= 0:
        raise ValueError("--max-valid-side 必须为正整数")

    sample_by_id = {str(sample["ID"]): sample for sample in samples}
    keep_valid: list[str] = []
    moved: list[str] = []
    for sid in valid_ids:
        sample = sample_by_id[sid]
        hw = image_hw(resolve_image_path(image_root, sample_image(sample)))
        if hw is not None and max(hw) > max_side:
            moved.append(sid)
        else:
            keep_valid.append(sid)
    if moved:
        LOGGER.info("验证集排除超大图: moved_to_train=%d max_valid_side=%d", len(moved), max_side)
    return sorted(set(train_ids).union(moved)), keep_valid


def main() -> None:
    args = parse_args()
    setup_logging()
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio 必须在 (0, 1) 内")
    if args.test_ratio < 0 or args.test_ratio >= 1:
        raise ValueError("--test-ratio 必须在 [0, 1) 内")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio 必须小于 1")

    raw_data = read_json(args.raw_json)
    samples = dataset_items(raw_data)
    all_ids = [str(sample["ID"]) for sample in samples]

    if args.test_ratio > 0:
        train_pool_ids, test_ids = split_ids(samples, args.image_root, args.test_ratio, args.seed)
        train_pool_samples = [sample for sample in samples if str(sample["ID"]) in set(train_pool_ids)]
        val_ratio_in_pool = args.val_ratio / (1.0 - args.test_ratio)
        train_ids, valid_ids = split_ids(train_pool_samples, args.image_root, val_ratio_in_pool, args.seed)
    else:
        train_ids, valid_ids = split_ids(samples, args.image_root, args.val_ratio, args.seed)
        test_ids = []

    if args.max_valid_side is not None:
        train_ids, valid_ids = _move_large_valid_to_train(samples, args.image_root, train_ids, valid_ids, args.max_valid_side)

    split_to_ids = {"train": train_ids, "valid": valid_ids}
    if test_ids:
        split_to_ids["test"] = test_ids

    manifest = write_rfdetr_dataset(raw_data, args.image_root, args.out_dir, split_to_ids, link_mode=args.link_mode)
    LOGGER.info(
        "RF-DETR 数据准备完成: total=%d train=%d valid=%d test=%d out=%s",
        len(all_ids),
        len(train_ids),
        len(valid_ids),
        len(test_ids),
        args.out_dir,
    )
    LOGGER.info("manifest=%s", manifest)


if __name__ == "__main__":
    main()
