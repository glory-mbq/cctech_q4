from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_prepare import write_rfdetr_dataset
from src.io_utils import dataset_items, image_hw, read_json, resolve_image_path, sample_id, sample_image, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a training dataset that excludes the full validation split from train, "
            "while keeping only small validation images for RF-DETR internal validation."
        )
    )
    parser.add_argument("--raw-json", required=True, help="原始 trainval.json")
    parser.add_argument("--image-root", required=True, help="原始图像根目录")
    parser.add_argument("--full-valid-json", required=True, help="完整 valid 的 COCO _annotations.coco.json")
    parser.add_argument("--out-dir", required=True, help="输出 RF-DETR 数据目录")
    parser.add_argument("--max-valid-side", type=int, default=2048, help="内置 valid 最大长边")
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "symlink", "copy"),
        default="auto",
        help="图像放置方式，auto=硬链接优先，失败软链接，再失败复制",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    if args.max_valid_side <= 0:
        raise ValueError("--max-valid-side 必须为正整数")

    raw_data = read_json(args.raw_json)
    full_valid = read_json(args.full_valid_json)
    full_valid_ids = {str(image["id"]) for image in full_valid.get("images", [])}
    if not full_valid_ids:
        raise ValueError(f"完整 valid JSON 没有 images: {args.full_valid_json}")

    samples = dataset_items(raw_data)
    sample_ids = {sample_id(sample) for sample in samples}
    missing = sorted(full_valid_ids - sample_ids)
    if missing:
        raise ValueError(f"完整 valid 中存在 raw-json 没有的 ID，前 10 个: {missing[:10]}")

    train_ids: list[str] = []
    internal_valid_ids: list[str] = []
    heldout_large_valid_ids: list[str] = []

    for sample in samples:
        sid = sample_id(sample)
        if sid not in full_valid_ids:
            train_ids.append(sid)
            continue
        hw = image_hw(resolve_image_path(args.image_root, sample_image(sample)))
        if hw is not None and max(hw) > args.max_valid_side:
            heldout_large_valid_ids.append(sid)
        else:
            internal_valid_ids.append(sid)

    if not internal_valid_ids:
        raise ValueError("内部 valid 为空；请调大 --max-valid-side 或检查完整 valid 划分")

    manifest = write_rfdetr_dataset(
        raw_data,
        args.image_root,
        args.out_dir,
        {"train": train_ids, "valid": internal_valid_ids},
        link_mode=args.link_mode,
    )
    summary = {
        "train": len(train_ids),
        "internal_valid": len(internal_valid_ids),
        "full_valid": len(full_valid_ids),
        "heldout_large_valid": len(heldout_large_valid_ids),
        "max_valid_side": args.max_valid_side,
        "heldout_large_valid_ids": heldout_large_valid_ids,
        "manifest": manifest,
    }
    write_json(summary, Path(args.out_dir) / "evalclean_summary.json")
    LOGGER.info(
        "eval-clean 数据准备完成: train=%d internal_valid=%d full_valid=%d heldout_large_valid=%d out=%s",
        len(train_ids),
        len(internal_valid_ids),
        len(full_valid_ids),
        len(heldout_large_valid_ids),
        args.out_dir,
    )


if __name__ == "__main__":
    main()
