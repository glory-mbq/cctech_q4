from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)


def add_file_logger(path: str | Path, level: int = logging.INFO) -> logging.Handler:
    """将根 logger 输出额外写入文件，返回 handler 以便训练结束/中断后移除。"""
    path = Path(path)
    ensure_parent(path)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler


def remove_file_logger(handler: logging.Handler | None) -> None:
    """移除并关闭 add_file_logger 返回的 handler，避免多折训练时日志串写。"""
    if handler is None:
        return
    logging.getLogger().removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON 文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {path}, {exc}") from exc


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_lines(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文本文件不存在: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(lines: list[str | int], path: str | Path) -> None:
    path = Path(path)
    ensure_parent(path)
    path.write_text("\n".join(str(x) for x in lines) + "\n", encoding="utf-8")


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get("Dataset")
    if not isinstance(items, list):
        raise ValueError("输入 JSON 缺少列表字段 Dataset")
    return items


def sample_id(sample: dict[str, Any]) -> str:
    if "ID" not in sample:
        raise ValueError(f"样本缺少 ID 字段: {sample}")
    return str(sample["ID"])


def sample_image(sample: dict[str, Any]) -> str:
    image = sample.get("Image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"样本 {sample.get('ID')} 缺少有效 Image 字段")
    return image


def resolve_image_path(image_root: str | Path, image_name: str) -> Path:
    image_root = Path(image_root)
    image_path = Path(image_name)
    if image_path.is_absolute():
        return image_path
    direct = image_root / image_path
    if direct.exists():
        return direct
    if len(image_path.parts) > 1 and image_path.parts[0].lower() == "images":
        stripped = image_root / Path(*image_path.parts[1:])
        if stripped.exists():
            return stripped
    return direct


def read_image_gray_or_color(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    return img


def read_image_bgr(path: str | Path) -> np.ndarray:
    img = read_image_gray_or_color(path)
    return ensure_3_channel(img)


def ensure_3_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"不支持的图像维度: shape={img.shape}")


def image_hw(path: str | Path) -> tuple[int, int] | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return int(img.shape[0]), int(img.shape[1])


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
