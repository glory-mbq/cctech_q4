from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(int(x) for x in starts))


def get_tiles(h: int, w: int, tile_size: int, stride: int) -> list[Tile]:
    if h <= 0 or w <= 0:
        raise ValueError(f"图像尺寸非法: h={h}, w={w}")
    if tile_size <= 0 or stride <= 0:
        raise ValueError(f"tile_size/stride 必须为正数: tile_size={tile_size}, stride={stride}")
    if h <= tile_size and w <= tile_size:
        return [Tile(0, 0, w, h)]

    xs = _axis_starts(w, tile_size, stride)
    ys = _axis_starts(h, tile_size, stride)
    tiles: list[Tile] = []
    for y in ys:
        for x in xs:
            tiles.append(Tile(x, y, min(x + tile_size, w), min(y + tile_size, h)))
    return tiles


def crop_tile(img: np.ndarray, tile: Tile) -> np.ndarray:
    return img[tile.y1 : tile.y2, tile.x1 : tile.x2].copy()


def map_boxes_back(boxes: np.ndarray, tile: Tile) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    mapped = boxes.astype(np.float32).copy()
    mapped[:, [0, 2]] += tile.x1
    mapped[:, [1, 3]] += tile.y1
    return mapped
