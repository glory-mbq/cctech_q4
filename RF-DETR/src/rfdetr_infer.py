from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.io_utils import ensure_3_channel, read_image_gray_or_color
from src.postprocess import clip_boxes, merge_tile_predictions
from src.rle_utils import mask_to_bbox, mask_to_rle
from src.tiling import crop_tile, get_tiles, map_boxes_back


MODEL_CLASSES = {
    "det-nano": "RFDETRNano",
    "det-small": "RFDETRSmall",
    "det-medium": "RFDETRMedium",
    "det-large": "RFDETRLarge",
    "seg-nano": "RFDETRSegNano",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
    "seg-large": "RFDETRSegLarge",
    "seg-xlarge": "RFDETRSegXLarge",
    "seg-2xlarge": "RFDETRSeg2XLarge",
}

DEFAULT_RESOLUTION = {
    "det-nano": 384,
    "det-small": 512,
    "det-medium": 576,
    "det-large": 704,
    "seg-nano": 312,
    "seg-small": 384,
    "seg-medium": 432,
    "seg-large": 504,
    "seg-xlarge": 624,
    "seg-2xlarge": 768,
}


LOGGER = logging.getLogger(__name__)


def _load_rfdetr_model(rfdetr_cls: Any, variants: Any, checkpoint: Path, model_size: str, device: str) -> Any:
    try:
        return rfdetr_cls.from_checkpoint(str(checkpoint), device=device)
    except KeyError as exc:
        if exc.args != ("args",):
            raise
        LOGGER.warning(
            "checkpoint 缺少 RF-DETR 推理元信息 args，按 model_size=%s 加载 Lightning checkpoint: %s",
            model_size,
            checkpoint,
        )
        klass = getattr(variants, MODEL_CLASSES[model_size])
        return klass(pretrain_weights=str(checkpoint), device=device)


def _patch_postprocess_bbox_only() -> None:
    """推理端禁用 seg PostProcess 的 mask 分支，使大图可整图推理而不触发 mask 插值 OOM。

    与训练侧 scripts/train.py 方案 D 同源。裂纹是细长目标，tiling 会把一条裂纹切成碎片框、
    无法与 GT 完整框匹配（实测大图 mAP_50 仅 0.198）；改为整图推理可输出完整框，但 seg 模型
    整图推理会把 mask 插值到原图全分辨率 OOM，故在此剥离 pred_masks 走 bbox-only。
    幂等；推理进程内只推理，patch 类方法不还原。target_sizes 参数名须与 detr.py 的关键字调用一致。
    """
    from rfdetr.models.postprocess import PostProcess

    if getattr(PostProcess.forward, "_bbox_only_patched", False):
        return
    orig_forward = PostProcess.forward

    def _bbox_only_forward(self, outputs, target_sizes):  # noqa: ANN001
        if isinstance(outputs, dict) and "pred_masks" in outputs:
            outputs = {k: v for k, v in outputs.items() if k != "pred_masks"}
        return orig_forward(self, outputs, target_sizes)

    _bbox_only_forward._bbox_only_patched = True  # type: ignore[attr-defined]
    PostProcess.forward = _bbox_only_forward


class RFDETRWrapper:
    def __init__(
        self,
        checkpoint: str | Path,
        model_size: str = "seg-large",
        device: str = "cuda:0",
        score_thr: float = 0.1,
        fp16: bool = False,
        optimize: bool = False,
        bbox_only: bool = False,
    ) -> None:
        try:
            import torch
            import rfdetr.variants as variants
            from rfdetr.detr import RFDETR
        except ImportError as exc:
            raise ImportError(
                "缺少 RF-DETR 运行环境，请安装 `pip install -r RF-DETR/requirements.txt`。"
            ) from exc

        if model_size not in MODEL_CLASSES:
            raise ValueError(f"未知 model_size={model_size!r}，可选 {sorted(MODEL_CLASSES)}")

        self.torch = torch
        self.device = device
        self.score_thr = float(score_thr)
        self.model_size = model_size
        self.resolution = DEFAULT_RESOLUTION[model_size]
        self.bbox_only = bool(bbox_only)
        if self.bbox_only:
            _patch_postprocess_bbox_only()

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"RF-DETR checkpoint 不存在: {checkpoint}")
        self.model = _load_rfdetr_model(RFDETR, variants, checkpoint, model_size, device)
        if optimize:
            dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
            self.model.optimize_for_inference(compile=False, dtype=dtype)

    def infer_path(
        self,
        image_path: str | Path,
        tile_size: int = 1024,
        stride: int = 896,
        use_tiling: bool = False,
        use_global: bool = False,
        nms_iou: float = 0.5,
    ) -> tuple[list[dict[str, Any]], float]:
        img = read_image_gray_or_color(image_path)
        return self.infer_image(
            img,
            tile_size=tile_size,
            stride=stride,
            use_tiling=use_tiling,
            use_global=use_global,
            nms_iou=nms_iou,
        )

    def infer_image(
        self,
        img: np.ndarray,
        tile_size: int = 1024,
        stride: int = 896,
        use_tiling: bool = False,
        use_global: bool = False,
        nms_iou: float = 0.5,
    ) -> tuple[list[dict[str, Any]], float]:
        self._sync_device()
        start = time.perf_counter()
        image = ensure_3_channel(img)
        h, w = image.shape[:2]

        if use_tiling:
            preds = self._infer_tiled(image, tile_size=tile_size, stride=stride)
            if use_global:
                preds.extend(self._infer_resized_global(image, target_size=tile_size))
            preds = merge_tile_predictions(
                preds,
                image_w=w,
                image_h=h,
                score_thr=self.score_thr,
                nms_iou=nms_iou,
                method="nms",
            )
        else:
            preds = self._infer_single(image)
            preds = clip_boxes(preds, w=w, h=h)

        self._sync_device()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return preds, elapsed_ms

    def _sync_device(self) -> None:
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _infer_tiled(self, image: np.ndarray, tile_size: int, stride: int) -> list[dict[str, Any]]:
        h, w = image.shape[:2]
        all_preds: list[dict[str, Any]] = []
        for tile in get_tiles(h, w, tile_size=tile_size, stride=stride):
            tile_img = crop_tile(image, tile)
            tile_preds = self._infer_single(tile_img)
            if not tile_preds:
                continue
            boxes = np.array([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in tile_preds], dtype=np.float32)
            mapped = map_boxes_back(boxes, tile)
            for pred, box in zip(tile_preds, mapped):
                item = dict(pred)
                item.pop("segmentation", None)
                item.update({"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3])})
                all_preds.append(item)
        return all_preds

    def _infer_resized_global(self, image: np.ndarray, target_size: int) -> list[dict[str, Any]]:
        h, w = image.shape[:2]
        scale = float(target_size) / float(max(h, w))
        if scale <= 0:
            return []
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        preds = self._infer_single(resized)
        for pred in preds:
            pred.pop("segmentation", None)
            pred["x1"] = float(pred["x1"]) / scale
            pred["y1"] = float(pred["y1"]) / scale
            pred["x2"] = float(pred["x2"]) / scale
            pred["y2"] = float(pred["y2"]) / scale
        return preds

    def _infer_single(self, image: np.ndarray) -> list[dict[str, Any]]:
        pil = _bgr_to_rgb_pil(image)
        detections = self.model.predict(pil, threshold=self.score_thr, include_source_image=False)
        return self._detections_to_predictions(detections, image.shape[1], image.shape[0])

    def _detections_to_predictions(self, detections: Any, image_w: int, image_h: int) -> list[dict[str, Any]]:
        boxes = np.asarray(getattr(detections, "xyxy", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(getattr(detections, "confidence", np.zeros((len(boxes),))), dtype=np.float32)
        class_ids = getattr(detections, "class_id", None)
        if class_ids is None:
            class_ids_np = np.zeros((len(boxes),), dtype=np.int64)
        else:
            class_ids_np = np.asarray(class_ids, dtype=np.int64)
        masks = getattr(detections, "mask", None)
        masks_np = np.asarray(masks) if masks is not None else None

        preds: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            if score < self.score_thr:
                continue
            if idx < len(class_ids_np) and int(class_ids_np[idx]) not in (0, 1):
                continue
            out_box = box.astype(np.float32)
            mask_rle: dict[str, Any] | None = None
            if masks_np is not None and idx < len(masks_np):
                mask = (masks_np[idx] > 0).astype(np.uint8)
                mask_box = mask_to_bbox(mask)
                if mask_box[2] > 0 and mask_box[3] > 0:
                    out_box = np.array(
                        [mask_box[0], mask_box[1], mask_box[0] + mask_box[2], mask_box[1] + mask_box[3]],
                        dtype=np.float32,
                    )
                    mask_rle = mask_to_rle(mask, for_json=True)
            x1, y1, x2, y2 = [float(v) for v in out_box.tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            item: dict[str, Any] = {
                "x1": max(0.0, min(x1, float(image_w))),
                "y1": max(0.0, min(y1, float(image_h))),
                "x2": max(0.0, min(x2, float(image_w))),
                "y2": max(0.0, min(y2, float(image_h))),
                "score": score,
                "label": "crack",
            }
            if mask_rle is not None:
                item["segmentation"] = mask_rle
            preds.append(item)
        return sorted(preds, key=lambda x: x["score"], reverse=True)


def _bgr_to_rgb_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = ensure_3_channel(image)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
