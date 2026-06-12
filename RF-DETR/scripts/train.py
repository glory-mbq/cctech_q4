from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coco_convert import coco_gt_boxes
from src.dataset_prepare import kfold_ids, split_ids, subset_samples, write_rfdetr_dataset
from src.eval_utils import build_per_image_times, evaluate_q4_metrics, filter_preds_by_score, predict_coco_images
from src.io_utils import (
    add_file_logger,
    dataset_items,
    read_json,
    remove_file_logger,
    setup_logging,
    write_json,
)
from src.metrics import compute_map50
from src.postprocess import box_iou
from src.rfdetr_infer import DEFAULT_RESOLUTION, MODEL_CLASSES, RFDETRWrapper

LOGGER = logging.getLogger(__name__)

try:
    from pytorch_lightning import Callback as _LightningCallback
except Exception:
    _LightningCallback = object

CV_MODES = ("none", "5-fold-cv", "5-fold-cv-standalone-test", "5-time-train+valid+test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR segmentation crack detector.")
    parser.add_argument("--model-size", choices=tuple(MODEL_CLASSES), default="seg-large", help="RF-DETR segmentation model size")
    parser.add_argument("--work-dir", default="RF-DETR/runs/rfdetr_seg_large_plain", help="训练输出目录")
    parser.add_argument("--data-dir", default="RF-DETR/data/plain", help="RF-DETR 数据目录，普通训练使用")
    parser.add_argument("--raw-json", default="dataset/trainval/trainval.json", help="原始 trainval.json")
    parser.add_argument("--image-root", default="dataset/trainval", help="原始训练图像根目录")
    parser.add_argument("--device", default="cuda:0", help="训练/评估设备；多卡训练用 cuda，单卡指定 cuda:0")
    parser.add_argument("--resolution", type=int, default=None, help="覆盖 RF-DETR 输入分辨率")
    parser.add_argument("--batch-size", default="auto", help="每卡 batch size，整数或 auto")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="梯度累计步数")
    parser.add_argument("--max-epochs", type=int, default=120, help="训练 epoch")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="主学习率")
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4, help="encoder 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="checkpoint interval")
    parser.add_argument("--eval-interval", type=int, default=1, help="RF-DETR eval interval")
    parser.add_argument("--num-select", type=int, default=None, help="验证/推理后处理保留的候选数量，降低可减少显存")
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True, help="启用早停")
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="早停 patience")
    parser.add_argument("--resume", default=None, help="RF-DETR resume checkpoint")
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="同一 work-dir 下若存在 last.ckpt 则自动续训（CV 各折独立续训）；--no-auto-resume 强制从头",
    )
    parser.add_argument(
        "--progress-bar",
        choices=("auto", "tqdm", "rich", "none"),
        default="auto",
        help="训练进度条；auto 在非交互终端（重定向/后台）自动关闭，避免日志被进度条刷屏",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
        help="训练精度；V100 等无 bf16 硬件单元的卡建议 fp32 或 fp16（auto 下 rfdetr 会在 V100 误用 bf16 软件模拟）",
    )
    parser.add_argument(
        "--val-bbox-only",
        action="store_true",
        help="验证/测试只评 bbox，跳过 segm mask 插值，避免大图 mask 上采样到原图分辨率导致 OOM（与 eval.py 的 bbox 评测口径一致）",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "symlink", "copy"),
        default="auto",
        help="图像放置方式，auto=硬链接优先，失败软链接，再失败复制",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="普通训练验证集比例")

    parser.add_argument("--cv-mode", choices=CV_MODES, default="none", help="交叉验证/重复实验模式")
    parser.add_argument("--cv-seed", type=int, default=42, help="交叉验证随机种子")
    parser.add_argument("--cv-folds", type=int, default=5, help="K 折数量")
    parser.add_argument("--repeat-times", type=int, default=5, help="重复实验次数")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="独立测试集比例")
    parser.add_argument("--repeat-val-ratio", type=float, default=0.1, help="重复实验验证集比例")

    parser.add_argument("--map-score-thr", type=float, default=0.05, help="算 mAP 用的分数阈值（低阈值保完整 PR 曲线，默认 0.05）")
    parser.add_argument("--q4-score-thr", type=float, default=0.5, help="算 Q4 指标(precision/recall/IoU)用的分数阈值（工作点，默认 0.5）")
    parser.add_argument("--tile-size", type=int, default=1024, help="评估切片尺寸")
    parser.add_argument("--stride", type=int, default=896, help="评估切片步长")
    parser.add_argument("--large-thr", type=int, default=2048, help="超过该边长启用切片推理")
    parser.add_argument(
        "--log-q4-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练期间每个验证 epoch 后显示并记录 Q4 指标",
    )
    parser.add_argument("--vote-iou", type=float, default=0.5, help="多数投票框匹配 IoU 阈值")
    parser.add_argument("--fp16", action="store_true", help="评估推理优化为 fp16")
    parser.add_argument("--optimize", action="store_true", help="评估前调用 RF-DETR optimize_for_inference")
    parser.add_argument("--use-global", action="store_true", help="大图切片时额外融合一次全局缩放预测")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    try:
        if args.cv_mode == "none":
            run_plain_train(args)
        elif args.cv_mode == "5-fold-cv":
            run_5_fold_cv(args)
        elif args.cv_mode == "5-fold-cv-standalone-test":
            run_5_fold_cv_standalone_test(args)
        elif args.cv_mode == "5-time-train+valid+test":
            run_5_time_train_valid_test(args)
        else:
            raise ValueError(f"未知 cv-mode: {args.cv_mode}")
    except KeyboardInterrupt:
        # _train_one 已在 rank0 打印可续训断点与续训方式，这里只做静默非零退出，避免重复 traceback。
        raise SystemExit(130)


def run_plain_train(args: argparse.Namespace) -> None:
    raw_data = read_json(args.raw_json)
    data_dir = Path(args.data_dir)
    if not (data_dir / "train" / "_annotations.coco.json").exists() or not (data_dir / "valid" / "_annotations.coco.json").exists():
        samples = dataset_items(raw_data)
        train_ids, valid_ids = split_ids(samples, args.image_root, args.val_ratio, args.cv_seed)
        write_rfdetr_dataset(
            raw_data,
            args.image_root,
            data_dir,
            {"train": train_ids, "valid": valid_ids},
            link_mode=args.link_mode,
        )
    checkpoint = _train_one(args, data_dir, Path(args.work_dir), resume=args.resume)
    LOGGER.info("训练完成: checkpoint=%s", checkpoint)


def _write_cv_summary(path: Path, payload: dict[str, Any], complete: bool) -> None:
    """增量写出交叉验证/重复实验汇总；complete=False 表示尚有折未完成（中断后可据此续跑）。"""
    out = dict(payload)
    out["complete"] = complete
    write_json(out, path)


def run_5_fold_cv(args: argparse.Namespace) -> None:
    raw_data = read_json(args.raw_json)
    samples = dataset_items(raw_data)
    folds = kfold_ids(samples, args.image_root, args.cv_folds, args.cv_seed)
    run_root = Path(args.work_dir) / "cv_5fold"
    data_root = ROOT / "data" / "cv_5fold"
    summary_path = run_root / "cv_summary.json"
    config = _cv_run_config(args, "5-fold-cv")
    _warn_cv_resume(args)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_ids, valid_ids) in enumerate(folds):
        fold_root = run_root / f"fold_{fold_idx}"
        result_path = fold_root / "fold_metric.json"
        if result_path.exists():
            LOGGER.info("fold_%d 已完成，跳过训练: %s", fold_idx, result_path)
            saved = read_json(result_path)
            _ensure_cv_result_matches(saved, config, result_path)
            fold_metrics.append(_metric_payload(saved))
            continue
        fold_data = data_root / f"fold_{fold_idx}"
        write_rfdetr_dataset(raw_data, args.image_root, fold_data, {"train": train_ids, "valid": valid_ids}, args.link_mode)
        checkpoint = _train_one(args, fold_data, fold_root / "train")

        valid_coco_path = fold_data / "valid" / "_annotations.coco.json"
        metric = _evaluate_checkpoint_on_coco(args, checkpoint, valid_coco_path, fold_data / "valid", fold_root / "eval")
        metric["fold"] = fold_idx
        metric["checkpoint"] = _as_posix_path(checkpoint)
        metric["config"] = config
        write_json(metric, result_path)
        fold_metrics.append(metric)
        _write_cv_summary(
            summary_path,
            {"mode": "5-fold-cv", "config": config, "fold_metrics": fold_metrics, "summary": _summary(fold_metrics)},
            complete=False,
        )

    output = {"mode": "5-fold-cv", "config": config, "fold_metrics": fold_metrics, "summary": _summary(fold_metrics)}
    _write_cv_summary(summary_path, output, complete=True)
    LOGGER.info("5-fold-cv 完成: %s", summary_path)


def run_5_fold_cv_standalone_test(args: argparse.Namespace) -> None:
    raw_data = read_json(args.raw_json)
    samples = dataset_items(raw_data)
    train_pool_ids, test_ids = split_ids(samples, args.image_root, args.test_ratio, args.cv_seed)
    train_pool_samples = subset_samples(samples, set(train_pool_ids))
    folds = kfold_ids(train_pool_samples, args.image_root, args.cv_folds, args.cv_seed)
    run_root = Path(args.work_dir) / "cv_5fold_standalone_test"
    data_root = ROOT / "data" / "cv_5fold_standalone_test"
    summary_path = run_root / "cv_summary.json"
    config = _cv_run_config(args, "5-fold-cv-standalone-test")
    _warn_cv_resume(args)
    test_data = data_root / "test_split"
    write_rfdetr_dataset(raw_data, args.image_root, test_data, {"test": test_ids}, args.link_mode)
    test_coco_path = test_data / "test" / "_annotations.coco.json"
    test_coco = read_json(test_coco_path)
    gts = coco_gt_boxes(test_coco)

    model_predictions: list[dict[str, list[dict[str, Any]]]] = []
    model_times: list[dict[str, float]] = []
    per_model_metrics: list[dict[str, Any]] = []
    image_ids = [str(img["id"]) for img in test_coco.get("images", [])]

    for fold_idx, (train_ids, valid_ids) in enumerate(folds):
        fold_root = run_root / f"fold_{fold_idx}"
        result_path = fold_root / "fold_result.json"
        if result_path.exists():
            LOGGER.info("fold_%d 已完成，跳过训练: %s", fold_idx, result_path)
            saved = read_json(result_path)
            _ensure_cv_result_matches(saved, config, result_path)
            model_predictions.append(saved["preds"])
            model_times.append(saved["times"])
            per_model_metrics.append(saved["metric"])
            continue
        fold_data = data_root / f"fold_{fold_idx}"
        write_rfdetr_dataset(raw_data, args.image_root, fold_data, {"train": train_ids, "valid": valid_ids}, args.link_mode)
        checkpoint = _train_one(args, fold_data, fold_root / "train")
        preds, times = _predict_checkpoint_on_coco(args, checkpoint, test_coco, test_data / "test")
        metric = _split_eval_q4(preds, gts, times, test_coco, args)
        metric["fold"] = fold_idx
        metric["checkpoint"] = _as_posix_path(checkpoint)
        write_json({"config": config, "metric": metric, "preds": preds, "times": times}, result_path)
        model_predictions.append(preds)
        model_times.append(times)
        per_model_metrics.append(metric)
        _write_cv_summary(
            summary_path,
            {
                "mode": "5-fold-cv-standalone-test",
                "config": config,
                "test_ratio": args.test_ratio,
                "map_score_thr": args.map_score_thr,
                "q4_score_thr": args.q4_score_thr,
                "vote_iou": args.vote_iou,
                "per_model_test_metrics": per_model_metrics,
                "per_model_summary": _summary(per_model_metrics),
            },
            complete=False,
        )

    ensemble_preds, ensemble_times = _ensemble_predictions(model_predictions, model_times, image_ids, args.vote_iou)
    ensemble_metrics = _split_eval_q4(ensemble_preds, gts, ensemble_times, test_coco, args)
    output = {
        "mode": "5-fold-cv-standalone-test",
        "config": config,
        "test_ratio": args.test_ratio,
        "map_score_thr": args.map_score_thr,
        "q4_score_thr": args.q4_score_thr,
        "vote_iou": args.vote_iou,
        "per_model_test_metrics": per_model_metrics,
        "per_model_summary": _summary(per_model_metrics),
        "ensemble_metrics": ensemble_metrics,
    }
    _write_cv_summary(summary_path, output, complete=True)
    write_json(ensemble_preds, run_root / "ensemble_predictions.json")
    LOGGER.info("5-fold-cv-standalone-test 完成: %s", summary_path)


def run_5_time_train_valid_test(args: argparse.Namespace) -> None:
    raw_data = read_json(args.raw_json)
    samples = dataset_items(raw_data)
    if args.test_ratio + args.repeat_val_ratio >= 1.0:
        raise ValueError("--test-ratio + --repeat-val-ratio 必须小于 1")
    val_ratio_in_pool = args.repeat_val_ratio / (1.0 - args.test_ratio)
    run_root = Path(args.work_dir) / "repeat_5_train_valid_test"
    data_root = ROOT / "data" / "repeat_5_train_valid_test"
    summary_path = run_root / "cv_summary.json"
    config = _cv_run_config(args, "5-time-train+valid+test")
    _warn_cv_resume(args)
    run_metrics: list[dict[str, Any]] = []

    for run_idx in range(args.repeat_times):
        seed = args.cv_seed + run_idx
        run_dir = run_root / f"run_{run_idx}"
        result_path = run_dir / "run_metric.json"
        if result_path.exists():
            LOGGER.info("run_%d 已完成，跳过训练: %s", run_idx, result_path)
            saved = read_json(result_path)
            _ensure_cv_result_matches(saved, config, result_path)
            run_metrics.append(_metric_payload(saved))
            continue
        train_pool_ids, test_ids = split_ids(samples, args.image_root, args.test_ratio, seed)
        train_pool_samples = subset_samples(samples, set(train_pool_ids))
        train_ids, valid_ids = split_ids(train_pool_samples, args.image_root, val_ratio_in_pool, seed)

        run_data = data_root / f"run_{run_idx}"
        write_rfdetr_dataset(
            raw_data,
            args.image_root,
            run_data,
            {"train": train_ids, "valid": valid_ids, "test": test_ids},
            args.link_mode,
        )
        checkpoint = _train_one(args, run_data, run_dir / "train")
        metric = _evaluate_checkpoint_on_coco(
            args,
            checkpoint,
            run_data / "test" / "_annotations.coco.json",
            run_data / "test",
            run_dir / "eval",
        )
        metric["run"] = run_idx
        metric["seed"] = seed
        metric["checkpoint"] = _as_posix_path(checkpoint)
        metric["config"] = config
        write_json(metric, result_path)
        run_metrics.append(metric)
        _write_cv_summary(
            summary_path,
            {
                "mode": "5-time-train+valid+test",
                "config": config,
                "repeat_times": args.repeat_times,
                "train_ratio": 1.0 - args.test_ratio - args.repeat_val_ratio,
                "val_ratio": args.repeat_val_ratio,
                "test_ratio": args.test_ratio,
                "run_metrics": run_metrics,
                "summary": _summary(run_metrics),
            },
            complete=False,
        )

    output = {
        "mode": "5-time-train+valid+test",
        "config": config,
        "repeat_times": args.repeat_times,
        "train_ratio": 1.0 - args.test_ratio - args.repeat_val_ratio,
        "val_ratio": args.repeat_val_ratio,
        "test_ratio": args.test_ratio,
        "run_metrics": run_metrics,
        "summary": _summary(run_metrics),
    }
    _write_cv_summary(summary_path, output, complete=True)
    LOGGER.info("5-time-train+valid+test 完成: %s", summary_path)


def _is_rank_zero() -> bool:
    """DDP 子进程由 Lightning 以 subprocess 重新拉起并设置 LOCAL_RANK；仅主进程返回 True。"""
    for var in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        value = os.environ.get(var)
        if value is not None and value.strip() not in ("", "0"):
            return False
    return True


def _auto_resume_checkpoint(output_dir: Path) -> Path | None:
    """返回可完整续训的 Lightning .ckpt：优先 last.ckpt，否则最新 checkpoint_*.ckpt。"""
    last = output_dir / "last.ckpt"
    if last.exists():
        return last
    candidates = sorted(output_dir.rglob("checkpoint_*.ckpt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _resolve_resume(args: argparse.Namespace, output_dir: Path, explicit: str | None) -> Path | None:
    """解析续训 checkpoint：显式 --resume 优先，否则按 --auto-resume 探测本目录断点。"""
    if explicit:
        return Path(explicit)
    if not getattr(args, "auto_resume", True):
        return None
    auto = _auto_resume_checkpoint(output_dir)
    if auto is not None and _is_rank_zero():
        LOGGER.info("检测到既有断点，自动续训: %s（如需从头训练请加 --no-auto-resume）", auto)
    return auto


def _progress_bar_mode(args: argparse.Namespace) -> str | None:
    """auto：仅交互式终端启用 tqdm，重定向/后台运行时关闭，避免日志被进度条刷屏。"""
    mode = getattr(args, "progress_bar", "auto")
    if mode == "auto":
        return "tqdm" if sys.stderr.isatty() else None
    if mode == "none":
        return None
    return mode


def _report_interrupt(output_dir: Path, reason: str = "KeyboardInterrupt") -> None:
    """训练中断（Ctrl+C 或异常）时，在 rank0 打印可续训断点与续训方式。"""
    if not _is_rank_zero():
        return
    resume_ckpt = _auto_resume_checkpoint(output_dir)
    LOGGER.warning("=" * 64)
    LOGGER.warning("训练被中断 (%s)，当前未完成 epoch 的进度不会保存。", reason)
    if resume_ckpt is not None:
        LOGGER.warning("最近可续训断点: %s", resume_ckpt)
        LOGGER.warning("普通训练续训: 追加 --resume %s", resume_ckpt)
        LOGGER.warning("CV/重复模式: 用相同 --work-dir 重跑即可跳过已完成折并自动续训未完成折")
    else:
        LOGGER.warning("尚未生成可续训断点(.ckpt)，重跑将从头开始。")
    LOGGER.warning("=" * 64)


def _warn_cv_resume(args: argparse.Namespace) -> None:
    """CV/重复模式下全局 --resume 无意义，按折自动续训，提示用户。"""
    if args.resume and _is_rank_zero():
        LOGGER.warning("CV/重复模式忽略全局 --resume；改为各折从自身 last.ckpt 自动续训")


def _cv_run_config(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    """记录影响 CV/重复实验可复用性的关键参数，避免同一 work-dir 混用旧结果。"""
    params = {
        "mode": mode,
        "raw_json": str(args.raw_json),
        "image_root": str(args.image_root),
        "model_size": args.model_size,
        "resolution": args.resolution or DEFAULT_RESOLUTION[args.model_size],
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_epochs": args.max_epochs,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "checkpoint_interval": args.checkpoint_interval,
        "eval_interval": args.eval_interval,
        "num_select": args.num_select,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "precision": args.precision,
        "val_bbox_only": args.val_bbox_only,
        "link_mode": args.link_mode,
        "cv_seed": args.cv_seed,
        "cv_folds": args.cv_folds,
        "repeat_times": args.repeat_times,
        "test_ratio": args.test_ratio,
        "repeat_val_ratio": args.repeat_val_ratio,
        "map_score_thr": args.map_score_thr,
        "q4_score_thr": args.q4_score_thr,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "large_thr": args.large_thr,
        "vote_iou": args.vote_iou,
        "fp16_eval": args.fp16,
        "optimize_eval": args.optimize,
        "use_global": args.use_global,
        "checkpoint_selection": "checkpoint_best_total_first",
    }
    serialized = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(), "params": params}


def _ensure_cv_result_matches(saved: Any, expected_config: dict[str, Any], path: Path) -> None:
    if not isinstance(saved, dict) or saved.get("config") != expected_config:
        actual_hash = saved.get("config", {}).get("hash") if isinstance(saved, dict) else None
        raise ValueError(
            "已存在的 CV 结果与当前参数不匹配，拒绝复用旧结果。"
            f" path={path}, expected_hash={expected_config['hash']}, actual_hash={actual_hash}. "
            "请更换 --work-dir，或确认旧结果不需要后手动删除对应 fold/run 目录。"
        )


def _metric_payload(saved: dict[str, Any]) -> dict[str, Any]:
    metric = saved.get("metric")
    return metric if isinstance(metric, dict) else saved


def _train_one(
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
    resume: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handler = add_file_logger(output_dir / "train.log") if _is_rank_zero() else None
    try:
        resume_path = _resolve_resume(args, output_dir, resume)
        model = _build_model(args)
        trainer_devices = _trainer_devices(args.device)
        train_kwargs = {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "resolution": args.resolution or DEFAULT_RESOLUTION[args.model_size],
            "batch_size": _parse_batch_size(args.batch_size),
            "grad_accum_steps": args.grad_accum_steps,
            "epochs": args.max_epochs,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "lr_encoder": args.lr_encoder,
            "weight_decay": args.weight_decay,
            "checkpoint_interval": args.checkpoint_interval,
            "eval_interval": args.eval_interval,
            "early_stopping": args.early_stopping,
            "early_stopping_patience": args.early_stopping_patience,
            "device": args.device,
            "devices": trainer_devices,
            "strategy": "ddp" if trainer_devices not in (1, "1") else "auto",
            "class_names": ["crack"],
            "tensorboard": True,
            "wandb": False,
            "progress_bar": _progress_bar_mode(args),
            "notes": {"project": "cctech_q4", "model_size": args.model_size},
        }
        if args.num_select is not None:
            train_kwargs["num_select"] = args.num_select
        if resume_path is not None:
            train_kwargs["resume"] = str(resume_path)
        if args.precision == "fp32":
            train_kwargs["amp"] = False
        try:
            _run_training(model, train_kwargs, args)
        except KeyboardInterrupt:
            _report_interrupt(output_dir)
            raise
        except Exception:
            if _is_rank_zero():
                LOGGER.exception("训练异常中断（非 Ctrl+C），原因见上方 traceback")
                _report_interrupt(output_dir, reason="训练异常")
            raise
        checkpoint = _find_best_checkpoint(output_dir)
        LOGGER.info("本次训练输出权重: %s", checkpoint)
        return checkpoint
    finally:
        remove_file_logger(log_handler)


class _Q4ValidationMetricsCallback(_LightningCallback):
    def __init__(self, map_score_thr: float, q4_score_thr: float, large_thr: int) -> None:
        self.map_score_thr = float(map_score_thr)
        self.q4_score_thr = float(q4_score_thr)
        self.large_thr = int(large_thr)
        self._preds_by_image: dict[str, list[dict[str, Any]]] = {}
        self._gts_by_image: dict[str, list[dict[str, Any]]] = {}
        self._times_by_image: dict[str, float] = {}
        self._image_infos: dict[str, dict[str, Any]] = {}
        self._batch_start = 0.0

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._preds_by_image = {}
        self._gts_by_image = {}
        self._times_by_image = {}
        self._image_infos = {}

    def on_validation_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_start = time.perf_counter()

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if not outputs:
            return
        elapsed_ms = max(0.0, (time.perf_counter() - self._batch_start) * 1000.0)
        results = outputs.get("results") or []
        targets = outputs.get("targets") or []
        per_image_ms = elapsed_ms / max(len(targets), 1)
        for result, target in zip(results, targets):
            image_id = _target_image_id(target)
            h, w = _target_hw(target)
            self._preds_by_image[image_id] = _result_to_q4_preds(result, min(self.map_score_thr, self.q4_score_thr))
            self._gts_by_image[image_id] = _target_to_q4_gts(target)
            self._times_by_image[image_id] = per_image_ms
            self._image_infos[image_id] = {
                "id": int(image_id) if image_id.isdigit() else image_id,
                "file_name": image_id,
                "width": w,
                "height": h,
            }

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        payloads = _distributed_gather_q4_payload(
            {
                "preds": self._preds_by_image,
                "gts": self._gts_by_image,
                "times": self._times_by_image,
                "images": self._image_infos,
            }
        )
        if not getattr(trainer, "is_global_zero", True):
            return
        preds_by_image: dict[str, list[dict[str, Any]]] = {}
        gts_by_image: dict[str, list[dict[str, Any]]] = {}
        times_by_image: dict[str, float] = {}
        images_by_id: dict[str, dict[str, Any]] = {}
        for payload in payloads:
            for image_id, preds in payload["preds"].items():
                preds_by_image.setdefault(image_id, preds)
            for image_id, gts in payload["gts"].items():
                gts_by_image.setdefault(image_id, gts)
            for image_id, elapsed_ms in payload["times"].items():
                times_by_image.setdefault(image_id, elapsed_ms)
            for image_id, image_info in payload["images"].items():
                images_by_id.setdefault(image_id, image_info)

        # mAP 与 Q4 各按自己的阈值过滤（与 eval.py 同口径）：mAP 用低阈值保完整 PR，Q4 用工作点
        map_preds = filter_preds_by_score(preds_by_image, self.map_score_thr)
        q4_preds = filter_preds_by_score(preds_by_image, self.q4_score_thr)
        map50 = compute_map50(map_preds, gts_by_image)
        metrics = evaluate_q4_metrics(
            q4_preds,
            gts_by_image,
            times_by_image,
            {"images": list(images_by_id.values())},
            large_thr=self.large_thr,
            map50=map50,
        )
        pl_module.log_dict(
            {f"q4/{key}": value for key, value in metrics.items()},
            on_epoch=True,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )
        LOGGER.info(
            "Q4 epoch=%s %s",
            getattr(trainer, "current_epoch", "?"),
            ", ".join(f"{key}={value:.6g}" for key, value in metrics.items()),
        )


def _patch_postprocess_bbox_only() -> Any:
    """方案 D：让 validation/test 只评 bbox，彻底跳过 mask 插值。

    seg 模型在 validation 时会把每个 query 的 mask 用 F.interpolate 上采样回原图全分辨率；
    数据集含 ~7460x9263 的离群巨图时，单次分配可达数十 GiB（num_select × H × W × 4B）直接 OOM。
    本函数消除三处 mask 插值：
      1) PostProcess.forward 对 regular 预测的 mask 上采样（rfdetr/models/postprocess.py）
      2) COCOEvalCallback 的 EMA 路径复用同一个 pl_module.postprocess（被 1 一并覆盖）
      3) COCOEvalCallback._convert_targets 把 GT mask 上采样到原图（rfdetr/training/callbacks/coco_eval.py）
    做法是把 outputs 里的 pred_masks / targets 里的 masks 在进入插值前剥离，让其走无 mask 分支。
    训练 loss（分割头监督）在 training_step 走 criterion，不经过 postprocess，故不受影响。

    返回 restore()，调用后还原所有被 patch 的方法。
    """
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

    orig_forward = PostProcess.forward
    orig_convert_targets = COCOEvalCallback._convert_targets

    def _bbox_only_forward(self: Any, outputs: Any, target_sizes: Any) -> Any:
        # 丢掉 pred_masks → PostProcess 走 else 分支，只产出 boxes/scores/labels，不做 mask 上采样。
        if isinstance(outputs, dict) and "pred_masks" in outputs:
            outputs = {k: v for k, v in outputs.items() if k != "pred_masks"}
        return orig_forward(self, outputs, target_sizes)

    def _bbox_only_convert_targets(self: Any, targets: Any) -> Any:
        # 去掉 GT masks → 不触发 _convert_targets 内的 GT mask 上采样。
        stripped = [
            {k: v for k, v in t.items() if k != "masks"} if isinstance(t, dict) and "masks" in t else t
            for t in targets
        ]
        return orig_convert_targets(self, stripped)

    PostProcess.forward = _bbox_only_forward
    COCOEvalCallback._convert_targets = _bbox_only_convert_targets

    def restore() -> None:
        PostProcess.forward = orig_forward
        COCOEvalCallback._convert_targets = orig_convert_targets

    return restore


def _set_callbacks_bbox_only(trainer: Any) -> None:
    """把 rfdetr 内置 COCOEvalCallback 切到 bbox-only：iou_type 不含 segm，不计算/记录 segm 指标。

    setup() 在 fit 时依据 _segmentation 选 iou_type，build_trainer 返回后置 False 即可在 setup 前生效。
    """
    found = False
    for callback in getattr(trainer, "callbacks", []):
        if type(callback).__name__ == "COCOEvalCallback":
            callback._segmentation = False
            found = True
    if not found:
        LOGGER.warning("未找到 COCOEvalCallback，--val-bbox-only 的 segm 关闭可能未生效")


def _precision_trainer_value(precision: str) -> str | None:
    """需要通过 trainer_kwargs 覆盖的 Lightning precision 串；auto/fp32 返回 None（不覆盖）。"""
    return {"fp16": "16-mixed", "bf16": "bf16-mixed"}.get(precision)


def _run_training(model: Any, train_kwargs: dict[str, Any], args: argparse.Namespace) -> None:
    """统一训练入口：按 --precision 覆盖 Lightning precision，并按需挂 Q4 验证指标 callback。

    fp32 已在调用方通过 train_kwargs["amp"]=False 生效（rfdetr -> "32-true"）；这里只处理需要
    monkey-patch build_trainer 的两件事：注入 fp16/bf16 precision、追加 Q4 callback。
    """
    import rfdetr.training as rfdetr_training

    precision_value = _precision_trainer_value(args.precision)
    enable_q4 = bool(args.log_q4_metrics)
    bbox_only = bool(getattr(args, "val_bbox_only", False))
    if precision_value is None and not enable_q4 and not bbox_only:
        model.train(**train_kwargs)
        return

    original_build_trainer = rfdetr_training.build_trainer

    def patched_build_trainer(*p_args: Any, **p_kwargs: Any) -> Any:
        if precision_value is not None:
            p_kwargs["precision"] = precision_value
        trainer = original_build_trainer(*p_args, **p_kwargs)
        if bbox_only:
            _set_callbacks_bbox_only(trainer)
        if enable_q4:
            trainer.callbacks.append(
                _Q4ValidationMetricsCallback(
                    map_score_thr=args.map_score_thr,
                    q4_score_thr=args.q4_score_thr,
                    large_thr=args.large_thr,
                )
            )
        return trainer

    rfdetr_training.build_trainer = patched_build_trainer
    restore_mask = _patch_postprocess_bbox_only() if bbox_only else None
    try:
        model.train(**train_kwargs)
    finally:
        rfdetr_training.build_trainer = original_build_trainer
        if restore_mask is not None:
            restore_mask()


def _distributed_gather_q4_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import torch.distributed as dist
    except Exception:
        return [payload]
    if not dist.is_available() or not dist.is_initialized():
        return [payload]
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    return [item for item in gathered if item is not None]


def _target_image_id(target: dict[str, Any]) -> str:
    value = target["image_id"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()
    return str(int(value))


def _target_hw(target: dict[str, Any]) -> tuple[int, int]:
    value = target["orig_size"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    h, w = value
    return int(h), int(w)


def _result_to_q4_preds(result: dict[str, Any], score_thr: float) -> list[dict[str, Any]]:
    boxes = result.get("boxes")
    scores = result.get("scores")
    if boxes is None or scores is None:
        return []
    boxes_np = boxes.detach().cpu().float().numpy()
    scores_np = scores.detach().cpu().float().numpy()
    preds: list[dict[str, Any]] = []
    for box, score in zip(boxes_np, scores_np):
        score_value = float(score)
        if score_value < score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        preds.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score_value, "label": "crack"})
    return sorted(preds, key=lambda item: float(item["score"]), reverse=True)


def _split_eval_q4(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    times_by_image: dict[str, float],
    coco: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float | int]:
    """离线评估：mAP 用 map 阈值过滤、Q4 用 q4 阈值过滤（与 eval.py / validation callback 同口径）。"""
    map_preds = filter_preds_by_score(preds_by_image, args.map_score_thr)
    q4_preds = filter_preds_by_score(preds_by_image, args.q4_score_thr)
    map50 = compute_map50(map_preds, gts_by_image)
    return evaluate_q4_metrics(q4_preds, gts_by_image, times_by_image, coco, large_thr=args.large_thr, map50=map50)


def _target_to_q4_gts(target: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = target.get("boxes")
    if boxes is None:
        return []
    h, w = _target_hw(target)
    boxes_np = boxes.detach().cpu().float().numpy()
    gts: list[dict[str, Any]] = []
    for cx, cy, bw, bh in boxes_np.tolist():
        x1 = (float(cx) - float(bw) / 2.0) * float(w)
        y1 = (float(cy) - float(bh) / 2.0) * float(h)
        x2 = (float(cx) + float(bw) / 2.0) * float(w)
        y2 = (float(cy) + float(bh) / 2.0) * float(h)
        if x2 <= x1 or y2 <= y1:
            continue
        gts.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": "crack"})
    return gts


def _build_model(args: argparse.Namespace) -> Any:
    try:
        import rfdetr.variants as variants
    except ImportError as exc:
        raise ImportError("缺少 RF-DETR 训练依赖，请先安装 RF-DETR/requirements.txt。") from exc
    klass = getattr(variants, MODEL_CLASSES[args.model_size])
    return klass(device=args.device)


def _find_best_checkpoint(work_dir: Path) -> Path:
    pth_patterns = (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "checkpoint_*.pth",
        "*.pth",
    )
    for pattern in pth_patterns:
        candidates = sorted(work_dir.rglob(pattern), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    # 中断态兜底：best_total.pth 仅在训练正常结束(on_fit_end)生成；若只剩 Lightning
    # .ckpt，回退到 last.ckpt / 最新 checkpoint_*.ckpt（RFDETRWrapper 会按
    # pretrain_weights 加载它，但优化器/EMA 等状态可能不完整）。
    for pattern in ("last.ckpt", "checkpoint_*.ckpt", "*.ckpt"):
        candidates = sorted(work_dir.rglob(pattern), key=lambda p: p.stat().st_mtime)
        if candidates:
            LOGGER.warning("未找到完整的 .pth 权重，回退到中断态断点: %s（评估状态可能不完整）", candidates[-1])
            return candidates[-1]
    raise FileNotFoundError(f"未找到 RF-DETR checkpoint: {work_dir}")


def _evaluate_checkpoint_on_coco(
    args: argparse.Namespace,
    checkpoint: Path,
    coco_path: Path,
    image_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    coco = read_json(coco_path)
    preds, times = _predict_checkpoint_on_coco(args, checkpoint, coco, image_root)
    metric = _split_eval_q4(preds, coco_gt_boxes(coco), times, coco, args)
    write_json(
        {
            "metrics": metric,
            "predictions": preds,
            "per_image_times": build_per_image_times(coco, times, large_thr=args.large_thr),
            "checkpoint": _as_posix_path(checkpoint),
            "map_score_thr": args.map_score_thr,
            "q4_score_thr": args.q4_score_thr,
            "bbox_only": bool(getattr(args, "val_bbox_only", False)),
        },
        out_dir / "q4_eval.json",
    )
    return metric


def _predict_checkpoint_on_coco(
    args: argparse.Namespace,
    checkpoint: Path,
    coco: dict[str, Any],
    image_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    bbox_only = bool(getattr(args, "val_bbox_only", False))
    predictor = RFDETRWrapper(
        checkpoint=checkpoint,
        model_size=args.model_size,
        device=args.device,
        score_thr=min(args.map_score_thr, args.q4_score_thr),
        fp16=args.fp16,
        optimize=args.optimize,
        bbox_only=bbox_only,
    )
    return predict_coco_images(
        predictor,
        coco,
        image_root=image_root,
        tile_size=args.tile_size,
        stride=args.stride,
        large_thr=args.large_thr,
        use_global=args.use_global,
        bbox_only=bbox_only,
    )


def _ensemble_predictions(
    model_predictions: list[dict[str, list[dict[str, Any]]]],
    model_times: list[dict[str, float]],
    image_ids: list[str],
    vote_iou: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    preds_by_image: dict[str, list[dict[str, Any]]] = {}
    times_by_image: dict[str, float] = {}
    for image_id in image_ids:
        per_model = [preds.get(image_id, []) for preds in model_predictions]
        preds_by_image[image_id] = _majority_vote_one_image(per_model, iou_thr=vote_iou)
        times_by_image[image_id] = sum(times.get(image_id, 0.0) for times in model_times)
    return preds_by_image, times_by_image


def _majority_vote_one_image(model_preds: list[list[dict[str, Any]]], iou_thr: float) -> list[dict[str, Any]]:
    min_votes = len(model_preds) // 2 + 1
    remaining: list[tuple[int, dict[str, Any]]] = []
    for model_idx, preds in enumerate(model_preds):
        for pred in preds:
            remaining.append((model_idx, pred))
    remaining.sort(key=lambda item: float(item[1].get("score", 0.0)), reverse=True)

    voted: list[dict[str, Any]] = []
    while remaining:
        seed_model, seed_pred = remaining.pop(0)
        group = [(seed_model, seed_pred)]
        used_models = {seed_model}
        rest: list[tuple[int, dict[str, Any]]] = []
        for model_idx, pred in remaining:
            if model_idx not in used_models and _box_iou(seed_pred, pred) >= iou_thr:
                group.append((model_idx, pred))
                used_models.add(model_idx)
            else:
                rest.append((model_idx, pred))
        remaining = rest

        if len(group) < min_votes:
            continue
        boxes = [pred for _, pred in group]
        voted.append(
            {
                "x1": mean(float(box["x1"]) for box in boxes),
                "y1": mean(float(box["y1"]) for box in boxes),
                "x2": mean(float(box["x2"]) for box in boxes),
                "y2": mean(float(box["y2"]) for box in boxes),
                "score": mean(float(box["score"]) for box in boxes),
                "label": "crack",
                "votes": len(group),
            }
        )
    return sorted(voted, key=lambda item: float(item["score"]), reverse=True)


def _box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    boxes_a = [[float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])]]
    boxes_b = [[float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]]
    return float(box_iou(boxes_a, boxes_b)[0, 0])


def _summary(metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    excluded_keys = {"fold", "run", "seed", "num_images"}
    numeric_keys = sorted(
        {
            key
            for metric in metrics
            for key, value in metric.items()
            if isinstance(value, int | float) and not isinstance(value, bool) and key not in excluded_keys
        }
    )
    result: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = [
            float(metric[key])
            for metric in metrics
            if key in metric and isinstance(metric[key], int | float) and not isinstance(metric[key], bool)
        ]
        result[key] = {"mean": mean(values) if values else 0.0, "std": pstdev(values) if len(values) > 1 else 0.0}
    return result


def _parse_batch_size(value: str) -> int | str:
    if value == "auto":
        return "auto"
    parsed = int(value)
    if parsed < 1:
        raise ValueError("--batch-size 必须为正整数或 auto")
    return parsed


def _trainer_devices(device: str) -> int | str:
    if not str(device).startswith("cuda"):
        return 1
    if ":" in str(device):
        return 1
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        cleaned = [part for part in visible.split(",") if part.strip()]
        return max(1, len(cleaned))
    try:
        import torch

        return max(1, int(torch.cuda.device_count()))
    except Exception:
        return 1


def _as_posix_path(path: str | Path) -> str:
    return Path(path).as_posix()


if __name__ == "__main__":
    main()
