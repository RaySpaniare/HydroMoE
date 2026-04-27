from __future__ import annotations

import argparse
import gc
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler
from tqdm.auto import tqdm

from cost_monitor import CostTracker, count_lstm_parameters
from native_lstm_baseline import (
    flatten_learning_rate_schedule,
    restore_learning_rate_schedule,
    run_native_training_job,
)
from project_config import DYNAMIC_INPUTS, TARGET_VARIABLES, get_paths


def _lr_schedule_candidates() -> List[Dict[int, float]]:
    options_0 = [1e-3, 1e-2, 5e-2]
    options_10 = [5e-4, 1e-3, 5e-3]
    options_25 = [1e-4, 1e-3]
    schedules: List[Dict[int, float]] = []
    for lr0 in options_0:
        for lr10 in options_10:
            for lr25 in options_25:
                schedules.append({0: lr0, 10: lr10, 25: lr25})
    return schedules


def _search_space() -> Dict[str, List[Any]]:
    return {
        "hidden_size": [16, 32, 64, 128, 256],
        "batch_size": [32, 64, 128, 256],
        "output_dropout": [0.0, 0.2, 0.4],
        "initial_forget_bias": [-3, -1, 0, 1, 3],
        "target_noise_std": [0.0, 0.01, 0.02, 0.05, 0.1],
        "loss_function": ["NSE", "RMSE"],
        "regularization": [None],
        "learning_rate": _lr_schedule_candidates(),
    }


def _sample_trial_rows(n_trials: int, random_state: int) -> pd.DataFrame:
    param_sampler = ParameterSampler(
        param_distributions=_search_space(),
        n_iter=int(n_trials),
        random_state=int(random_state),
    )

    rows: List[Dict[str, Any]] = []
    for trial_id, sampled in enumerate(param_sampler, start=1):
        trial_row: Dict[str, Any] = {
            "trial_id": int(trial_id),
            "seed": int(1000 + trial_id),
            "hidden_size": int(sampled["hidden_size"]),
            "batch_size": int(sampled["batch_size"]),
            "output_dropout": float(sampled["output_dropout"]),
            "initial_forget_bias": float(sampled["initial_forget_bias"]),
            "target_noise_std": float(sampled["target_noise_std"]),
            "loss_function": str(sampled["loss_function"]),
            "regularization": json.dumps(sampled.get("regularization"), ensure_ascii=False),
            "learning_rate_schedule": json.dumps(sampled["learning_rate"], ensure_ascii=False, sort_keys=True),
        }
        trial_row.update(flatten_learning_rate_schedule(sampled["learning_rate"]))
        rows.append(trial_row)

    return pd.DataFrame(rows)


def generate_random_search_configs(
    n_trials: int,
    random_state: int,
    epochs: int,
    device: str,
    num_workers: int,
    cache_validation_data: bool,
) -> pd.DataFrame:
    paths = get_paths()
    trial_df = _sample_trial_rows(n_trials=n_trials, random_state=random_state)
    out_csv = paths["artifacts_root"] / "random_search_samples.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(out_csv, index=False, encoding="utf-8")
    return trial_df


def _trial_row_to_params(trial_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hidden_size": int(trial_row["hidden_size"]),
        "output_dropout": float(trial_row["output_dropout"]),
        "initial_forget_bias": float(trial_row["initial_forget_bias"]),
        "target_noise_std": float(trial_row["target_noise_std"]),
        "loss_function": str(trial_row["loss_function"]),
        "learning_rate": restore_learning_rate_schedule(trial_row),
    }


def _run_single_trial(
    trial_row: Dict[str, Any],
    gpu: Optional[int],
    num_workers: int,
    epochs: int,
) -> Dict[str, Any]:
    trial_params = _trial_row_to_params(trial_row)
    result = run_native_training_job(
        trial_params=trial_params,
        batch_size=int(trial_row["batch_size"]),
        epochs=int(epochs),
        seed=int(trial_row["seed"]),
        num_workers=int(num_workers),
        gpu=gpu,
        evaluate_test=False,
    )

    hidden_size = int(trial_row["hidden_size"])
    param_count = count_lstm_parameters(
        input_size=len(DYNAMIC_INPUTS),
        hidden_size=hidden_size,
        output_size=len(TARGET_VARIABLES),
    )

    record: Dict[str, Any] = dict(trial_row)
    record.update(
        {
            "elapsed_seconds": float(result["train_seconds"]),
            "peak_vram_mb": float(result["peak_vram_mb"]),
            "param_count": int(param_count),
            "best_epoch": int(result["best_epoch"]),
            "best_val_loss": float(result["best_val_loss"]),
        }
    )
    record.update({f"val_{key}": value for key, value in result["best_val_metrics"].items()})
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return record


def _trial_worker(trial_row: Dict[str, Any], gpu: Optional[int], num_workers: int, epochs: int) -> Dict[str, Any]:
    result = _run_single_trial(trial_row=trial_row, gpu=gpu, num_workers=num_workers, epochs=epochs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def _normalize_gpu_ids(gpu_ids: Sequence[int]) -> List[int]:
    normalized = [int(v) for v in gpu_ids]
    if not normalized:
        raise ValueError("gpu_ids 为空，无法并行调度")
    return normalized


def execute_search_serial(
    trial_df: pd.DataFrame,
    gpu: Optional[int],
    track_cost: bool,
    num_workers: int,
    epochs: int,
    resume: bool = False,
) -> pd.DataFrame:
    paths = get_paths()
    tracker = CostTracker.load_json(paths["cost_json"]) if (resume and paths["cost_json"].exists()) else CostTracker()

    row_by_trial: Dict[int, Dict[str, Any]] = {}
    if resume and paths["search_metrics_csv"].exists():
        existing_df = pd.read_csv(paths["search_metrics_csv"])
        if "trial_id" in existing_df.columns:
            for record in existing_df.to_dict(orient="records"):
                trial_id = int(record.get("trial_id", 0))
                if trial_id > 0:
                    row_by_trial[trial_id] = record

    pending_rows = [row for row in trial_df.to_dict(orient="records") if int(row["trial_id"]) not in row_by_trial]
    if resume:
        print("Resume 模式: 已完成 {0} 个 trial，待执行 {1} 个 trial".format(len(row_by_trial), len(pending_rows)), flush=True)

    progress = tqdm(total=len(pending_rows), desc="Random Search", unit="trial", dynamic_ncols=True)
    for trial_row in pending_rows:
        trial_id = int(trial_row["trial_id"])
        progress.set_postfix_str("trial_id={0:03d}".format(trial_id), refresh=False)
        print("[{0}/{1}] 训练并验证 trial_{2:03d}".format(progress.n + 1, len(pending_rows), trial_id), flush=True)
        row = _run_single_trial(trial_row=trial_row, gpu=gpu, num_workers=num_workers, epochs=epochs)
        row_by_trial[trial_id] = row

        tracker.set_param_count(int(row["param_count"]))
        tracker.add_time("search", float(row["elapsed_seconds"]))
        tracker.update_peak_vram(float(row["peak_vram_mb"]))

        partial_df = pd.DataFrame(list(row_by_trial.values())).sort_values(by="val_NSE", ascending=False, na_position="last")
        partial_df.to_csv(paths["search_metrics_csv"], index=False, encoding="utf-8")
        if track_cost:
            tracker.save_json(paths["cost_json"])
        progress.update(1)

    progress.close()

    tracker.set_counts(search_trials=len(trial_df), ensemble_models=100)
    if tracker.summary.peak_vram_mb <= 0:
        tracker.add_note("未检测到 nvidia-smi，建议手动记录: nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 2")
    if track_cost:
        tracker.save_json(paths["cost_json"])

    df = pd.DataFrame(list(row_by_trial.values())).sort_values(by="val_NSE", ascending=False, na_position="last")
    paths["search_metrics_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["search_metrics_csv"], index=False, encoding="utf-8")
    return df


def execute_search_parallel(
    trial_df: pd.DataFrame,
    gpu_ids: Sequence[int],
    track_cost: bool,
    num_workers: int,
    epochs: int,
    resume: bool = False,
) -> pd.DataFrame:
    paths = get_paths()
    tracker = CostTracker.load_json(paths["cost_json"]) if (resume and paths["cost_json"].exists()) else CostTracker()

    row_by_trial: Dict[int, Dict[str, Any]] = {}
    if resume and paths["search_metrics_csv"].exists():
        existing_df = pd.read_csv(paths["search_metrics_csv"])
        if "trial_id" in existing_df.columns:
            for record in existing_df.to_dict(orient="records"):
                trial_id = int(record.get("trial_id", 0))
                if trial_id > 0:
                    row_by_trial[trial_id] = record

    pending_rows = [row for row in trial_df.to_dict(orient="records") if int(row["trial_id"]) not in row_by_trial]
    if resume:
        print("Resume 模式: 已完成 {0} 个 trial，待执行 {1} 个 trial".format(len(row_by_trial), len(pending_rows)), flush=True)
    if not pending_rows:
        df_done = pd.DataFrame(list(row_by_trial.values()))
        if not df_done.empty:
            df_done = df_done.sort_values(by="val_NSE", ascending=False, na_position="last")
        return df_done

    gpu_pool = _normalize_gpu_ids(gpu_ids)
    pending: Deque[Dict[str, Any]] = deque(pending_rows)
    future_to_task: Dict[Any, Tuple[Dict[str, Any], int]] = {}
    total_pending = len(pending_rows)
    completed = 0

    print(
        "【后台并行训练提示】主终端仅显示进度条，子进程的实时 Epoch 细节请查看 runs/worker_logs/ 目录下对应日志。",
        flush=True,
    )
    progress = tqdm(total=total_pending, desc="Random Search", unit="trial", dynamic_ncols=True)
    with ProcessPoolExecutor(max_workers=len(gpu_pool)) as executor:
        for gpu_id in gpu_pool:
            if not pending:
                break
            trial_row = pending.popleft()
            future = executor.submit(_trial_worker, trial_row, gpu_id, num_workers, epochs)
            future_to_task[future] = (trial_row, gpu_id)

        while future_to_task:
            done, _ = wait(set(future_to_task.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                trial_row, gpu_id = future_to_task.pop(future)
                row = future.result()

                completed += 1
                trial_id = int(trial_row["trial_id"])
                progress.update(1)
                progress.set_postfix_str("trial_id={0:03d}, gpu={1}".format(trial_id, gpu_id), refresh=False)
                print("[{0}/{1}] 完成 trial_{2:03d} (gpu={3})".format(completed, total_pending, trial_id, gpu_id), flush=True)

                row_by_trial[trial_id] = row
                tracker.set_param_count(int(row["param_count"]))
                tracker.add_time("search", float(row["elapsed_seconds"]))
                tracker.update_peak_vram(float(row["peak_vram_mb"]))

                partial_df = pd.DataFrame(list(row_by_trial.values())).sort_values(by="val_NSE", ascending=False, na_position="last")
                partial_df.to_csv(paths["search_metrics_csv"], index=False, encoding="utf-8")
                if track_cost:
                    tracker.save_json(paths["cost_json"])

                if pending:
                    next_row = pending.popleft()
                    next_future = executor.submit(_trial_worker, next_row, gpu_id, num_workers, epochs)
                    future_to_task[next_future] = (next_row, gpu_id)

    progress.close()

    tracker.set_counts(search_trials=len(trial_df), ensemble_models=100)
    if tracker.summary.peak_vram_mb <= 0:
        tracker.add_note("未检测到 nvidia-smi，建议手动记录: nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 2")
    if track_cost:
        tracker.save_json(paths["cost_json"])

    df = pd.DataFrame(list(row_by_trial.values())).sort_values(by="val_NSE", ascending=False, na_position="last")
    paths["search_metrics_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["search_metrics_csv"], index=False, encoding="utf-8")
    return df


def _default_num_workers() -> int:
    return 0 if os.name == "nt" else max(4, min(8, (os.cpu_count() or 8)))


def _suggest_parallel_commands(config_dir: Path) -> str:
    return (
        "并行训练命令示例:\n"
        "python run_random_search.py --execute --gpu-ids 0 0 0 0\n"
        "并行评估命令示例:\n"
        "python build_top10_ensemble.py --execute --gpu-ids 0 0 0 0"
    ).format(config_dir)


def parse_args() -> argparse.Namespace:
    default_workers = _default_num_workers()

    parser = argparse.ArgumentParser(description="生成并执行 100 次原生 PyTorch LSTM 随机搜索")
    parser.add_argument("--n-trials", type=int, default=100, help="随机搜索试验数")
    parser.add_argument("--random-state", type=int, default=42, help="ParameterSampler 随机种子")
    parser.add_argument("--epochs", type=int, default=50, help="每个 trial 的训练 epoch")
    parser.add_argument("--device", type=str, default="cuda:0", help="保留兼容参数；当前原生训练直接使用 --gpu / --gpu-ids")
    parser.add_argument("--gpu", type=int, default=None, help="单进程模式下使用的 GPU 编号，CPU 可设为 -1")
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="并行模式下可用的 GPU 列表，例如: --gpu-ids 0 0 0 0",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        help="DataLoader worker 数（Windows建议设为0）",
    )
    parser.add_argument(
        "--cache-validation-data",
        dest="cache_validation_data",
        action="store_true",
        help="保留兼容参数；原生数据加载器会自动使用进程内缓存",
    )
    parser.add_argument(
        "--no-cache-validation-data",
        dest="cache_validation_data",
        action="store_false",
        help="保留兼容参数",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="若开启则直接训练并验证；否则仅生成 100 个 trial 采样记录",
    )
    parser.add_argument(
        "--skip-cost-tracking",
        action="store_true",
        help="不写入 cost_summary.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点恢复：跳过 search_validation_metrics.csv 中已完成的 trial",
    )
    parser.set_defaults(cache_validation_data=(os.name != "nt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()

    trial_df = generate_random_search_configs(
        n_trials=args.n_trials,
        random_state=args.random_state,
        epochs=args.epochs,
        device=args.device,
        num_workers=args.num_workers,
        cache_validation_data=args.cache_validation_data,
    )

    print("已生成 {0} 个 trial 采样记录。".format(len(trial_df)), flush=True)
    print(_suggest_parallel_commands(config_dir=paths["search_configs_root"]), flush=True)

    if args.execute:
        if args.gpu_ids:
            metrics_df = execute_search_parallel(
                trial_df=trial_df,
                gpu_ids=args.gpu_ids,
                track_cost=not args.skip_cost_tracking,
                num_workers=args.num_workers,
                epochs=args.epochs,
                resume=args.resume,
            )
        else:
            metrics_df = execute_search_serial(
                trial_df=trial_df,
                gpu=args.gpu,
                track_cost=not args.skip_cost_tracking,
                num_workers=args.num_workers,
                epochs=args.epochs,
                resume=args.resume,
            )
        print("随机搜索完成，验证指标已保存到:", flush=True)
        print(paths["search_metrics_csv"], flush=True)
        print(metrics_df.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
