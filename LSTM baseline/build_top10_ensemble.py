from __future__ import annotations

import argparse
import gc
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
from tqdm.auto import tqdm

from cost_monitor import CostTracker, render_cost_markdown
from native_lstm_baseline import ensure_prediction_frame, run_native_training_job, restore_learning_rate_schedule
from project_config import DYNAMIC_INPUTS, TARGET_VARIABLES, get_paths


def _select_topk_configs(
    search_metrics_csv: Path,
    top_k: int,
    primary_metric: str = "val_NSE",
    secondary_metric: str = "val_KGE",
) -> pd.DataFrame:
    if not search_metrics_csv.exists():
        raise FileNotFoundError("未找到随机搜索指标文件: {0}".format(search_metrics_csv))

    df = pd.read_csv(search_metrics_csv)
    if primary_metric not in df.columns:
        raise ValueError("搜索指标中缺少列: {0}".format(primary_metric))

    sort_columns = [primary_metric]
    ascending = [False]
    if secondary_metric in df.columns:
        sort_columns.append(secondary_metric)
        ascending.append(False)

    return df.sort_values(by=sort_columns, ascending=ascending).head(int(top_k)).copy()


def _ensure_fixed_constraints(cfg: Dict[str, Any], num_workers: int, cache_validation_data: bool) -> Dict[str, Any]:
    cfg["dynamic_inputs"] = list(DYNAMIC_INPUTS)
    cfg["target_variables"] = list(TARGET_VARIABLES)
    cfg["num_workers"] = int(num_workers)
    cfg["cache_validation_data"] = bool(cache_validation_data)
    return cfg


def _row_to_trial_params(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hidden_size": int(row["hidden_size"]),
        "output_dropout": float(row["output_dropout"]),
        "initial_forget_bias": float(row["initial_forget_bias"]),
        "target_noise_std": float(row["target_noise_std"]),
        "loss_function": str(row["loss_function"]),
        "learning_rate": restore_learning_rate_schedule(row),
    }


def _generate_ensemble_plans(
    topk_df: pd.DataFrame,
    seeds: Sequence[int],
    num_workers: int,
    cache_validation_data: bool,
) -> pd.DataFrame:
    paths = get_paths()

    records: List[Dict[str, Any]] = []
    member_id = 0

    for rank, row in enumerate(topk_df.itertuples(index=False), start=1):
        row_dict = row._asdict()
        trial_params = _row_to_trial_params(row_dict)
        for seed in seeds:
            member_id += 1
            member_row = {
                "member_id": int(member_id),
                "rank_in_top10": int(rank),
                "seed": int(seed),
                "source_trial_id": int(row_dict.get("trial_id", 0)),
                "batch_size": int(row_dict["batch_size"]),
                "hidden_size": int(row_dict["hidden_size"]),
                "output_dropout": float(row_dict["output_dropout"]),
                "initial_forget_bias": float(row_dict["initial_forget_bias"]),
                "target_noise_std": float(row_dict["target_noise_std"]),
                "loss_function": str(row_dict["loss_function"]),
                "regularization": row_dict.get("regularization", "None"),
            }
            member_row.update({f"lr_epoch_{epoch}": lr for epoch, lr in trial_params["learning_rate"].items()})
            records.append(member_row)

    records_df = pd.DataFrame(records)
    out_csv = paths["artifacts_root"] / "ensemble_member_plan.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    records_df.to_csv(out_csv, index=False, encoding="utf-8")
    return records_df


def _member_row_to_params(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hidden_size": int(row["hidden_size"]),
        "output_dropout": float(row["output_dropout"]),
        "initial_forget_bias": float(row["initial_forget_bias"]),
        "target_noise_std": float(row["target_noise_std"]),
        "loss_function": str(row["loss_function"]),
        "learning_rate": restore_learning_rate_schedule(row),
    }

def _member_worker(member_row: Dict[str, Any], gpu: Optional[int], num_workers: int) -> Dict[str, Any]:
    trial_params = _member_row_to_params(member_row)
    result = run_native_training_job(
        trial_params=trial_params,
        batch_size=int(member_row["batch_size"]),
        epochs=50,
        seed=int(member_row["seed"]),
        num_workers=int(num_workers),
        gpu=gpu,
        evaluate_test=True,
    )

    record: Dict[str, Any] = dict(member_row)
    record.update(
        {
            "run_dir": f"native_pytorch_seed_{int(member_row['seed'])}",
            "retrain_seconds": float(result["train_seconds"]),
            "inference_seconds": float(result["inference_seconds"]),
            "peak_vram_mb": float(result["peak_vram_mb"]),
            "best_epoch": int(result["best_epoch"]),
            "best_val_loss": float(result["best_val_loss"]),
            "test_loss": float(result["test_loss"]),
        }
    )
    record.update({f"val_{key}": value for key, value in result["best_val_metrics"].items()})
    record.update({f"test_{key}": value for key, value in result["test_metrics"].items()})

    history_df = result["history_df"].copy()
    if not history_df.empty:
        history_df.insert(0, "member_id", int(member_row["member_id"]))

    test_predictions = ensure_prediction_frame(result["test_predictions"])
    if not test_predictions.empty:
        test_predictions.insert(0, "member_id", int(member_row["member_id"]))

    payload = {
        "record": record,
        "history_df": history_df,
        "test_predictions": test_predictions,
    }

    del trial_params, result, record, history_df, test_predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return payload


def _normalize_gpu_ids(gpu_ids: Sequence[int]) -> List[int]:
    normalized = [int(v) for v in gpu_ids]
    if not normalized:
        raise ValueError("gpu_ids 为空，无法并行调度")
    return normalized


def _require_duckdb():
    try:
        import duckdb

        return duckdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("未安装 duckdb。请先安装: pip install duckdb") from exc


def _safe_save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv_fallback = path.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False, encoding="utf-8")
        print("Parquet 写入失败，已回退 CSV: {0}".format(csv_fallback), flush=True)
        return csv_fallback


def _load_prediction_table(paths: Dict[str, Path]) -> pd.DataFrame:
    parquet_path = paths["ensemble_member_predictions"]
    csv_path = parquet_path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"])
    return pd.DataFrame(columns=["member_id", "station_id", "date", "pred_sim", "obs"])


def _aggregate_member_predictions(member_predictions_df: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    if member_predictions_df.empty:
        raise RuntimeError("未收集到任何 test 预测结果，无法做集成中位数。")

    saved_path = _safe_save_parquet(ensure_prediction_frame(member_predictions_df), paths["ensemble_member_predictions"])
    duckdb = _require_duckdb()
    print("正在使用 DuckDB 聚合集成成员中位数...", flush=True)

    source_sql = "read_csv_auto(?)" if saved_path.suffix.lower() == ".csv" else "read_parquet(?)"
    query = f"""
    SELECT
        station_id,
        CAST(date AS TIMESTAMP) AS date,
        median(pred_sim) AS pred_median,
        avg(obs) AS obs,
        count(DISTINCT member_id) AS n_members
    FROM {source_sql}
    GROUP BY station_id, date
    ORDER BY station_id, date
    """

    with duckdb.connect(database=":memory:") as con:
        median_df = con.execute(query, [str(saved_path)]).df()

    _safe_save_parquet(median_df, paths["ensemble_median_predictions"])
    return median_df


def train_retrain_and_infer(
    member_plan_df: pd.DataFrame,
    gpu: Optional[int],
    gpu_ids: Optional[Sequence[int]],
    run_inference: bool,
    resume: bool = False,
    num_workers: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = get_paths()
    tracker = CostTracker.load_json(paths["cost_json"])

    run_records_path = paths["artifacts_root"] / "ensemble_run_records.csv"
    existing_by_member: Dict[int, Dict[str, Any]] = {}
    if resume and run_records_path.exists():
        existing_df = pd.read_csv(run_records_path)
        for rec in existing_df.to_dict(orient="records"):
            member_id = int(rec.get("member_id", 0))
            if member_id > 0:
                existing_by_member[member_id] = rec
        print("Resume 模式: 检测到 {0} 条既有 member 记录".format(len(existing_by_member)), flush=True)

    if resume and paths["ensemble_epoch_losses_csv"].exists():
        epoch_df_all = pd.read_csv(paths["ensemble_epoch_losses_csv"])
    else:
        epoch_df_all = pd.DataFrame(columns=["member_id", "epoch", "train_loss", "val_loss", "val_NSE"])

    existing_predictions_df = _load_prediction_table(paths) if resume else pd.DataFrame(columns=["member_id", "station_id", "date", "pred_sim", "obs"])
    if not existing_predictions_df.empty:
        existing_predictions_df["date"] = pd.to_datetime(existing_predictions_df["date"])

    pending_rows: List[Dict[str, Any]] = []
    for row in member_plan_df.to_dict(orient="records"):
        member_id = int(row.get("member_id", 0))
        if member_id <= 0:
            continue
        if resume and member_id in existing_by_member:
            print("[member {0:03d}] 已完成，resume 跳过".format(member_id), flush=True)
            continue
        pending_rows.append(row)

    if gpu_ids:
        records, epoch_df_all, member_predictions_df = _run_members_parallel(
            pending_rows=pending_rows,
            gpu_ids=gpu_ids,
            run_inference=run_inference,
            existing_by_member=existing_by_member,
            epoch_df_all=epoch_df_all,
            existing_predictions_df=existing_predictions_df,
            tracker=tracker,
            run_records_path=run_records_path,
            epoch_loss_path=paths["ensemble_epoch_losses_csv"],
            num_workers=num_workers,
        )
    else:
        records, epoch_df_all, member_predictions_df = _run_members_serial(
            pending_rows=pending_rows,
            gpu=gpu,
            run_inference=run_inference,
            existing_by_member=existing_by_member,
            epoch_df_all=epoch_df_all,
            existing_predictions_df=existing_predictions_df,
            tracker=tracker,
            run_records_path=run_records_path,
            epoch_loss_path=paths["ensemble_epoch_losses_csv"],
            num_workers=num_workers,
        )

    tracker.set_counts(search_trials=tracker.summary.search_trials or 100, ensemble_models=len(records))
    if tracker.summary.peak_vram_mb <= 0:
        tracker.add_note("建议外部监控命令: nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 2")
    tracker.save_json(paths["cost_json"])

    run_records_df = pd.DataFrame(records).sort_values("member_id")
    run_records_df.to_csv(run_records_path, index=False, encoding="utf-8")

    if not member_predictions_df.empty:
        member_predictions_df = ensure_prediction_frame(member_predictions_df)
        if not existing_predictions_df.empty:
            combined_predictions = pd.concat([existing_predictions_df, member_predictions_df], ignore_index=True)
            combined_predictions = combined_predictions.drop_duplicates(subset=["member_id", "station_id", "date"], keep="last")
        else:
            combined_predictions = member_predictions_df
    else:
        combined_predictions = existing_predictions_df

    if not combined_predictions.empty:
        combined_predictions["date"] = pd.to_datetime(combined_predictions["date"])
        _safe_save_parquet(combined_predictions, paths["ensemble_member_predictions"])

    return run_records_df, combined_predictions


def _run_members_serial(
    pending_rows: List[Dict[str, Any]],
    gpu: Optional[int],
    run_inference: bool,
    existing_by_member: Dict[int, Dict[str, Any]],
    epoch_df_all: pd.DataFrame,
    existing_predictions_df: pd.DataFrame,
    tracker: CostTracker,
    run_records_path: Path,
    epoch_loss_path: Path,
    num_workers: int,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    member_predictions_df = existing_predictions_df.copy()
    progress = tqdm(total=len(pending_rows), desc="Top-10 Ensemble", unit="member", dynamic_ncols=True)

    for row in pending_rows:
        member_id = int(row["member_id"])
        progress.set_postfix_str("member_id={0:03d}".format(member_id), refresh=False)
        result = _member_worker(member_row=row, gpu=gpu, num_workers=num_workers)

        record = result["record"]
        existing_by_member[member_id] = record
        tracker.add_time("retrain", float(record["retrain_seconds"]))
        tracker.add_time("inference", float(record["inference_seconds"]))
        tracker.update_peak_vram(float(record["peak_vram_mb"]))

        history_df = result.get("history_df", pd.DataFrame())
        if not history_df.empty:
            epoch_df_all = pd.concat([epoch_df_all, history_df], ignore_index=True)
            epoch_df_all = epoch_df_all.drop_duplicates(subset=["member_id", "epoch"], keep="last")
            epoch_df_all = epoch_df_all.sort_values(["member_id", "epoch"])
            epoch_df_all.to_csv(epoch_loss_path, index=False, encoding="utf-8")

        if run_inference:
            preds_df = result.get("test_predictions", pd.DataFrame())
            if not preds_df.empty:
                member_predictions_df = pd.concat([member_predictions_df, preds_df], ignore_index=True)

        pd.DataFrame(list(existing_by_member.values())).sort_values("member_id").to_csv(
            run_records_path,
            index=False,
            encoding="utf-8",
        )
        progress.update(1)

    progress.close()

    return list(existing_by_member.values()), epoch_df_all, member_predictions_df


def _run_members_parallel(
    pending_rows: List[Dict[str, Any]],
    gpu_ids: Sequence[int],
    run_inference: bool,
    existing_by_member: Dict[int, Dict[str, Any]],
    epoch_df_all: pd.DataFrame,
    existing_predictions_df: pd.DataFrame,
    tracker: CostTracker,
    run_records_path: Path,
    epoch_loss_path: Path,
    num_workers: int,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    if not pending_rows:
        return list(existing_by_member.values()), epoch_df_all, existing_predictions_df

    gpu_pool = _normalize_gpu_ids(gpu_ids)
    pending: Deque[Dict[str, Any]] = deque(pending_rows)
    future_to_task: Dict[Any, Tuple[Dict[str, Any], int]] = {}
    total_pending = len(pending_rows)
    completed = 0
    member_predictions_df = existing_predictions_df.copy()

    print(
        "【后台并行训练提示】主终端仅显示进度条，子进程的实时 Epoch 细节请查看 runs/worker_logs/ 目录下对应日志。",
        flush=True,
    )
    progress = tqdm(total=total_pending, desc="Top-10 Ensemble", unit="member", dynamic_ncols=True)
    with ProcessPoolExecutor(max_workers=len(gpu_pool)) as executor:
        for gpu_id in gpu_pool:
            if not pending:
                break
            member_row = pending.popleft()
            future = executor.submit(_member_worker, member_row, gpu_id, num_workers)
            future_to_task[future] = (member_row, gpu_id)

        while future_to_task:
            done, _ = wait(set(future_to_task.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                member_row, gpu_id = future_to_task.pop(future)
                result = future.result()
                member_id = int(member_row["member_id"])
                completed += 1
                progress.update(1)
                progress.set_postfix_str("member_id={0:03d}, gpu={1}".format(member_id, gpu_id), refresh=False)
                print("[{0}/{1}] member {2:03d} 完成 (gpu={3})".format(completed, total_pending, member_id, gpu_id), flush=True)

                record = result["record"]
                existing_by_member[member_id] = record
                tracker.add_time("retrain", float(record["retrain_seconds"]))
                tracker.add_time("inference", float(record["inference_seconds"]))
                tracker.update_peak_vram(float(record["peak_vram_mb"]))

                history_df = result.get("history_df", pd.DataFrame())
                if not history_df.empty:
                    epoch_df_all = pd.concat([epoch_df_all, history_df], ignore_index=True)
                    epoch_df_all = epoch_df_all.drop_duplicates(subset=["member_id", "epoch"], keep="last")
                    epoch_df_all = epoch_df_all.sort_values(["member_id", "epoch"])
                    epoch_df_all.to_csv(epoch_loss_path, index=False, encoding="utf-8")

                if run_inference:
                    preds_df = result.get("test_predictions", pd.DataFrame())
                    if not preds_df.empty:
                        member_predictions_df = pd.concat([member_predictions_df, preds_df], ignore_index=True)

                pd.DataFrame(list(existing_by_member.values())).sort_values("member_id").to_csv(
                    run_records_path,
                    index=False,
                    encoding="utf-8",
                )

                if pending:
                    next_row = pending.popleft()
                    next_future = executor.submit(_member_worker, next_row, gpu_id, num_workers)
                    future_to_task[next_future] = (next_row, gpu_id)

    progress.close()

    return list(existing_by_member.values()), epoch_df_all, member_predictions_df


def export_validation_test_loss_plots(run_records_df: pd.DataFrame, epoch_loss_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = get_paths()
    if run_records_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ordered = run_records_df.sort_values(["rank_in_top10", "member_id"]).reset_index(drop=True)
    representative_member = int(ordered.iloc[0]["member_id"])

    member_curve = epoch_loss_df[epoch_loss_df["member_id"] == float(representative_member)].copy()
    member_curve = member_curve.sort_values("epoch")

    if not member_curve.empty:
        axes[0].plot(member_curve["epoch"], member_curve["train_loss"], label="Train Loss", linewidth=1.8)
        axes[0].plot(member_curve["epoch"], member_curve["val_loss"], label="Validation Loss", linewidth=1.8)
        axes[0].set_title("Representative Member #{0} Loss Curves".format(representative_member))
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.2)
        axes[0].legend(frameon=False)
    else:
        axes[0].text(0.5, 0.5, "No epoch loss data available", ha="center", va="center")
        axes[0].set_axis_off()

    rmse_col = "test_RMSE" if "test_RMSE" in run_records_df.columns else "test_loss"
    rmse_df = run_records_df[["member_id", rmse_col]].dropna().sort_values("member_id")
    if not rmse_df.empty:
        axes[1].plot(rmse_df["member_id"], rmse_df[rmse_col], color="#2F5597", linewidth=1.6)
        axes[1].axhline(
            rmse_df[rmse_col].median(),
            color="#C00000",
            linestyle="--",
            linewidth=1.3,
            label="Median Test RMSE",
        )
        axes[1].set_title("Test Loss (RMSE) Across 100 Ensemble Members")
        axes[1].set_xlabel("Ensemble Member ID")
        axes[1].set_ylabel("Test RMSE")
        axes[1].grid(alpha=0.2)
        axes[1].legend(frameon=False)
    else:
        axes[1].text(0.5, 0.5, "No test RMSE data available", ha="center", va="center")
        axes[1].set_axis_off()

    paths["loss_plot_jpg"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["loss_plot_jpg"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["loss_plot_pdf"], bbox_inches="tight")
    plt.close(fig)


def update_cost_table() -> None:
    paths = get_paths()
    tracker = CostTracker.load_json(paths["cost_json"])
    render_cost_markdown(baseline=tracker.summary, output_path=paths["cost_markdown"])


def parse_args() -> argparse.Namespace:
    default_workers = 0 if os.name == "nt" else max(4, min(8, (os.cpu_count() or 8)))

    parser = argparse.ArgumentParser(description="构建 Top-10 x 10 seeds 的原生 PyTorch LSTM 集成并输出中位数预测")
    parser.add_argument("--top-k", type=int, default=10, help="从随机搜索中选择的最优配置数量")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(101, 111)),
        help="每个 Top 配置对应的随机种子列表",
    )
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
        help="执行重训练与测试推理；默认只生成计划",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="跳过重训练，直接读取已存在的 ensemble_member_predictions 并聚合",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点恢复：若已存在且有效的 member 结果则跳过，仅补跑未完成成员",
    )
    parser.set_defaults(cache_validation_data=(os.name != "nt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()

    topk_df = _select_topk_configs(
        search_metrics_csv=paths["search_metrics_csv"],
        top_k=args.top_k,
    )
    topk_df.to_csv(paths["top10_metrics_csv"], index=False, encoding="utf-8")

    member_plan_df = _generate_ensemble_plans(
        topk_df=topk_df,
        seeds=args.seeds,
        num_workers=args.num_workers,
        cache_validation_data=args.cache_validation_data,
    )
    print("已生成重训练计划，总模型数: {0}".format(len(member_plan_df)), flush=True)

    if args.aggregate_only:
        member_predictions_df = _load_prediction_table(paths)
        median_df = _aggregate_member_predictions(member_predictions_df, paths)
        update_cost_table()
        print("仅聚合完成，最终中位数样本数: {0}".format(len(median_df)), flush=True)
        print("成本表输出: {0}".format(paths["cost_markdown"]), flush=True)
        return

    if args.execute:
        run_records_df, member_predictions_df = train_retrain_and_infer(
            member_plan_df=member_plan_df,
            gpu=args.gpu,
            gpu_ids=args.gpu_ids,
            run_inference=True,
            resume=args.resume,
            num_workers=args.num_workers,
        )

        if paths["ensemble_epoch_losses_csv"].exists():
            epoch_loss_df = pd.read_csv(paths["ensemble_epoch_losses_csv"])
        else:
            epoch_loss_df = pd.DataFrame(columns=["member_id", "epoch", "train_loss", "val_loss"])

        export_validation_test_loss_plots(run_records_df, epoch_loss_df)
        median_df = _aggregate_member_predictions(member_predictions_df, paths)
        update_cost_table()
        print("Top-10 Ensemble 完成。", flush=True)
        print("最终中位数结果: {0}".format(paths["ensemble_median_predictions"]), flush=True)
        print("成本表输出: {0}".format(paths["cost_markdown"]), flush=True)
        print("损失图输出: {0}".format(paths["loss_plot_jpg"]), flush=True)
        print("损失图输出: {0}".format(paths["loss_plot_pdf"]), flush=True)
        print("中位数时间步数量: {0}".format(len(median_df)), flush=True)


if __name__ == "__main__":
    main()
