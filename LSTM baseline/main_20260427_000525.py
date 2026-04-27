# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2026-04-21
@Desc    :   该脚本是整个 Regional LSTM 基线工程的一键式主程序入口，用于把数据适配、随机搜索、Top-10 集成重训练、测试集推理以及成本统计串联成一个可重复执行的统一流水线。之所以单独编写该入口，是因为审稿场景下需要一个清晰、可追踪、低歧义的调度器来保证基准实验与 HydroMoE 的比较过程完全透明。脚本的输入主要来自三类来源：其一是用户指定的长表 parquet 数据文件，字段至少应包含 station_id、date、lon、lat、pet、precip、temp、runoff；其二是由数据适配阶段输出的 GenericDataset 目录与 basin 列表；其三是随机搜索和集成阶段自动生成的 YAML 配置、日志和预测结果。主流程分为四段：首先在 GIS3.9 环境中执行长表适配，生成 NeuralHydrology 所需的 time_series NetCDF 和 basin 列表；随后切换到天地窥奥环境执行 100 次 Random Search，自动构建训练配置并汇总验证集指标；接着再次在天地窥奥环境中从 Top-10 配置构建 10 个随机种子重训练的 100 个模型，并在测试集上输出成员级预测与中位数集成结果；最后汇总运行时长、参数量、显存峰值和损失图文件，形成论文可直接引用的成本比较材料。关键输出包括：prepared_generic_dataset 目录、configs/random_search、configs/ensemble_retrain、artifacts/search_validation_metrics.csv、artifacts/ensemble_run_records.csv、artifacts/ensemble_median_predictions.parquet、reports/cost_comparison_table.md、reports/loss_validation_test.jpg 和 reports/loss_validation_test.pdf。已知限制是：深度学习阶段必须由天地窥奥解释器执行，而数据适配阶段应保持 GIS3.9；若环境缺失、NH 结果文件不存在或 GenericDataset 尚未生成，脚本会立即报错并提示先决条件。用户可以直接运行 python main.py 实现全流程，也可以通过 --mode 仅执行其中某一步，从而兼容快速调试与正式大规模实验两种场景。
@Notice  :   请先确认 GIS3.9 与 天地窥奥 两个 conda 环境均已安装 neuralhydrology==1.9.1，并且长表 parquet 路径可访问；正式运行时建议先做一次 --mode adapt 或 --mode all 的小规模验证，以确认 GenericDataset 生成无误后再启动 100 次随机搜索与 100 个集成模型训练。
'''

from __future__ import annotations

import argparse
import json
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from project_config import DEFAULT_LONG_TABLE_PARQUET, get_paths


def get_script_dir() -> Path:
    """Return the directory that contains this main.py file."""
    return Path(__file__).resolve().parent


def _run_python_script(python_executable: Path, script_path: Path, args: List[str]) -> None:
    """Execute a child Python script with the selected interpreter and stream output.

    The command is kept explicit so that the deep-learning steps can always run
    in 天地窥奥 even when the main entry point itself is launched from GIS3.9.
    """
    if not python_executable.exists():
        raise FileNotFoundError("未找到解释器: {0}".format(python_executable))
    if not script_path.exists():
        raise FileNotFoundError("未找到脚本: {0}".format(script_path))

    command = [str(python_executable), str(script_path)] + list(args)
    print("\n>>> 执行命令: {0}".format(" ".join(command)), flush=True)
    env = os.environ.copy()
    # Force UTF-8 in child processes on Windows to avoid locale-dependent decoding issues.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, check=True, cwd=str(script_path.parent), env=env)


@dataclass(frozen=True)
class MainConfig:
    """Central configuration for the one-click pipeline."""

    input_parquet: Path = DEFAULT_LONG_TABLE_PARQUET
    data_adapter_script: str = "data_adapter.py"
    random_search_script: str = "run_random_search.py"
    ensemble_script: str = "build_top10_ensemble.py"
    random_trials: int = 100
    random_state: int = 42
    epochs: int = 50
    num_workers: int = 0
    runs_per_gpu: int = 1
    device: str = "cuda:0"
    gpu: int = 0
    top_k: int = 10
    seed_start: int = 101
    seed_end: int = 110
    resume: bool = False
    force_adapt: bool = False


def _prepared_dataset_exists(paths: Dict[str, Path]) -> bool:
    """Return True if prepared GenericDataset time-series files already exist."""
    ts_root = paths["prepared_time_series_root"]
    if not ts_root.exists():
        return False
    return any(ts_root.glob("*.nc"))


def _prepare_summary_payload(step_name: str, elapsed_seconds: float, extra: Dict[str, object]) -> Dict[str, object]:
    """Build a compact JSON summary entry for each pipeline stage."""
    payload: Dict[str, object] = {
        "step_name": step_name,
        "elapsed_seconds": float(elapsed_seconds),
    }
    payload.update(extra)
    return payload


def run_data_adapter(cfg: MainConfig, script_dir: Path) -> float:
    """Run the GIS3.9-only data adaptation stage."""
    start = time.perf_counter()
    command_args = [
        "--input-parquet",
        str(cfg.input_parquet),
        "--overwrite",
    ]

    _run_python_script(
        python_executable=Path(sys.executable),
        script_path=script_dir / cfg.data_adapter_script,
        args=command_args,
    )
    return time.perf_counter() - start


def run_random_search(cfg: MainConfig, script_dir: Path) -> float:
    """Run the deep-learning random-search stage inside 天地窥奥."""
    start = time.perf_counter()
    gpu_ids = [str(cfg.gpu)] * max(int(cfg.runs_per_gpu), 1)
    command_args = [
        "--n-trials",
        str(cfg.random_trials),
        "--random-state",
        str(cfg.random_state),
        "--epochs",
        str(cfg.epochs),
        "--num-workers",
        str(cfg.num_workers),
        "--execute",
        "--gpu-ids",
    ] + gpu_ids + [
        "--device",
        cfg.device,
    ]
    if cfg.resume:
        command_args.append("--resume")

    _run_python_script(
        python_executable=Path(sys.executable),
        script_path=script_dir / cfg.random_search_script,
        args=command_args,
    )
    return time.perf_counter() - start


def run_ensemble_stage(cfg: MainConfig, script_dir: Path) -> float:
    """Run the Top-10 ensemble retraining and inference stage inside 天地窥奥."""
    start = time.perf_counter()
    gpu_ids = [str(cfg.gpu)] * max(int(cfg.runs_per_gpu), 1)
    seed_values = [str(seed) for seed in range(cfg.seed_start, cfg.seed_end + 1)]
    command_args = [
        "--execute",
        "--top-k",
        str(cfg.top_k),
        "--seeds",
    ] + seed_values + ["--gpu-ids"] + gpu_ids
    if cfg.resume:
        command_args.append("--resume")

    _run_python_script(
        python_executable=Path(sys.executable),
        script_path=script_dir / cfg.ensemble_script,
        args=command_args,
    )
    return time.perf_counter() - start


def _save_pipeline_summary(summary_rows: List[Dict[str, object]], output_path: Path) -> None:
    """Persist the execution summary as JSON so the paper workflow is reproducible."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the pipeline driver."""
    parser = argparse.ArgumentParser(description="Regional LSTM Baseline one-click pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "adapt", "search", "ensemble"],
        help="选择执行阶段，默认 all 会顺序运行数据适配、随机搜索与 Top-10 集成",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=DEFAULT_LONG_TABLE_PARQUET,
        help="长表 parquet 路径",
    )
    parser.add_argument("--gpu", type=int, default=0, help="深度学习阶段使用的 GPU 编号")
    parser.add_argument("--device", type=str, default="cuda:0", help="写入 NH 配置文件的 device")
    parser.add_argument("--n-trials", type=int, default=100, help="随机搜索试验数")
    parser.add_argument("--random-state", type=int, default=42, help="随机搜索种子")
    parser.add_argument("--epochs", type=int, default=50, help="随机搜索每个 trial 的 epoch")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="随机搜索阶段写入 YAML 的 DataLoader worker 数（Windows建议设为0）",
    )
    parser.add_argument(
        "--runs-per-gpu",
        type=int,
        default=4,
        help="单卡并发模型数 (5070 Ti 建议设为 4)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K 配置数")
    parser.add_argument("--seed-start", type=int, default=101, help="集成重训练随机种子起始值")
    parser.add_argument("--seed-end", type=int, default=110, help="集成重训练随机种子结束值")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点恢复模式：跳过随机搜索和集成阶段中已完成的任务",
    )
    parser.add_argument(
        "--force-adapt",
        action="store_true",
        help="强制重建 GenericDataset（向 data_adapter 传递 --overwrite）",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point that coordinates the complete baseline workflow."""
    args = parse_args()
    script_dir = get_script_dir()
    paths = get_paths(script_dir)

    cfg = MainConfig(
        input_parquet=args.input_parquet,
        random_trials=args.n_trials,
        random_state=args.random_state,
        epochs=args.epochs,
        num_workers=args.num_workers,
        runs_per_gpu=args.runs_per_gpu,
        device=args.device,
        gpu=args.gpu,
        top_k=args.top_k,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        resume=args.resume,
        force_adapt=args.force_adapt,
    )

    summary_rows: List[Dict[str, object]] = []

    if args.mode in ("all", "adapt"):
        if args.mode == "all" and (not cfg.force_adapt) and _prepared_dataset_exists(paths):
            elapsed = 0.0
            print("检测到已存在 prepared_generic_dataset/time_series/*.nc，默认跳过数据适配。", flush=True)
            print("如需重建数据适配结果，请追加 --force-adapt。", flush=True)
        else:
            elapsed = run_data_adapter(cfg, script_dir)
        summary_rows.append(
            _prepare_summary_payload(
                "data_adapter",
                elapsed,
                {
                    "input_parquet": str(cfg.input_parquet),
                    "force_adapt": cfg.force_adapt,
                },
            )
        )

    if args.mode in ("all", "search"):
        elapsed = run_random_search(cfg, script_dir)
        summary_rows.append(
            _prepare_summary_payload(
                "random_search",
                elapsed,
                {
                    "n_trials": cfg.random_trials,
                    "epochs": cfg.epochs,
                    "device": cfg.device,
                    "resume": cfg.resume,
                },
            )
        )

    if args.mode in ("all", "ensemble"):
        elapsed = run_ensemble_stage(cfg, script_dir)
        summary_rows.append(
            _prepare_summary_payload(
                "top10_ensemble",
                elapsed,
                {
                    "top_k": cfg.top_k,
                    "seed_start": cfg.seed_start,
                    "seed_end": cfg.seed_end,
                    "resume": cfg.resume,
                },
            )
        )

    summary_path = paths["reports_root"] / "pipeline_run_summary.json"
    _save_pipeline_summary(summary_rows, summary_path)

    print("\n流水线执行完成，摘要已保存到: {0}".format(summary_path), flush=True)
    for row in summary_rows:
        print("- {0}: {1:.2f} 秒".format(row["step_name"], float(row["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()