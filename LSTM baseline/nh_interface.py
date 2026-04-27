from __future__ import annotations

import importlib
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cost_monitor import GpuMemoryMonitor


def ensure_deep_learning_environment() -> None:
    """跳过环境名称检查，允许在任何具备依赖的环境中运行。"""
    return


def ensure_neuralhydrology_available() -> None:
    """Raise a clear error if NeuralHydrology is not installed."""
    try:
        importlib.import_module("neuralhydrology")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "未检测到 neuralhydrology。请先在天地窥奥环境安装: pip install neuralhydrology==1.9.1"
        ) from exc


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def _list_subdirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return [p for p in path.iterdir() if p.is_dir()]


def _guess_run_dir_after_training(
    run_root: Path,
    experiment_name: str,
    before_dirs: Sequence[Path],
) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    before_set = {p.resolve() for p in before_dirs}
    after_dirs = _list_subdirs(run_root)

    # Preferred: newly created directory.
    new_dirs = [p for p in after_dirs if p.resolve() not in before_set]
    if new_dirs:
        return sorted(new_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    # Fallback: newest directory with experiment name prefix.
    prefix_dirs = [p for p in after_dirs if p.name.startswith(experiment_name)]
    if prefix_dirs:
        return sorted(prefix_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    # Last resort: run_root itself if it contains an output log.
    if (run_root / "output.log").exists():
        return run_root

    raise RuntimeError("训练结束后未能定位 NeuralHydrology 的 run 目录。")


def train_with_config(config_file: Path, gpu: Optional[int] = None) -> Path:
    """Train a NeuralHydrology run from one config and return its run directory."""
    ensure_deep_learning_environment()
    ensure_neuralhydrology_available()

    cfg = read_yaml(config_file)
    run_root = Path(cfg.get("run_dir", "runs"))
    experiment_name = str(cfg.get("experiment_name", "nh_run"))

    before_dirs = _list_subdirs(run_root)

    from neuralhydrology.nh_run import start_run

    if gpu is None:
        start_run(config_file=config_file)
    else:
        start_run(config_file=config_file, gpu=gpu)

    return _guess_run_dir_after_training(
        run_root=run_root,
        experiment_name=experiment_name,
        before_dirs=before_dirs,
    )


def evaluate_run(run_dir: Path, period: str, gpu: Optional[int] = None) -> None:
    """Evaluate an existing run for one period ('validation' or 'test')."""
    ensure_deep_learning_environment()
    ensure_neuralhydrology_available()
    from neuralhydrology.nh_run import eval_run

    if gpu is None:
        eval_run(run_dir=run_dir, period=period)
    else:
        eval_run(run_dir=run_dir, period=period, gpu=gpu)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def summarize_metrics_from_result_pickle(
    result_pickle: Path,
    preferred_metrics: Sequence[str] = ("NSE", "KGE", "RMSE"),
) -> Dict[str, float]:
    """Summarize per-basin metrics as basin-mean values.

    This parser is intentionally defensive because result dictionaries can vary by
    NeuralHydrology version, frequency setup, and target settings.
    """
    with result_pickle.open("rb") as fp:
        results = pickle.load(fp)

    collected: Dict[str, List[float]] = {m: [] for m in preferred_metrics}

    for _, basin_payload in results.items():
        if not isinstance(basin_payload, dict):
            continue

        freq_payload = None
        if "1D" in basin_payload and isinstance(basin_payload["1D"], dict):
            freq_payload = basin_payload["1D"]
        else:
            for _, value in basin_payload.items():
                if isinstance(value, dict):
                    freq_payload = value
                    break
        if not isinstance(freq_payload, dict):
            continue

        for metric in preferred_metrics:
            value = None
            if metric in freq_payload and _is_number(freq_payload[metric]):
                value = float(freq_payload[metric])
            else:
                # Some outputs may use target-specific names like runoff_NSE.
                for key, candidate in freq_payload.items():
                    if (
                        isinstance(key, str)
                        and metric.lower() in key.lower()
                        and _is_number(candidate)
                    ):
                        value = float(candidate)
                        break
            if value is not None:
                collected[metric].append(value)

    summary: Dict[str, float] = {}
    for metric, values in collected.items():
        if values:
            summary[metric] = float(sum(values) / len(values))
    return summary


def find_result_pickle(run_dir: Path, period: str) -> Path:
    """Locate a result pickle for a given period in a run directory."""
    period_dir = run_dir / period
    if not period_dir.exists():
        raise FileNotFoundError("未找到评估目录: {0}".format(period_dir))

    patterns = [
        f"{period}_results.p",
        "*_results.p",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(period_dir.rglob(pattern))

    if not candidates:
        raise FileNotFoundError("未找到评估结果 pickle，目录: {0}".format(period_dir))

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def parse_output_log_losses(run_dir: Path) -> List[Dict[str, float]]:
    """Parse epoch-level train/validation losses from NH output.log."""
    log_path = run_dir / "output.log"
    if not log_path.exists():
        return []

    train_pattern = re.compile(r"Epoch\s+(\d+)\s+average\s+loss:\s+([0-9eE.+\-]+)")
    val_pattern = re.compile(r"Epoch\s+(\d+)\s+average\s+validation\s+loss:\s+([0-9eE.+\-]+)")

    values: Dict[int, Dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        train_match = train_pattern.search(line)
        if train_match:
            epoch = int(train_match.group(1))
            values.setdefault(epoch, {})["train_loss"] = float(train_match.group(2))

        val_match = val_pattern.search(line)
        if val_match:
            epoch = int(val_match.group(1))
            values.setdefault(epoch, {})["val_loss"] = float(val_match.group(2))

    rows: List[Dict[str, float]] = []
    for epoch in sorted(values.keys()):
        row: Dict[str, float] = {"epoch": float(epoch)}
        row.update(values[epoch])
        rows.append(row)
    return rows


def train_and_eval(
    config_file: Path,
    eval_periods: Iterable[str],
    gpu: Optional[int] = None,
    monitor_vram: bool = True,
) -> Dict[str, Any]:
    """Train one config, evaluate requested periods, and return runtime metadata."""
    monitor = GpuMemoryMonitor(interval_seconds=2.0)
    if monitor_vram:
        monitor.start()

    start = time.perf_counter()
    run_dir = train_with_config(config_file=config_file, gpu=gpu)

    for period in eval_periods:
        evaluate_run(run_dir=run_dir, period=period, gpu=gpu)

    elapsed = time.perf_counter() - start
    peak_vram_mb = monitor.stop() if monitor_vram else 0.0

    return {
        "config_file": str(config_file),
        "run_dir": str(run_dir),
        "elapsed_seconds": float(elapsed),
        "peak_vram_mb": float(peak_vram_mb),
    }
