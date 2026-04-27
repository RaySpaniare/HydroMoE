from __future__ import annotations

# pyright: reportMissingImports=false

from contextlib import nullcontext
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from cost_monitor import GpuMemoryMonitor
from project_config import DEFAULT_LONG_TABLE_PARQUET, DYNAMIC_INPUTS, TARGET_VARIABLES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HYDROMOE_DIR = PROJECT_ROOT / "hydropy1.0MoE" / "hydromoe"
if HYDROMOE_DIR.exists() and str(HYDROMOE_DIR) not in sys.path:
    sys.path.insert(0, str(HYDROMOE_DIR))

from MoE_data_loader import FixedDataConfig, create_fixed_data_loaders
from MoE_metrics import compute_all_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: Optional[int] = None) -> torch.device:
    if gpu is None or gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


def build_fixed_data_config(csv_path: Optional[Path] = None) -> FixedDataConfig:
    return FixedDataConfig(
        csv_path=str(csv_path or DEFAULT_LONG_TABLE_PARQUET),
        feature_cols=list(DYNAMIC_INPUTS),
        target_col=TARGET_VARIABLES[0],
    )


class NativeRegionalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_dropout: float = 0.0,
        initial_forget_bias: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.lstm = nn.LSTM(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(output_dropout))
        self.head = nn.Linear(int(hidden_size), 1)
        self._init_forget_bias(float(initial_forget_bias))

    def _init_forget_bias(self, value: float) -> None:
        hidden_size = self.hidden_size
        for layer in range(self.lstm.num_layers):
            for suffix in ("ih", "hh"):
                bias = getattr(self.lstm, f"bias_{suffix}_l{layer}", None)
                if bias is None:
                    continue
                with torch.no_grad():
                    bias[hidden_size : 2 * hidden_size].fill_(value)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(features)
        last_hidden = outputs[:, -1, :]
        return self.head(self.dropout(last_hidden)).squeeze(-1)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return [value]


def _limit_torch_threads() -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _inverse_transform_values(scaler: Any, values: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    try:
        transformed = scaler.inverse_transform(arr)
    except Exception:
        return arr.reshape(-1)
    return np.asarray(transformed, dtype=np.float64).reshape(-1)


def _get_target_scaler(loader: DataLoader) -> Any:
    dataset = getattr(loader, "dataset", None)
    scalers = getattr(dataset, "scalers", None)
    if isinstance(scalers, dict):
        return scalers.get("target_scaler")
    return None


def _normalize_lr_schedule(schedule: Dict[Any, Any]) -> Dict[int, float]:
    normalized: Dict[int, float] = {}
    for key, value in schedule.items():
        normalized[int(key)] = float(value)
    return dict(sorted(normalized.items(), key=lambda item: item[0]))


def _apply_lr_schedule(optimizer: torch.optim.Optimizer, schedule: Dict[int, float], epoch_idx: int) -> None:
    if epoch_idx in schedule:
        for group in optimizer.param_groups:
            group["lr"] = schedule[epoch_idx]


def _train_one_epoch(
    model: NativeRegionalLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_noise_std: float,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    criterion = nn.MSELoss(reduction="sum")
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True).view(-1)

        if target_noise_std > 0:
            targets = targets + torch.randn_like(targets) * float(target_noise_std)

        optimizer.zero_grad(set_to_none=True)
        with amp_context:
            predictions = model(features)
            loss = criterion(predictions, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_count += int(targets.numel())

    return total_loss / max(total_count, 1)


@torch.no_grad()
def _evaluate_loader(
    model: NativeRegionalLSTM,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Any,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    pred_chunks: List[np.ndarray] = []
    true_chunks: List[np.ndarray] = []
    station_ids_all: List[Any] = []
    dates_all: List[Any] = []

    criterion = nn.MSELoss(reduction="sum")
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True).view(-1)
        with amp_context:
            predictions = model(features)

        batch_loss = criterion(predictions, targets)
        total_loss += float(batch_loss.item())
        total_count += int(targets.numel())

        pred_cpu = predictions.detach().float().cpu().numpy().reshape(-1)
        true_cpu = targets.detach().float().cpu().numpy().reshape(-1)
        pred_raw = _inverse_transform_values(target_scaler, pred_cpu)
        true_raw = _inverse_transform_values(target_scaler, true_cpu)

        pred_chunks.append(pred_raw)
        true_chunks.append(true_raw)
        station_ids_all.extend(_as_list(batch["station_id"]))
        dates_all.extend(_as_list(batch.get("end_date", batch.get("start_date"))))

    if pred_chunks:
        pred_all = np.concatenate(pred_chunks)
        true_all = np.concatenate(true_chunks)
    else:
        pred_all = np.array([], dtype=np.float64)
        true_all = np.array([], dtype=np.float64)

    metrics = compute_all_metrics(true_all, pred_all)
    if pred_chunks:
        prediction_df = pd.DataFrame(
            {
                "station_id": [str(station_id) for station_id in station_ids_all],
                "date": pd.to_datetime(dates_all),
                "pred_sim": pred_all,
                "obs": true_all,
            }
        )
    else:
        prediction_df = pd.DataFrame(columns=["station_id", "date", "pred_sim", "obs"])
    if not prediction_df.empty:
        prediction_df = prediction_df.sort_values(["station_id", "date"]).reset_index(drop=True)

    return {
        "loss": total_loss / max(total_count, 1),
        "metrics": metrics,
        "prediction_df": prediction_df,
        "predictions": pred_all,
        "targets": true_all,
    }


def run_native_training_job(
    trial_params: Dict[str, Any],
    batch_size: int,
    epochs: int,
    seed: int,
    num_workers: int,
    gpu: Optional[int] = None,
    csv_path: Optional[Path] = None,
    evaluate_test: bool = True,
) -> Dict[str, Any]:
    log_dir = Path("runs/worker_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_tag = "cpu" if gpu is None else str(gpu)
    log_file_path = log_dir / f"trial_seed_{seed}_gpu_{gpu_tag}.log"
    log_file = open(log_file_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        device = resolve_device(gpu)
        seed_everything(seed)

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = None
        _limit_torch_threads()

        data_config = build_fixed_data_config(csv_path=csv_path)
        train_loader, val_loader, test_loader = create_fixed_data_loaders(
            data_config,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            pin_memory=(device.type == "cuda"),
        )

        target_scaler = _get_target_scaler(train_loader)

        model = NativeRegionalLSTM(
            input_size=len(DYNAMIC_INPUTS),
            hidden_size=int(trial_params["hidden_size"]),
            output_dropout=float(trial_params.get("output_dropout", 0.0)),
            initial_forget_bias=float(trial_params.get("initial_forget_bias", 0.0)),
        ).to(device)

        if hasattr(torch, "compile") and os.name != "nt":
            try:
                model = torch.compile(model)
            except Exception:
                pass

        learning_rate_schedule = _normalize_lr_schedule(trial_params.get("learning_rate", {0: 1e-3}))
        initial_lr = learning_rate_schedule.get(0, next(iter(learning_rate_schedule.values())))
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        target_noise_std = float(trial_params.get("target_noise_std", 0.0))

        monitor = GpuMemoryMonitor(interval_seconds=2.0)
        monitor.start()
        start_time = time.perf_counter()

        history_rows: List[Dict[str, Any]] = []
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = 0
        best_score = float("-inf")
        best_val_loss = float("inf")
        best_val_metrics: Dict[str, float] = {}
        best_val_predictions = pd.DataFrame(columns=["station_id", "date", "pred_sim", "obs"])

        for epoch_idx in range(int(epochs)):
            _apply_lr_schedule(optimizer, learning_rate_schedule, epoch_idx)
            train_loss = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                target_noise_std=target_noise_std,
                scaler=scaler,
            )
            val_eval = _evaluate_loader(model, val_loader, device=device, target_scaler=target_scaler)
            val_loss = float(val_eval["loss"])
            val_metrics = dict(val_eval["metrics"])
            val_nse = float(val_metrics.get("NSE", float("nan")))
            score = val_nse if not np.isnan(val_nse) else float("-inf")

            history_rows.append(
                {
                    "epoch": int(epoch_idx + 1),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_R2": float(val_metrics.get("R2", float("nan"))),
                    "val_KGE": float(val_metrics.get("KGE", float("nan"))),
                    "val_RMSE": float(val_metrics.get("RMSE", float("nan"))),
                    "val_MSE": float(val_metrics.get("MSE", float("nan"))),
                    "val_bias": float(val_metrics.get("bias", float("nan"))),
                    "val_MAE": float(val_metrics.get("MAE", float("nan"))),
                    "val_NSE": float(val_metrics.get("NSE", float("nan"))),
                }
            )

            if (score > best_score) or (np.isclose(score, best_score) and val_loss < best_val_loss):
                best_score = score
                best_val_loss = val_loss
                best_epoch = int(epoch_idx + 1)
                best_val_metrics = val_metrics
                best_val_predictions = val_eval["prediction_df"].copy()
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}

            print(
                "[epoch {0:03d}/{1:03d}] train_loss={2:.6f} val_loss={3:.6f} val_NSE={4:.6f}".format(
                    int(epoch_idx + 1),
                    int(epochs),
                    float(train_loss),
                    float(val_loss),
                    float(val_metrics.get("NSE", float("nan"))),
                ),
                flush=True,
            )

        train_seconds = time.perf_counter() - start_time
        peak_vram_mb = monitor.stop()

        if best_state is not None:
            model.load_state_dict(best_state)

        test_eval: Dict[str, Any] = {
            "loss": float("nan"),
            "metrics": {},
            "prediction_df": pd.DataFrame(columns=["station_id", "date", "pred_sim", "obs"]),
        }
        inference_seconds = 0.0
        if evaluate_test:
            inference_start = time.perf_counter()
            test_eval = _evaluate_loader(model, test_loader, device=device, target_scaler=target_scaler)
            inference_seconds = time.perf_counter() - inference_start

        return {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "history_df": pd.DataFrame(history_rows),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_val_loss": best_val_loss,
            "best_val_metrics": best_val_metrics,
            "best_val_predictions": best_val_predictions,
            "test_loss": float(test_eval.get("loss", float("nan"))),
            "test_metrics": dict(test_eval.get("metrics", {})),
            "test_predictions": test_eval.get("prediction_df", pd.DataFrame()),
            "train_seconds": float(train_seconds),
            "inference_seconds": float(inference_seconds),
            "peak_vram_mb": float(peak_vram_mb),
            "device": str(device),
        }
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def flatten_learning_rate_schedule(schedule: Dict[Any, Any]) -> Dict[str, float]:
    normalized = _normalize_lr_schedule(schedule)
    return {f"lr_epoch_{epoch}": float(lr) for epoch, lr in normalized.items()}


def restore_learning_rate_schedule(row: Dict[str, Any], prefix: str = "lr_epoch_") -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    for key, value in row.items():
        if not key.startswith(prefix):
            continue
        try:
            epoch = int(key[len(prefix) :])
        except Exception:
            continue
        try:
            schedule[epoch] = float(value)
        except Exception:
            continue
    if not schedule:
        schedule = {0: 1e-3}
    return dict(sorted(schedule.items(), key=lambda item: item[0]))


def ensure_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["station_id", "date", "pred_sim", "obs"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["station_id", "date"]).reset_index(drop=True)
