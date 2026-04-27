from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _query_nvidia_smi_memory_mb() -> Optional[float]:
    """Return the maximum used memory (MB) across visible GPUs.

    Returns None when nvidia-smi is not available or parsing fails.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            values: List[float] = []
            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                values.append(float(info.used) / (1024.0 * 1024.0))
            if values:
                return max(values)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except ImportError:
        pass
    except Exception:
        pass

    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    values: List[float] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


class GpuMemoryMonitor:
    """Background monitor that tracks peak GPU VRAM via nvidia-smi."""

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = max(interval_seconds, 0.5)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_mb = 0.0

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=self.interval_seconds * 2.0)
        return self._peak_mb

    def _run(self) -> None:
        while not self._stop_event.is_set():
            used_mb = _query_nvidia_smi_memory_mb()
            if used_mb is not None:
                self._peak_mb = max(self._peak_mb, used_mb)
            time.sleep(self.interval_seconds)


@dataclass
class CostSummary:
    param_count_per_model: Optional[int] = None
    search_total_seconds: float = 0.0
    retrain_total_seconds: float = 0.0
    inference_total_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    search_trials: int = 0
    ensemble_model_count: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def search_hours(self) -> float:
        return self.search_total_seconds / 3600.0

    @property
    def retrain_hours(self) -> float:
        return self.retrain_total_seconds / 3600.0

    @property
    def inference_minutes(self) -> float:
        return self.inference_total_seconds / 60.0

    @property
    def peak_vram_gb(self) -> float:
        return self.peak_vram_mb / 1024.0


class CostTracker:
    """Track wall-clock runtime and VRAM peaks for different experiment phases."""

    def __init__(self) -> None:
        self.summary = CostSummary()

    def set_param_count(self, value: int) -> None:
        self.summary.param_count_per_model = int(value)

    def add_time(self, phase: str, seconds: float) -> None:
        seconds = float(max(seconds, 0.0))
        if phase == "search":
            self.summary.search_total_seconds += seconds
        elif phase == "retrain":
            self.summary.retrain_total_seconds += seconds
        elif phase == "inference":
            self.summary.inference_total_seconds += seconds
        else:
            self.summary.notes.append("Unknown phase ignored: {0}".format(phase))

    def update_peak_vram(self, used_mb: float) -> None:
        self.summary.peak_vram_mb = max(self.summary.peak_vram_mb, float(used_mb))

    def set_counts(self, search_trials: int, ensemble_models: int) -> None:
        self.summary.search_trials = int(search_trials)
        self.summary.ensemble_model_count = int(ensemble_models)

    def add_note(self, note: str) -> None:
        self.summary.notes.append(note)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self.summary)
        payload["search_hours"] = self.summary.search_hours
        payload["retrain_hours"] = self.summary.retrain_hours
        payload["inference_minutes"] = self.summary.inference_minutes
        payload["peak_vram_gb"] = self.summary.peak_vram_gb
        return payload

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: Path) -> "CostTracker":
        tracker = CostTracker()
        if not path.exists():
            return tracker
        data = json.loads(path.read_text(encoding="utf-8"))
        tracker.summary.param_count_per_model = data.get("param_count_per_model")
        tracker.summary.search_total_seconds = float(data.get("search_total_seconds", 0.0))
        tracker.summary.retrain_total_seconds = float(data.get("retrain_total_seconds", 0.0))
        tracker.summary.inference_total_seconds = float(data.get("inference_total_seconds", 0.0))
        tracker.summary.peak_vram_mb = float(data.get("peak_vram_mb", 0.0))
        tracker.summary.search_trials = int(data.get("search_trials", 0))
        tracker.summary.ensemble_model_count = int(data.get("ensemble_model_count", 0))
        tracker.summary.notes = list(data.get("notes", []))
        return tracker


def count_lstm_parameters(
    input_size: int,
    hidden_size: int,
    output_size: int = 1,
    num_layers: int = 1,
) -> int:
    """Count parameters using a real PyTorch module when available.

    Falls back to analytical counting if torch is unavailable.
    """
    try:
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            ),
            nn.Linear(hidden_size, output_size),
        )

        total = 0
        for p in model.parameters():
            total += int(p.numel())
        return total
    except Exception:
        total = 0
        layer_input = input_size
        for _ in range(num_layers):
            total += 4 * hidden_size * layer_input
            total += 4 * hidden_size * hidden_size
            total += 8 * hidden_size
            layer_input = hidden_size
        total += hidden_size * output_size + output_size
        return int(total)


def render_cost_markdown(
    baseline: CostSummary,
    output_path: Path,
    hydromoe_params_placeholder: str = "[My Params]",
    hydromoe_search_placeholder: str = "[My Search Cost]",
    hydromoe_train_placeholder: str = "[My Total Train Time]",
    hydromoe_vram_placeholder: str = "[My Peak VRAM]",
    hydromoe_infer_placeholder: str = "[My Total Inference Time]",
) -> None:
    """Render the reviewer-requested cost table as Markdown."""
    baseline_param = (
        str(baseline.param_count_per_model)
        if baseline.param_count_per_model is not None
        else "N/A"
    )
    baseline_ensemble_count = baseline.ensemble_model_count or 100

    lines = [
        "| Metric Category | Specific Indicator | Top-10 Regional LSTM Ensemble (Baseline) | HydroMoE (Proposed) |",
        "| :--- | :--- | :--- | :--- |",
        "| **Input Constraint** | Dynamic Inputs | P, T, PET (No Static Attributes) | P, T, PET (No Static Attributes) |",
        "| **Model Complexity** | Parameters (per model) | {0} | {1} |".format(
            baseline_param, hydromoe_params_placeholder
        ),
        "| **Model Complexity** | Total Ensemble Size | {0} | 1 |".format(
            baseline_ensemble_count
        ),
        "| **Search Phase Cost** | Hyperparameter Trials | {0} Random Search Trials | 0 |".format(
            baseline.search_trials or 100
        ),
        "| **Search Phase Cost** | Search GPU Time | {0:.2f} GPU-Hours (wall-clock approximation) | {1} |".format(
            baseline.search_hours, hydromoe_search_placeholder
        ),
        "| **Training Phase** | Total Train Time | {0:.2f} Hours | {1} |".format(
            baseline.retrain_hours, hydromoe_train_placeholder
        ),
        "| **Training Phase** | Peak GPU VRAM Usage | {0:.2f} GB | {1} |".format(
            baseline.peak_vram_gb, hydromoe_vram_placeholder
        ),
        "| **Inference Phase** | Total Inference Time | {0:.2f} Minutes | {1} |".format(
            baseline.inference_minutes, hydromoe_infer_placeholder
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
