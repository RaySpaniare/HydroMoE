from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class SplitConfig:
    train_start: datetime = datetime(1980, 1, 1)
    train_end: datetime = datetime(1999, 12, 31)
    val_start: datetime = datetime(2000, 1, 1)
    val_end: datetime = datetime(2007, 12, 31)
    test_start: datetime = datetime(2008, 1, 1)
    test_end: datetime = datetime(2014, 9, 30)
    seq_length: int = 96

    @property
    def test_lookback_start(self) -> datetime:
        # Reserve seq_length - 1 days before the official test start for warmup.
        return self.test_start - timedelta(days=self.seq_length - 1)


DYNAMIC_INPUTS = ["pet", "precip", "temp"]
TARGET_VARIABLES = ["runoff"]
STATIC_ATTRIBUTES: list = []

DEFAULT_LONG_TABLE_PARQUET = Path(
    r"F:\python项目\Science科研项目\PBM+ML\LSTM基线训练\特征合并长表.parquet"
)


def nh_date(dt: datetime) -> str:
    """Return date string in the format expected by NeuralHydrology."""
    return dt.strftime("%d/%m/%Y")


def get_paths(project_root: Optional[Path] = None) -> Dict[str, Path]:
    if project_root is None:
        project_root = Path(__file__).resolve().parent

    paths = {
        "project_root": project_root,
        "prepared_data_root": project_root / "prepared_generic_dataset",
        "prepared_time_series_root": project_root / "prepared_generic_dataset" / "time_series",
        "prepared_attributes_root": project_root / "prepared_generic_dataset" / "attributes",
        "basin_list_root": project_root / "basin_lists",
        "configs_root": project_root / "configs",
        "search_configs_root": project_root / "configs" / "random_search",
        "ensemble_configs_root": project_root / "configs" / "ensemble_retrain",
        "runs_root": project_root / "runs",
        "search_runs_root": project_root / "runs" / "random_search",
        "ensemble_runs_root": project_root / "runs" / "ensemble_retrain",
        "artifacts_root": project_root / "artifacts",
        "reports_root": project_root / "reports",
        "cost_json": project_root / "reports" / "cost_summary.json",
        "cost_markdown": project_root / "reports" / "cost_comparison_table.md",
        "search_metrics_csv": project_root / "artifacts" / "search_validation_metrics.csv",
        "top10_metrics_csv": project_root / "artifacts" / "top10_validation_metrics.csv",
        "ensemble_epoch_losses_csv": project_root / "artifacts" / "ensemble_epoch_losses.csv",
        "ensemble_member_predictions": project_root / "artifacts" / "ensemble_member_predictions.parquet",
        "ensemble_median_predictions": project_root / "artifacts" / "ensemble_median_predictions.parquet",
        "loss_plot_jpg": project_root / "reports" / "loss_validation_test.jpg",
        "loss_plot_pdf": project_root / "reports" / "loss_validation_test.pdf",
    }

    for key, path in paths.items():
        if key.endswith("_root"):
            path.mkdir(parents=True, exist_ok=True)
    paths["prepared_time_series_root"].mkdir(parents=True, exist_ok=True)
    paths["prepared_attributes_root"].mkdir(parents=True, exist_ok=True)
    return paths
