"""按 station_performance_real_runoff.csv 中 R²<0 的站点重新训练的脚本。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from MoE_config import get_default_config
from MoE_data_config import FixedDataConfig
import torch
from MoE_data_loader import clear_data_cache, create_fixed_data_loaders
from MoE_hybrid_model import create_hybrid_hydro_moe_model
from MoE_trainer_enhanced import enhanced_training_loop
from MoE_evaluator_simple import evaluate_enhanced_model, load_best_model_if_exists


CSV_PATH = Path("outputs/enhanced_real_runoff_predictions/station_performance_real_runoff.csv")
OUTPUT_PREFIX = "outputs/lowR2"


def load_low_r2_stations(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path)
    return df.loc[df["R2"] < 0, "station_id"].astype(str).tolist()


def build_data_config(stations: List[str]) -> FixedDataConfig:
    cfg = FixedDataConfig()
    cfg.filter_station_ids = stations
    cfg.use_all_stations = False
    cfg.quick_test = False
    cfg.csv_path = cfg.csv_path
    return cfg


def train_for_low_r2() -> Dict[str, float]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到统计文件: {CSV_PATH}")

    low_stations = load_low_r2_stations(CSV_PATH)
    if not low_stations:
        print("✅ 没有 R² < 0 的站点，跳过重新训练")
        return {}

    print(f"🎯 低 R² 站点数量: {len(low_stations)}")

    config = get_default_config()
    data_config = build_data_config(low_stations)

    clear_data_cache()
    train_loader, val_loader, test_loader = create_fixed_data_loaders(
        data_config,
        batch_size=config.training.batch_size,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        prefetch_factor=config.system.prefetch_factor
    )

    model = create_hybrid_hydro_moe_model(config)
    load_best_model_if_exists(model)

    device = torch.device(
        config.system.device
        if config.system.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    model.to(device)

    enhanced_training_loop(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=config.training.epochs,
        patience=config.training.early_stopping_patience
    )

    metrics = evaluate_enhanced_model(model, test_loader, device, output_prefix='lowR2')

    output_dir = Path(OUTPUT_PREFIX)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "lowR2_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 低 R² 站点指标已保存: {metrics_path}")

    return metrics


if __name__ == "__main__":
    metrics = train_for_low_r2()
    if metrics:
        print("最终指标:", metrics)

