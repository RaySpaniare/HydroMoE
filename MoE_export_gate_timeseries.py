"""
导出测试集逐日门控权重与径流时间序列

运行方式：在VS Code中直接运行本文件。
输出：`outputs/gate_timeseries/test_gate_timeseries.csv`
"""

import os
from pathlib import Path
from typing import Dict, Any

import torch
import pandas as pd

from MoE_config import get_default_config
from MoE_data_loader import FixedDataConfig, create_fixed_data_loaders
from MoE_hybrid_model import create_hybrid_hydro_moe_model
from MoE_evaluator_simple import load_best_model_if_exists


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def export_gate_timeseries():
    config = get_default_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    # 仅构建测试集数据加载器
    data_cfg = FixedDataConfig()
    _, _, test_loader = create_fixed_data_loaders(
        data_cfg,
        batch_size=config.training.batch_size,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        prefetch_factor=config.system.prefetch_factor,
    )

    model = create_hybrid_hydro_moe_model(config)

    checkpoint_path = Path("outputs/enhanced_hydromoe_best.pth")
    if checkpoint_path.exists():
        print(f"🔄 尝试加载模型: {checkpoint_path}")
        load_best_model_if_exists(model, path=str(checkpoint_path))
    else:
        print("⚠️ 未找到预训练权重，将使用当前模型参数")

    model.to(device)
    model.eval()

    dataset = test_loader.dataset
    target_scaler = None
    if hasattr(dataset, "scalers") and dataset.scalers:
        target_scaler = dataset.scalers.get("target_scaler", None)

    records = []

    with torch.no_grad():
        for batch in test_loader:
            batch_device = _move_batch_to_device(batch, device)
            outputs = model(batch_device, return_gate_info=True)

            predictions = outputs["runoff"].detach().cpu().numpy().reshape(-1)
            targets = batch_device["targets"].detach().cpu().numpy().reshape(-1)

            if target_scaler is not None:
                predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                targets = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

            gate_info = outputs.get("gate_info", {})
            module_gates = gate_info.get("module_gates", {})
            regime_weights = outputs.get("regime_weights", None)
            alpha_weights = outputs.get("alpha_weights", None)
            available_water = outputs.get("available_water", None)

            station_ids = batch.get("station_id", [])
            end_dates = batch.get("end_date", [])

            batch_size = len(predictions)

            # 预提取并转为CPU，避免重复调用
            module_weight_cache: Dict[str, torch.Tensor] = {}
            for module_name, info in module_gates.items():
                eff_gate = info.get("effective_gate", None)
                if eff_gate is not None:
                    module_weight_cache[module_name] = eff_gate.detach().cpu()

            regime_cache = None
            if regime_weights is not None:
                regime_cache = regime_weights.detach().cpu()

            alpha_cache = None
            if alpha_weights is not None:
                alpha_cache = alpha_weights.detach().cpu()

            available_cache = None
            if available_water is not None:
                available_cache = available_water.detach().cpu().reshape(-1)

            for idx in range(batch_size):
                row: Dict[str, Any] = {
                    "station_id": station_ids[idx] if idx < len(station_ids) else None,
                    "date": end_dates[idx] if idx < len(end_dates) else None,
                    "predicted_runoff": float(predictions[idx]),
                    "actual_runoff": float(targets[idx]),
                }

                for module_name in ["snow", "runoff", "et", "drainage"]:
                    weights_tensor = module_weight_cache.get(module_name, None)
                    if weights_tensor is not None and idx < weights_tensor.shape[0]:
                        weights_list = weights_tensor[idx].tolist()
                        if len(weights_list) >= 2:
                            row[f"{module_name}_pbm_weight"] = float(weights_list[0])
                            row[f"{module_name}_nn_weight"] = float(weights_list[1])
                        elif len(weights_list) == 1:
                            row[f"{module_name}_pbm_weight"] = float(weights_list[0])
                            row[f"{module_name}_nn_weight"] = None
                    else:
                        row[f"{module_name}_pbm_weight"] = None
                        row[f"{module_name}_nn_weight"] = None

                if regime_cache is not None and idx < regime_cache.shape[0]:
                    regime_list = regime_cache[idx].tolist()
                    if len(regime_list) == 3:
                        row["regime_low_weight"] = float(regime_list[0])
                        row["regime_mid_weight"] = float(regime_list[1])
                        row["regime_high_weight"] = float(regime_list[2])
                else:
                    row["regime_low_weight"] = None
                    row["regime_mid_weight"] = None
                    row["regime_high_weight"] = None

                if alpha_cache is not None and idx < alpha_cache.shape[0]:
                    alpha_list = alpha_cache[idx].tolist()
                    if len(alpha_list) == 2:
                        row["alpha_quickflow"] = float(alpha_list[0])
                        row["alpha_baseflow"] = float(alpha_list[1])
                else:
                    row["alpha_quickflow"] = None
                    row["alpha_baseflow"] = None

                if available_cache is not None and idx < len(available_cache):
                    row["available_water"] = float(available_cache[idx])
                else:
                    row["available_water"] = None

                records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values(["station_id", "date"], inplace=True)

    output_dir = Path("outputs/gate_timeseries")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_gate_timeseries.csv"

    df.to_csv(output_path, index=False)
    print(f"✅ 导出完成，共 {len(df)} 条记录")
    print(f"📄 输出路径: {output_path.resolve()}")


if __name__ == "__main__":
    export_gate_timeseries()

