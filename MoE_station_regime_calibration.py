"""
站点×水期校准包装器

用途：
- 不改动原有 `HybridHydroMoEModel` 的实现与体积，新增一个轻量包装器对 runoff 做站点×水期的仿射校准。
- 端到端训练：校准层随主模型一起学习；默认近似恒等映射，稳定不伤好站。

输入依赖：batch 需包含 `station_idx`（Long）、可选 `lon`/`lat`（[B,1]）。
从基础模型输出获取 `regime_weights`（若无则降级为均匀分布）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any


class CalibratedHybridModel(nn.Module):
    def __init__(self, base_model: nn.Module,
                 max_stations: int = 10000,
                 station_emb_dim: int = 16,
                 hidden_dim: int = 64):
        super().__init__()
        self.base = base_model
        self.max_stations = max_stations
        self.station_emb_dim = station_emb_dim
        self.hidden_dim = hidden_dim

        # 站点嵌入
        self.station_emb = nn.Embedding(self.max_stations, self.station_emb_dim)

        # 校准 MLP：输入 [station_emb(16), regime(3), lon(1), lat(1)] → [scale_raw, shift_raw]
        calib_in_dim = self.station_emb_dim + 3 + 2
        self.calib_mlp = nn.Sequential(
            nn.Linear(calib_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )

        # 开关：需要时可关闭校准
        self.use_station_regime_calibration = True

    def forward(self, batch: Dict[str, Any], return_gate_info: bool = False) -> Dict[str, torch.Tensor]:
        # 先跑基础模型
        base_out = self.base(batch, return_gate_info=True)

        runoff_base = base_out.get('runoff')  # [B]
        if runoff_base is None:
            raise RuntimeError("Base model output missing 'runoff'")

        # 统一形状为 [B,1]
        if runoff_base.dim() == 1:
            runoff_base_2d = runoff_base.unsqueeze(-1)
        elif runoff_base.dim() == 2 and runoff_base.shape[1] == 1:
            runoff_base_2d = runoff_base
        else:
            runoff_base_2d = runoff_base.unsqueeze(-1)

        if not self.use_station_regime_calibration:
            # 仅透传并返回 base 结果
            out = dict(base_out)
            out['runoff_base'] = runoff_base_2d.squeeze(-1)
            return out

        device = runoff_base_2d.device
        batch_size = runoff_base_2d.shape[0]

        # station_idx 处理
        sid = batch.get('station_idx', None)
        if isinstance(sid, torch.Tensor):
            sid_flat = sid.view(-1).to(device=device, dtype=torch.long)
            sid_safe = torch.clamp(sid_flat, 0, self.max_stations - 1)
        else:
            sid_safe = torch.zeros(batch_size, dtype=torch.long, device=device)

        station_vec = self.station_emb(sid_safe)  # [B, emb]

        # regime 权重
        weights = base_out.get('regime_weights', None)
        if isinstance(weights, torch.Tensor) and weights.dim() == 2 and weights.shape[1] == 3:
            reg = weights.to(device=device)
        else:
            reg = torch.full((batch_size, 3), 1.0 / 3.0, device=device)

        # lon/lat（可选）
        lon = batch.get('lon', None)
        lat = batch.get('lat', None)
        if isinstance(lon, torch.Tensor) and lon.dim() == 2 and lon.shape[1] == 1:
            lon_in = lon.to(device=device)
        else:
            lon_in = torch.zeros(batch_size, 1, device=device)
        if isinstance(lat, torch.Tensor) and lat.dim() == 2 and lat.shape[1] == 1:
            lat_in = lat.to(device=device)
        else:
            lat_in = torch.zeros(batch_size, 1, device=device)

        calib_in = torch.cat([station_vec, reg, lon_in, lat_in], dim=-1)  # [B, emb+3+2]
        scale_raw, shift_raw = torch.chunk(self.calib_mlp(calib_in), chunks=2, dim=-1)  # [B,1],[B,1]

        # 约束：保持小幅、稳定的校准（初始近似恒等）
        scale = 1.0 + 0.1 * torch.tanh(scale_raw)
        shift = 0.1 * torch.tanh(shift_raw)
        runoff_calibrated = torch.clamp(scale * runoff_base_2d + shift, min=0.0)

        # 组织输出：更新 runoff；保留 base 与 gate_info
        out = dict(base_out)
        out['runoff_base'] = runoff_base_2d.squeeze(-1)
        out['runoff'] = runoff_calibrated.squeeze(-1)
        if return_gate_info:
            # 透传 base 的 gate_info
            pass
        # 额外暴露校准信息（便于调试）
        out['calibration_scale'] = scale.squeeze(-1)
        out['calibration_shift'] = shift.squeeze(-1)
        return out


def wrap_with_calibration(base_model: nn.Module,
                          max_stations: int = 10000,
                          station_emb_dim: int = 16,
                          hidden_dim: int = 64) -> CalibratedHybridModel:
    """便捷封装函数。"""
    return CalibratedHybridModel(base_model,
                                 max_stations=max_stations,
                                 station_emb_dim=station_emb_dim,
                                 hidden_dim=hidden_dim)


