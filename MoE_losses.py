"""
MoE损失函数模块 - 包含所有自定义损失函数
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union


def compute_all_metrics(y_true, y_pred):
    """计算所有评估指标"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 基础指标
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 相关系数
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # KGE指标计算
    bias = np.mean(y_pred) - np.mean(y_true)
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else 0
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) > 0 else 0
    
    kge = 1 - np.sqrt((correlation - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation,
        'kge': kge,
        'bias': bias,
        'alpha': alpha,
        'beta': beta
    }


def format_metrics_string(metrics):
    """格式化指标字符串"""
    return f"R²: {metrics['r2']:.4f}, KGE: {metrics['kge']:.4f}, RMSE: {metrics['rmse']:.4f}"


class HydroKGELoss(nn.Module):
    """
    水文KGE损失函数，结合均方误差、绝对误差和负载均衡
    """
    
    def __init__(self, mse_weight=0.8, l1_weight=0.2, load_balance_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.load_balance_weight = load_balance_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, predictions, targets, gate_info=None):
        """
        计算混合损失
        
        Args:
            predictions: 模型预测值 [batch_size, ...]
            targets: 真实目标值 [batch_size, ...]
            gate_info: 门控信息，用于负载均衡损失
        """
        # 基础回归损失
        mse_loss = self.mse_loss(predictions, targets)
        l1_loss = self.l1_loss(predictions, targets)
        
        # 组合回归损失
        regression_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        
        # 负载均衡损失
        load_balance_loss = 0.0
        if gate_info is not None and self.load_balance_weight > 0:
            load_balance_loss = self._compute_load_balance_loss(gate_info)
        
        # 总损失
        total_loss = regression_loss + self.load_balance_weight * load_balance_loss
        
        return total_loss
    
    def _compute_load_balance_loss(self, gate_info):
        """计算负载均衡损失"""
        total_loss = 0.0
        count = 0
        
        for module_name, module_gate_info in gate_info.items():
            if 'gate_weights' in module_gate_info:
                gate_weights = module_gate_info['gate_weights']  # [batch_size, num_experts]
                
                # 如启用，计算使用频率与均匀分布的差异（默认不启用）
                expert_usage = torch.mean(gate_weights, dim=0)
                target_usage = 1.0 / len(expert_usage)
                balance_loss = torch.mean((expert_usage - target_usage) ** 2)
                total_loss += balance_loss
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=gate_weights.device)


class StationR2Loss(nn.Module):
    """
    基于站点R²的损失函数
    """
    
    def __init__(self, min_r2: float = -1.0, max_r2: float = 1.0, min_samples_per_station: int = 5):
        super().__init__()
        self.min_r2 = min_r2
        self.max_r2 = max_r2
        self.min_samples_per_station = min_samples_per_station
    
    def forward(self, predictions, targets, station_ids=None):
        """
        计算站点级R²损失
        
        Args:
            predictions: 预测值 [batch_size]
            targets: 真实值 [batch_size]
            station_ids: 站点ID [batch_size], 可选
        """
        # 统一展平到1D，兼容 [B] 与 [B,1] 情况
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        if station_ids is None:
            station_ids = torch.zeros_like(predictions, dtype=torch.long)
        else:
            station_ids = station_ids.view(-1)
        # 对齐长度（取最小长度以防不一致）
        n = min(predictions.shape[0], targets.shape[0], station_ids.shape[0])
        predictions = predictions[:n]
        targets = targets[:n]
        station_ids = station_ids[:n]
        
        unique_stations = torch.unique(station_ids)
        total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        for station_id in unique_stations:
            # 获取该站点的数据
            mask = station_ids == station_id
            if mask.ndim != 1:
                mask = mask.view(-1)
            if mask.sum() == 0:
                continue
            station_preds = predictions[mask]
            station_targets = targets[mask]
            
            if len(station_preds) < self.min_samples_per_station:
                # 样本太少时，使用MSE作为稳定回退，避免R²数值不稳定
                station_loss = torch.mean((station_targets - station_preds) ** 2)
                total_loss = total_loss + station_loss
                continue
                
            # 计算R²
            ss_res = torch.sum((station_targets - station_preds) ** 2)
            ss_tot = torch.sum((station_targets - torch.mean(station_targets)) ** 2)
            
            if ss_tot > 1e-8:
                r2 = 1 - (ss_res / ss_tot)
                # 限制R²范围，避免极端值
                r2 = torch.clamp(r2, self.min_r2, self.max_r2)
                # 将R²转换为损失（1-R²，R²越高损失越低）
                station_loss = 1 - r2
            else:
                # 当targets方差过小时，使用MSE损失作为回退（保持梯度）
                station_loss = torch.mean((station_targets - station_preds) ** 2)
            
            total_loss = total_loss + station_loss
        
        if len(unique_stations) > 0:
            return total_loss / len(unique_stations)
        else:
            # 确保返回可微分的张量 - 使用预测值的MSE作为回退
            return torch.mean((predictions - targets) ** 2)


class CombinedHydroLoss(nn.Module):
    """
    组合水文损失函数，结合MSE、站点R²和负载均衡
    """
    
    def __init__(self, mse_weight=0.5, r2_weight=0.4, load_balance_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.r2_weight = r2_weight
        self.load_balance_weight = load_balance_weight
        
        self.mse_loss = nn.MSELoss()
        self.r2_loss = StationR2Loss()
        self.kge_loss = HydroKGELoss(load_balance_weight=0.0)  # 不重复计算负载均衡
    
    def forward(self, predictions, targets, gate_info=None, station_ids=None):
        """
        计算组合损失
        """
        # MSE损失
        mse_loss = self.mse_loss(predictions, targets)
        
        # 站点R²损失
        r2_loss = self.r2_loss(predictions, targets, station_ids)
        
        # 负载均衡损失
        load_balance_loss = 0.0
        if gate_info is not None and self.load_balance_weight > 0:
            load_balance_loss = self.kge_loss._compute_load_balance_loss(gate_info)
        
        # 组合损失
        total_loss = (self.mse_weight * mse_loss + 
                     self.r2_weight * r2_loss + 
                     self.load_balance_weight * load_balance_loss)
        
        return total_loss


class WeightedHydroLoss(nn.Module):
    """
    加权水文损失函数 - 重视高径流事件
    """

    def __init__(self, base_loss='mse', high_flow_threshold=2.0, high_flow_weight=3.0,
                 extreme_flow_threshold=4.0, extreme_flow_weight=5.0):
        """
        初始化加权损失函数

        Args:
            base_loss: 基础损失函数类型 ('mse', 'huber')
            high_flow_threshold: 高径流阈值（标准化后）
            high_flow_weight: 高径流权重
            extreme_flow_threshold: 极端径流阈值（标准化后）
            extreme_flow_weight: 极端径流权重
        """
        super().__init__()
        self.high_flow_threshold = high_flow_threshold
        self.high_flow_weight = high_flow_weight
        self.extreme_flow_threshold = extreme_flow_threshold
        self.extreme_flow_weight = extreme_flow_weight

        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss(reduction='none')
        elif base_loss == 'huber':
            self.base_criterion = nn.SmoothL1Loss(reduction='none', beta=1.0)
        else:
            raise ValueError(f"不支持的基础损失函数: {base_loss}")

    def forward(self, predictions, targets, gate_info=None):
        """
        计算加权损失 - 优化版本
        """
        # 🚀 优化：简化有效性检查，减少重复操作
        if not (torch.isfinite(predictions).all() and torch.isfinite(targets).all()):
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 计算基础损失
        base_loss = self.base_criterion(predictions, targets)

        # 🚀 优化：提前检查并返回
        if not torch.isfinite(base_loss).all():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 🚀 优化：简化权重计算，使用inplace操作
        target_std = torch.std(targets)
        target_mean = torch.mean(targets)
        
        high_threshold = target_mean + target_std
        extreme_threshold = target_mean + 2.0 * target_std

        # 🚀 优化：直接计算加权损失，避免创建weights张量
        high_mask = targets > high_threshold
        extreme_mask = targets > extreme_threshold
        
        weighted_loss = base_loss.clone()
        weighted_loss[high_mask] *= 1.5
        weighted_loss[extreme_mask] *= 2.0

        # 限制损失范围
        final_loss = torch.clamp(weighted_loss.mean(), max=100.0)

        return final_loss


class AdaptiveHydroLoss(nn.Module):
    """
    自适应水文损失函数 - 根据径流量级动态调整权重
    """

    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        """
        初始化自适应损失函数

        Args:
            alpha: MSE损失权重
            beta: 相对误差损失权重
            gamma: 峰值保持损失权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets, gate_info=None):
        """
        计算自适应损失 - 数值稳定版本
        """
        # 检查输入有效性
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 1. MSE损失
        mse_loss = self.mse_loss(predictions, targets)

        # 检查MSE损失
        if torch.isnan(mse_loss).any() or torch.isinf(mse_loss).any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 2. 相对误差损失（对小径流值更敏感）- 更保守的计算
        epsilon = 1e-3  # 增大epsilon，提高数值稳定性
        denominator = torch.abs(targets) + epsilon
        relative_error = torch.abs(predictions - targets) / denominator
        relative_error = torch.clamp(relative_error, max=10.0)  # 限制相对误差
        relative_loss = relative_error ** 2

        # 3. 峰值保持损失（对高径流值更敏感）- 更保守的策略
        try:
            peak_threshold = torch.quantile(targets, 0.95)  # 改为前5%，更保守
            peak_mask = targets > peak_threshold
            peak_loss = torch.zeros_like(mse_loss)
            if peak_mask.any():
                peak_loss[peak_mask] = mse_loss[peak_mask] * 1.2  # 降低峰值权重
        except:
            peak_loss = torch.zeros_like(mse_loss)

        # 组合损失 - 降低各分量权重
        total_loss = (self.alpha * mse_loss +
                     self.beta * 0.1 * relative_loss +  # 大幅降低相对误差权重
                     self.gamma * 0.1 * peak_loss)      # 大幅降低峰值权重

        # 适度限制损失范围，不要太严格
        final_loss = total_loss.mean()
        final_loss = torch.clamp(final_loss, max=100.0)  # 放宽限制

        return final_loss


class ExpertSpecializationLoss(nn.Module):
    """
    专家专业化损失 - 鼓励专家差异化，避免趋同
    """
    
    def __init__(self, diversity_weight: float = 0.01, min_specialization: float = 0.6):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.min_specialization = min_specialization
    
    def forward(self, gate_info: Dict) -> torch.Tensor:
        """
        计算专家专业化损失
        
        Args:
            gate_info: 门控信息，包含各模块的门控权重
        
        Returns:
            专业化损失（越低表示专家越专业化）
        """
        total_diversity_loss = torch.tensor(0.0)
        count = 0
        
        if gate_info and isinstance(gate_info, dict):
            module_gates = gate_info.get('module_gates', {})
            
            for module_name, module_info in module_gates.items():
                if 'effective_gate' in module_info:
                    weights = module_info['effective_gate']  # [batch_size, num_experts]
                    
                    # 计算专家专业化程度（权重方差）
                    # 权重方差越大，说明专家越专业化
                    weight_variance = torch.var(weights, dim=-1)  # [batch_size]
                    
                    # 鼓励高方差（专业化），惩罚低方差（趋同）
                    # 目标：每个样本至少有一个专家权重 > min_specialization
                    max_weight_per_sample = torch.max(weights, dim=-1)[0]  # [batch_size]
                    specialization_penalty = F.relu(self.min_specialization - max_weight_per_sample)
                    
                    diversity_loss = specialization_penalty.mean()
                    total_diversity_loss += diversity_loss
                    count += 1
        
        return total_diversity_loss / max(count, 1) * self.diversity_weight


class EnhancedCombinedLoss(nn.Module):
    """
    增强组合损失函数 - 集成多种损失策略
    """

    def __init__(self, mse_weight=0.4, kge_weight=0.3, weighted_weight=0.2, adaptive_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.kge_weight = kge_weight
        self.weighted_weight = weighted_weight
        self.adaptive_weight = adaptive_weight

        self.mse_loss = nn.MSELoss()
        self.kge_loss = HydroKGELoss(load_balance_weight=0.0)
        self.weighted_loss = WeightedHydroLoss()
        self.adaptive_loss = AdaptiveHydroLoss()
        
        # 🚀 添加专家专业化损失
        self.specialization_loss = ExpertSpecializationLoss()

    def forward(self, predictions, targets, gate_info=None, station_ids=None):
        """
        计算增强组合损失 - 数值稳定版本
        """
        # 检查输入有效性
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 各种损失分量 - 添加异常处理
        try:
            mse_loss = self.mse_loss(predictions, targets)
            if torch.isnan(mse_loss) or torch.isinf(mse_loss):
                mse_loss = torch.tensor(0.0, device=predictions.device)
        except:
            mse_loss = torch.tensor(0.0, device=predictions.device)

        try:
            kge_loss = self.kge_loss(predictions, targets, gate_info)
            if torch.isnan(kge_loss) or torch.isinf(kge_loss):
                kge_loss = torch.tensor(0.0, device=predictions.device)
        except:
            kge_loss = torch.tensor(0.0, device=predictions.device)

        try:
            weighted_loss = self.weighted_loss(predictions, targets)
            if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                weighted_loss = torch.tensor(0.0, device=predictions.device)
        except:
            weighted_loss = torch.tensor(0.0, device=predictions.device)

        try:
            adaptive_loss = self.adaptive_loss(predictions, targets)
            if torch.isnan(adaptive_loss) or torch.isinf(adaptive_loss):
                adaptive_loss = torch.tensor(0.0, device=predictions.device)
        except:
            adaptive_loss = torch.tensor(0.0, device=predictions.device)

        # 🚀 计算专家专业化损失
        try:
            specialization_loss = self.specialization_loss(gate_info)
            if torch.isnan(specialization_loss) or torch.isinf(specialization_loss):
                specialization_loss = torch.tensor(0.0, device=predictions.device)
        except:
            specialization_loss = torch.tensor(0.0, device=predictions.device)

        # 组合损失 - 更保守的权重 + 专业化鼓励
        total_loss = (0.6 * mse_loss +           # 提高MSE权重，更稳定
                     0.2 * kge_loss +            # 降低KGE权重
                     0.1 * weighted_loss +       # 降低加权损失权重
                     0.1 * adaptive_loss -       # 降低自适应损失权重
                     0.01 * specialization_loss) # 🚀 减去专业化损失（鼓励专业化）

        # 适度限制损失范围
        total_loss = torch.clamp(total_loss, max=50.0)  # 放宽限制

        return total_loss


# 损失函数工厂
def create_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    创建损失函数

    Args:
        loss_type: 损失函数类型 ('mse', 'kge', 'station_r2', 'combined', 'weighted', 'adaptive', 'enhanced')
        **kwargs: 损失函数参数
    """
    if loss_type.lower() == 'mse':
        return nn.MSELoss()
    elif loss_type.lower() == 'kge':
        return HydroKGELoss(**kwargs)
    elif loss_type.lower() == 'station_r2':
        return StationR2Loss(**kwargs)
    elif loss_type.lower() == 'combined':
        return CombinedHydroLoss(**kwargs)
    elif loss_type.lower() == 'weighted':
        return WeightedHydroLoss(**kwargs)
    elif loss_type.lower() == 'adaptive':
        return AdaptiveHydroLoss(**kwargs)
    elif loss_type.lower() == 'enhanced':
        return EnhancedCombinedLoss(**kwargs)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")
