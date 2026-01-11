"""
Advanced Normalization Strategies for HydroMoE v2.0
解决归一化与还原不一致性问题的高级策略
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """归一化配置"""
    # 策略选择
    strategy: str = "time_window"  # "time_window", "sliding_window", "station_wise", "robust", "none"
    
    # 时间窗口归一化参数
    window_size: int = 365  # 时间窗口大小（天）
    window_stride: int = 30  # 窗口滑动步长（天）
    min_window_data: int = 100  # 窗口内最小数据点数
    
    # 滑动窗口参数
    lookback_days: int = 730  # 回望天数（2年）
    update_frequency: int = 90  # 更新频率（天）
    
    # 站点级归一化
    use_station_stats: bool = True  # 是否使用站点级统计
    station_min_samples: int = 1000  # 站点最小样本数
    
    # 鲁棒性参数
    outlier_threshold: float = 3.0  # 异常值阈值（标准差倍数）
    use_robust_scaler: bool = False  # 是否使用鲁棒标准化
    
    # 特征特定配置
    feature_specific: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.feature_specific is None:
            self.feature_specific = {
                'precip': {'log_transform': True, 'add_constant': 0.1},
                'temp': {'log_transform': False, 'seasonal_adjust': True},
                'pet': {'log_transform': True, 'add_constant': 0.01},
                'runoff': {'log_transform': True, 'add_constant': 0.001}
            }


class TimeWindowNormalizer:
    """时间窗口归一化器"""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.window_stats = {}  # 存储每个窗口的统计信息
        self.fitted = False
    
    def fit(self, data: pd.DataFrame, feature_cols: List[str], 
            target_col: str, time_col: str = 'date') -> 'TimeWindowNormalizer':
        """
        拟合时间窗口归一化器
        
        Args:
            data: 包含时间列的数据框
            feature_cols: 特征列名列表
            target_col: 目标列名
            time_col: 时间列名
        """
        logger.info("🔄 开始拟合时间窗口归一化器...")
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        # 按时间排序
        data = data.sort_values(time_col)
        
        # 创建时间窗口
        start_date = data[time_col].min()
        end_date = data[time_col].max()
        
        window_starts = pd.date_range(
            start=start_date, 
            end=end_date - pd.Timedelta(days=self.config.window_size),
            freq=pd.Timedelta(days=self.config.window_stride)
        )
        
        logger.info(f"创建了 {len(window_starts)} 个时间窗口")
        
        # 计算每个窗口的统计信息
        for i, window_start in enumerate(window_starts):
            window_end = window_start + pd.Timedelta(days=self.config.window_size)
            
            # 获取窗口内数据
            window_mask = (data[time_col] >= window_start) & (data[time_col] < window_end)
            window_data = data[window_mask]
            
            if len(window_data) < self.config.min_window_data:
                continue
            
            # 计算统计信息
            window_stats = {
                'start_date': window_start,
                'end_date': window_end,
                'feature_stats': {},
                'target_stats': {}
            }
            
            # 特征统计
            for col in feature_cols:
                window_stats['feature_stats'][col] = {
                    'mean': window_data[col].mean(),
                    'std': window_data[col].std(),
                    'median': window_data[col].median(),
                    'q25': window_data[col].quantile(0.25),
                    'q75': window_data[col].quantile(0.75)
                }
            
            # 目标变量统计
            window_stats['target_stats'] = {
                'mean': window_data[target_col].mean(),
                'std': window_data[target_col].std(),
                'median': window_data[target_col].median(),
                'q25': window_data[target_col].quantile(0.25),
                'q75': window_data[target_col].quantile(0.75)
            }
            
            self.window_stats[i] = window_stats
        
        self.fitted = True
        logger.info(f"✅ 时间窗口归一化器拟合完成，共 {len(self.window_stats)} 个有效窗口")
        return self
    
    def transform(self, data: pd.DataFrame, feature_cols: List[str], 
                  target_col: str, time_col: str = 'date') -> pd.DataFrame:
        """
        应用时间窗口归一化
        """
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用 fit() 方法")
        
        data = data.copy()
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        # 为每行数据找到对应的时间窗口
        for idx, row in data.iterrows():
            current_time = row[time_col]
            
            # 找到最合适的时间窗口
            best_window = self._find_best_window(current_time)
            
            if best_window is None:
                continue
            
            # 应用归一化
            for col in feature_cols:
                if col in best_window['feature_stats']:
                    stats = best_window['feature_stats'][col]
                    if stats['std'] > 1e-8:  # 避免除零
                        data.loc[idx, col] = (row[col] - stats['mean']) / stats['std']
            
            # 目标变量归一化
            if target_col in data.columns:
                stats = best_window['target_stats']
                if stats['std'] > 1e-8:
                    data.loc[idx, target_col] = (row[target_col] - stats['mean']) / stats['std']
        
        return data
    
    def inverse_transform(self, data: pd.DataFrame, target_col: str, 
                         time_col: str = 'date') -> pd.DataFrame:
        """
        反归一化目标变量
        """
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用 fit() 方法")
        
        data = data.copy()
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        # 为每行数据找到对应的时间窗口并反归一化
        for idx, row in data.iterrows():
            current_time = row[time_col]
            
            # 找到最合适的时间窗口
            best_window = self._find_best_window(current_time)
            
            if best_window is None:
                continue
            
            # 反归一化目标变量
            stats = best_window['target_stats']
            if stats['std'] > 1e-8:
                data.loc[idx, target_col] = row[target_col] * stats['std'] + stats['mean']
        
        return data
    
    def _find_best_window(self, target_time) -> Optional[Dict]:
        """找到最适合的时间窗口"""
        best_window = None
        min_distance = float('inf')
        
        for window in self.window_stats.values():
            window_center = window['start_date'] + (window['end_date'] - window['start_date']) / 2
            distance = abs((target_time - window_center).total_seconds())
            
            if distance < min_distance:
                min_distance = distance
                best_window = window
        
        return best_window


class StationWiseNormalizer:
    """站点级归一化器"""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.station_scalers = {}  # 每个站点的标准化器
        self.fitted = False
    
    def fit(self, data: pd.DataFrame, feature_cols: List[str], 
            target_col: str, station_col: str = 'station_id') -> 'StationWiseNormalizer':
        """
        拟合站点级归一化器
        """
        logger.info("🔄 开始拟合站点级归一化器...")
        
        unique_stations = data[station_col].unique()
        logger.info(f"发现 {len(unique_stations)} 个独特站点")
        
        for station in unique_stations:
            station_data = data[data[station_col] == station]
            
            if len(station_data) < self.config.station_min_samples:
                logger.warning(f"站点 {station} 样本数 ({len(station_data)}) 少于最小要求 ({self.config.station_min_samples})")
                continue
            
            # 为每个站点创建标准化器
            station_scalers = {}
            
            # 特征标准化器
            if self.config.use_robust_scaler:
                feature_scaler = RobustScaler()
                target_scaler = RobustScaler()
            else:
                feature_scaler = StandardScaler()
                target_scaler = StandardScaler()
            
            # 拟合标准化器
            feature_scaler.fit(station_data[feature_cols])
            target_scaler.fit(station_data[[target_col]])
            
            station_scalers['feature_scaler'] = feature_scaler
            station_scalers['target_scaler'] = target_scaler
            
            self.station_scalers[station] = station_scalers
        
        self.fitted = True
        logger.info(f"✅ 站点级归一化器拟合完成，覆盖 {len(self.station_scalers)} 个站点")
        return self
    
    def transform(self, data: pd.DataFrame, feature_cols: List[str], 
                  target_col: str, station_col: str = 'station_id') -> pd.DataFrame:
        """
        应用站点级归一化
        """
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用 fit() 方法")
        
        data = data.copy()
        
        for station in data[station_col].unique():
            if station not in self.station_scalers:
                logger.warning(f"站点 {station} 没有对应的标准化器，跳过归一化")
                continue
            
            station_mask = data[station_col] == station
            station_scalers = self.station_scalers[station]
            
            # 归一化特征
            data.loc[station_mask, feature_cols] = station_scalers['feature_scaler'].transform(
                data.loc[station_mask, feature_cols]
            )
            
            # 归一化目标变量
            if target_col in data.columns:
                data.loc[station_mask, [target_col]] = station_scalers['target_scaler'].transform(
                    data.loc[station_mask, [target_col]]
                )
        
        return data
    
    def inverse_transform(self, data: pd.DataFrame, target_col: str, 
                         station_col: str = 'station_id') -> pd.DataFrame:
        """
        反归一化目标变量
        """
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用 fit() 方法")
        
        data = data.copy()
        
        for station in data[station_col].unique():
            if station not in self.station_scalers:
                logger.warning(f"站点 {station} 没有对应的标准化器，跳过反归一化")
                continue
            
            station_mask = data[station_col] == station
            target_scaler = self.station_scalers[station]['target_scaler']
            
            # 反归一化目标变量
            data.loc[station_mask, [target_col]] = target_scaler.inverse_transform(
                data.loc[station_mask, [target_col]]
            )
        
        return data


class AdvancedNormalizer:
    """高级归一化器 - 统一接口"""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.normalizer = None
        self.global_scaler = None  # 备用全局标准化器
        self._initialize_normalizer()
    
    def _initialize_normalizer(self):
        """初始化具体的归一化器"""
        if self.config.strategy == "time_window":
            self.normalizer = TimeWindowNormalizer(self.config)
        elif self.config.strategy == "station_wise":
            self.normalizer = StationWiseNormalizer(self.config)
        elif self.config.strategy == "robust":
            self.global_scaler = RobustScaler()
        elif self.config.strategy == "none":
            self.normalizer = None
        else:
            logger.warning(f"未知的归一化策略: {self.config.strategy}，使用标准归一化")
            self.global_scaler = StandardScaler()
    
    def fit_transform(self, train_data: pd.DataFrame, feature_cols: List[str], 
                      target_col: str, **kwargs) -> pd.DataFrame:
        """
        拟合并转换训练数据
        """
        if self.config.strategy == "none":
            logger.info("🚫 跳过归一化")
            return train_data.copy()
        
        if self.normalizer is not None:
            # 使用高级归一化器
            self.normalizer.fit(train_data, feature_cols, target_col, **kwargs)
            return self.normalizer.transform(train_data, feature_cols, target_col, **kwargs)
        
        elif self.global_scaler is not None:
            # 使用全局标准化器
            data = train_data.copy()
            
            # 特征归一化
            data[feature_cols] = self.global_scaler.fit_transform(data[feature_cols])
            
            # 目标变量归一化
            if hasattr(self, 'target_scaler'):
                self.target_scaler = StandardScaler() if not self.config.use_robust_scaler else RobustScaler()
            else:
                self.target_scaler = StandardScaler() if not self.config.use_robust_scaler else RobustScaler()
            
            data[[target_col]] = self.target_scaler.fit_transform(data[[target_col]])
            
            return data
        
        return train_data.copy()
    
    def transform(self, data: pd.DataFrame, feature_cols: List[str], 
                  target_col: str, **kwargs) -> pd.DataFrame:
        """
        转换验证/测试数据
        """
        if self.config.strategy == "none":
            return data.copy()
        
        if self.normalizer is not None:
            return self.normalizer.transform(data, feature_cols, target_col, **kwargs)
        
        elif self.global_scaler is not None:
            data = data.copy()
            
            # 特征归一化
            data[feature_cols] = self.global_scaler.transform(data[feature_cols])
            
            # 目标变量归一化
            if target_col in data.columns and hasattr(self, 'target_scaler'):
                data[[target_col]] = self.target_scaler.transform(data[[target_col]])
            
            return data
        
        return data.copy()
    
    def inverse_transform_targets(self, data: pd.DataFrame, target_col: str, 
                                  **kwargs) -> pd.DataFrame:
        """
        反归一化目标变量
        """
        if self.config.strategy == "none":
            return data.copy()
        
        if self.normalizer is not None and hasattr(self.normalizer, 'inverse_transform'):
            return self.normalizer.inverse_transform(data, target_col, **kwargs)
        
        elif hasattr(self, 'target_scaler') and self.target_scaler is not None:
            data = data.copy()
            data[[target_col]] = self.target_scaler.inverse_transform(data[[target_col]])
            return data
        
        return data.copy()


def create_gradient_stable_normalizer(strategy: str = "station_wise") -> AdvancedNormalizer:
    """
    创建梯度稳定的归一化器 - 针对径流数据优化

    Args:
        strategy: 归一化策略，推荐 "station_wise" 或 "time_window"

    Returns:
        配置好的高级归一化器
    """
    config = NormalizationConfig(
        strategy=strategy,
        use_robust_scaler=True,  # 使用鲁棒标准化，减少异常值影响
        outlier_threshold=3.0,   # 放宽异常值阈值，保留更多极值信息
        window_size=730,         # 2年窗口，捕捉季节性
        station_min_samples=300, # 进一步降低站点最小样本要求
        feature_specific={
            'precip': {'log_transform': True, 'add_constant': 0.1},
            'temp': {'log_transform': False, 'seasonal_adjust': False},
            'pet': {'log_transform': True, 'add_constant': 0.01},
            'runoff': {'log_transform': True, 'add_constant': 0.01}  # 增加常数，更好处理小径流值
        }
    )

    return AdvancedNormalizer(config)


class HydroLogNormalizer:
    """
    专门针对径流数据的对数正态分布归一化器
    """

    def __init__(self, add_constant: float = 0.1, use_robust: bool = True):
        """
        初始化径流专用归一化器 - 数值稳定版本

        Args:
            add_constant: 对数变换前添加的常数，避免log(0)，增大以提高稳定性
            use_robust: 是否使用鲁棒标准化（中位数+MAD）
        """
        self.add_constant = add_constant  # 增大常数，提高数值稳定性
        self.use_robust = use_robust
        self.fitted = False

        # 存储统计参数
        self.log_mean = None
        self.log_std = None
        self.log_median = None
        self.log_mad = None  # Median Absolute Deviation

    def fit(self, runoff_data: np.ndarray) -> 'HydroLogNormalizer':
        """
        拟合归一化参数

        Args:
            runoff_data: 径流数据，形状为 (n_samples,)
        """
        # 确保数据为正值
        runoff_data = np.maximum(runoff_data, 0.0)

        # 对数变换
        log_data = np.log1p(runoff_data + self.add_constant)

        if self.use_robust:
            # 使用鲁棒统计量
            self.log_median = np.median(log_data)
            self.log_mad = np.median(np.abs(log_data - self.log_median))
            # 避免MAD为0的情况
            if self.log_mad < 1e-8:
                self.log_mad = np.std(log_data)
        else:
            # 使用标准统计量
            self.log_mean = np.mean(log_data)
            self.log_std = np.std(log_data)
            # 避免标准差为0的情况
            if self.log_std < 1e-8:
                self.log_std = 1.0

        self.fitted = True
        return self

    def transform(self, runoff_data: np.ndarray) -> np.ndarray:
        """
        应用归一化变换

        Args:
            runoff_data: 径流数据

        Returns:
            归一化后的数据
        """
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")

        # 确保数据为正值
        runoff_data = np.maximum(runoff_data, 0.0)

        # 对数变换
        log_data = np.log1p(runoff_data + self.add_constant)

        if self.use_robust:
            # 鲁棒标准化
            normalized = (log_data - self.log_median) / self.log_mad
        else:
            # 标准标准化
            normalized = (log_data - self.log_mean) / self.log_std

        return normalized

    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        反归一化变换

        Args:
            normalized_data: 归一化后的数据

        Returns:
            原始尺度的径流数据
        """
        if not self.fitted:
            raise ValueError("归一化器尚未拟合，请先调用fit()方法")

        if self.use_robust:
            # 反鲁棒标准化
            log_data = normalized_data * self.log_mad + self.log_median
        else:
            # 反标准标准化
            log_data = normalized_data * self.log_std + self.log_mean

        # 反对数变换
        runoff_data = np.expm1(log_data) - self.add_constant

        # 确保结果为非负
        return np.maximum(runoff_data, 0.0)

    def fit_transform(self, runoff_data: np.ndarray) -> np.ndarray:
        """
        拟合并变换数据
        """
        return self.fit(runoff_data).transform(runoff_data)


# 梯度稳定工具函数
def apply_gradient_clipping(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    应用梯度裁剪
    
    Args:
        model: PyTorch模型
        max_norm: 最大梯度范数
    
    Returns:
        裁剪前的梯度范数
    """
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm


def check_gradient_health(model: nn.Module) -> Dict[str, float]:
    """
    检查梯度健康状况
    
    Args:
        model: PyTorch模型
    
    Returns:
        梯度统计信息
    """
    total_norm = 0
    param_count = 0
    max_grad = 0
    min_grad = float('inf')
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += param.numel()
            
            max_grad = max(max_grad, param.grad.data.abs().max().item())
            min_grad = min(min_grad, param.grad.data.abs().min().item())
    
    total_norm = total_norm ** (1. / 2)
    
    return {
        'total_norm': total_norm,
        'average_norm': total_norm / max(param_count, 1),
        'max_gradient': max_grad,
        'min_gradient': min_grad if min_grad != float('inf') else 0,
        'param_count': param_count
    }