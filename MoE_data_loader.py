"""
修复版本的数据加载器 - 正确处理数据标准化，包含全局缓存优化
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
import time
import gc  # 🔥 添加垃圾回收模块

# 导入特征工程模块
try:
    from MoE_feature_engineering import HydroFeatureEngineer, AdaptiveFeatureSelector
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    print("警告：特征工程模块不可用，将使用基础特征")

from MoE_data_utils import (
    build_cache_path,
    read_table_auto,
    read_from_cache,
    write_cache,
    has_parquet_support,
)


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🚀 增强全局数据缓存，避免重复加载大文件和重复序列创建
_GLOBAL_DATA_CACHE = {
    'raw_data': None,           # 原始完整数据
    'filtered_data': None,      # 筛选后数据（快速测试模式）
    'cache_params': None,       # 缓存参数（文件路径、快速测试配置等）
    'load_time': None,          # 加载时间戳
    # 🚀 新增：序列缓存
    'sequences_cache': {},      # 分split缓存序列：{'train': [...], 'val': [...], 'test': [...]}
    'scalers_cache': None,      # 标准化器缓存
    'grouped_data_cache': None, # 分组数据缓存（按站点分组的结果）
    'sequence_cache_params': None  # 序列缓存参数
}


@dataclass
class FixedDataConfig:
    """修复版数据配置"""
    # 文件路径
    csv_path: str = r"D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理\特征合并长表.csv"
    
    # 序列配置
    sequence_length: int = 96  # 序列长度（时间步）
    sequence_stride: int = 16  # 序列滑动步长
    
    # 特征列
    feature_cols: List[str] = None
    target_col: str = "runoff"
    
    # 时间划分 - 使用具体日期而不是比例
    train_start: str = '1980-01-01'
    train_end: str = '1999-12-31'
    val_start: str = '2000-01-01'
    val_end: str = '2007-12-31'
    test_start: str = '2008-01-01'
    test_end: str = '2014-09-30'
    
    # 数据分割（保留作为备选方案）
    use_date_split: bool = True  # 是否使用具体日期分割
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 数据处理
    normalize_features: bool = True
    normalize_targets: bool = True
    
    # 全站点训练配置
    use_all_stations: bool = True   # 启用全部50个站点训练
    quick_test: bool = False
    quick_test_stations: int = 10   # 快速测试时的站点数
    
    # 站点分批处理规模（创建序列时一次处理的站点数）
    station_batch_size: int = 100
    
    # 🚀 性能优化参数
    use_sequence_cache: bool = True      # 是否使用序列缓存
    parallel_sequence_creation: bool = False  # 是否启用并行序列创建（实验性）
    max_sequence_workers: int = 4        # 并行创建序列的最大工作进程数

    # 可选：仅训练指定站点（用于读取上一轮低R²站点加速训练）
    filter_station_ids: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.feature_cols is None:
            # 根据架构设计：输入是蒸散发、降水、温度，输出是径流
            self.feature_cols = ["pet", "precip", "temp"]
        
        # 验证分割比例
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6


class FixedHydroDataset(Dataset):
    """修复版水文数据集类 - 正确处理标准化"""
    
    def __init__(self, config: FixedDataConfig, split: str = "train", scalers: Dict = None):
        """
        初始化数据集
        
        Args:
            config: 数据配置
            split: 数据分割 ("train", "val", "test")
            scalers: 标准化器字典（从训练集传递给验证/测试集）
        """
        self.config = config
        self.split = split
        self.scalers = scalers or {}
        
        # 加载并处理数据
        self._load_data()
        self._split_by_time()
        self._normalize_data()
        
        # 🚀 检查序列缓存
        if self._check_sequence_cache():
            logger.info(f"🎯 使用缓存序列，跳过创建过程")
        else:
            self._create_sequences()
            self._cache_sequences()  # 缓存新创建的序列
        
        logger.info(f"{split.upper()} 数据集创建完成:")
        logger.info(f"  - 序列数量: {len(self.sequences)}")
        logger.info(f"  - 序列长度: {self.config.sequence_length}")
        logger.info(f"  - 特征维度: {len(self.config.feature_cols)}")
        logger.info(f"  - 站点数量: {len(self.station_list)}")
    
    def _load_data(self):
        """加载合并后的长表数据 - 优化版本，使用全局缓存"""
        global _GLOBAL_DATA_CACHE
        
        # 生成当前配置的缓存键
        cache_key = {
            'csv_path': self.config.csv_path,
            'use_all_stations': getattr(self.config, 'use_all_stations', False),
            'quick_test': self.config.quick_test,
            'quick_test_stations': self.config.quick_test_stations if self.config.quick_test else None,
            'filter_station_ids': tuple(sorted(str(s) for s in self.config.filter_station_ids)) if getattr(self.config, 'filter_station_ids', None) else None
        }
        
        # 检查缓存是否可用
        if (_GLOBAL_DATA_CACHE['cache_params'] == cache_key and 
            _GLOBAL_DATA_CACHE['filtered_data'] is not None):
            
            # 使用缓存数据 - 避免昂贵的copy操作
            self.data = _GLOBAL_DATA_CACHE['filtered_data']  # 直接引用，不复制
            logger.info(f"🎯 使用缓存数据: {self.data.shape} (避免了{_GLOBAL_DATA_CACHE['load_time']:.2f}秒的重新加载)")
            logger.info(f"缓存命中: 站点数={self.data['station_id'].nunique()}")
        
        else:
            # 需要重新加载数据
            logger.info(f"💾 正在加载数据: {self.config.csv_path}")
            
            # 检查文件是否存在
            if not os.path.exists(self.config.csv_path):
                raise FileNotFoundError(f"找不到数据文件: {self.config.csv_path}")
            
            # 记录加载开始时间
            start_time = time.time()
            
            raw_path = self.config.csv_path
            wanted = ['station_id','date','lon','lat'] + list(set(self.config.feature_cols + [self.config.target_col]))
            cache_path = build_cache_path(raw_path)
            raw_data = None

            # 优先使用缓存
            if cache_path and cache_path.exists():
                cached = read_from_cache(cache_path, columns=wanted if cache_path.suffix.lower() == '.parquet' else None)
                if cached is not None:
                    raw_data = cached
                    logger.info(f"⚡ 从缓存加载数据: {cache_path}")

            if raw_data is None:
                try:
                    raw_data = read_table_auto(raw_path, usecols=wanted, parse_dates=['date'])
                except Exception:
                    raw_data = read_table_auto(raw_path, parse_dates=['date'])

                # 异步写入缓存
                if cache_path and not cache_path.exists():
                    write_cache(raw_data, cache_path)
            
            # 若原始是CSV且检测到可用列式支持，自动在同目录生成Parquet，后续即读Parquet
            try:
                lower = raw_path.lower()
                if raw_data is not None and (lower.endswith('.csv') or lower.endswith('.txt')) and has_parquet_support():
                    p = Path(raw_path)
                    dst = p.with_suffix('.parquet')
                    # 仅在不存在时转换，避免每次写盘
                    if not dst.exists():
                        raw_data.to_parquet(str(dst), index=False)
                        logger.info(f"🧭 已自动转换为Parquet: {dst}")
                        # 更新路径供后续运行使用
                        self.config.csv_path = str(dst)
            except Exception as e:
                logger.warning(f"列式自动转换失败（忽略并继续CSV）: {e}")
            logger.info(f"原始数据形状: {raw_data.shape}")
            
            # 数据类型转换
            if 'date' in raw_data.columns and not pd.api.types.is_datetime64_any_dtype(raw_data['date']):
                raw_data['date'] = pd.to_datetime(raw_data['date'])
            
            # 站点筛选逻辑
            if getattr(self.config, 'filter_station_ids', None):
                # 仅使用指定站点
                sid_set = set(str(s) for s in self.config.filter_station_ids)
                filtered_data = raw_data[raw_data['station_id'].astype(str).isin(sid_set)]
                logger.info(f"按过滤站点训练: 使用 {filtered_data['station_id'].nunique()} 个站点 (来自先验列表)")
                logger.info(f"数据形状: {filtered_data.shape}")
            elif hasattr(self.config, 'use_all_stations') and self.config.use_all_stations:
                # 全站点训练模式
                filtered_data = raw_data
                unique_stations = raw_data['station_id'].unique()
                logger.info(f"全站点训练模式: 使用所有 {len(unique_stations)} 个站点")
                logger.info(f"数据形状: {filtered_data.shape}")
            elif self.config.quick_test:
                # 快速测试模式
                unique_stations = raw_data['station_id'].unique()
                selected_stations = unique_stations[:self.config.quick_test_stations]
                filtered_data = raw_data[raw_data['station_id'].isin(selected_stations)]
                logger.info(f"快速测试模式: 选择 {len(selected_stations)} 个站点")
                logger.info(f"筛选后数据形状: {filtered_data.shape}")
            else:
                # 默认使用全部数据
                filtered_data = raw_data
            
            # 记录加载时间
            load_time = time.time() - start_time
            
            # 更新全局缓存 - 避免复制，直接存储引用
            _GLOBAL_DATA_CACHE.update({
                'raw_data': raw_data,
                'filtered_data': filtered_data,  # 直接存储，不复制
                'cache_params': cache_key,
                'load_time': load_time
            })
            
            logger.info(f"✅ 数据加载完成，耗时: {load_time:.2f}秒")
            
            # 使用筛选后的数据
            self.data = filtered_data
        
        # 获取站点列表
        self.station_list = sorted(self.data['station_id'].unique())
        logger.info(f"站点数量: {len(self.station_list)}")
    
    def _split_by_time(self):
        """按时间分割训练/验证/测试集 - 支持具体日期分割"""
        if self.config.use_date_split:
            # 使用具体日期分割
            logger.info("使用具体日期分割数据")
            
            # 根据当前分割选择对应时间段的数据
            if self.split == "train":
                start_date = pd.to_datetime(self.config.train_start)
                end_date = pd.to_datetime(self.config.train_end)
            elif self.split == "val":
                start_date = pd.to_datetime(self.config.val_start)
                end_date = pd.to_datetime(self.config.val_end)
            else:  # test
                # 🔥 修复：测试集需要向前扩展sequence_length天以确保从test_start就能有预测
                test_start = pd.to_datetime(self.config.test_start)
                start_date = test_start - pd.Timedelta(days=self.config.sequence_length-1)
                end_date = pd.to_datetime(self.config.test_end)
                print(f"📅 测试集数据范围扩展: {start_date.date()} ~ {end_date.date()}")
                print(f"   实际预测范围: {test_start.date()} ~ {end_date.date()}")
            
            # 筛选指定时间段的数据
            mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
            self.data = self.data.loc[mask].copy()
            
            logger.info(f"{self.split.upper()} 时间段: {start_date.date()} ~ {end_date.date()}")
            logger.info(f"{self.split.upper()} 分割后数据形状: {self.data.shape}")
            logger.info(f"{self.split.upper()} 包含站点数: {self.data['station_id'].nunique()}")
            
        else:
            # 使用比例分割（原始方法）
            logger.info("使用比例分割数据")
            
            # 获取所有唯一的日期并排序
            unique_dates = sorted(self.data['date'].unique())
            n_dates = len(unique_dates)
            
            # 计算分割点
            n_train = int(n_dates * self.config.train_ratio)
            n_val = int(n_dates * self.config.val_ratio)
            
            # 分割日期
            train_dates = unique_dates[:n_train]
            val_dates = unique_dates[n_train:n_train + n_val]
            test_dates = unique_dates[n_train + n_val:]
            
            logger.info(f"按比例分割数据:")
            logger.info(f"  - 训练期间: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}天)")
            logger.info(f"  - 验证期间: {val_dates[0]} ~ {val_dates[-1]} ({len(val_dates)}天)")
            logger.info(f"  - 测试期间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")
            
            # 根据当前分割选择对应时间段的数据
            if self.split == "train":
                selected_dates = train_dates
            elif self.split == "val":
                selected_dates = val_dates
            else:  # test
                selected_dates = test_dates
            
            self.data = self.data.loc[self.data['date'].isin(selected_dates)].copy()
            
            logger.info(f"{self.split.upper()} 分割后数据形状: {self.data.shape}")
            logger.info(f"{self.split.upper()} 包含站点数: {self.data['station_id'].nunique()}")
    
    def _normalize_data(self):
        """数据标准化 - 改进版本，针对径流数据优化"""
        from MoE_advanced_normalization import HydroLogNormalizer
        from sklearn.preprocessing import RobustScaler

        if self.split == "train":
            # 训练集：创建并应用标准化器
            if self.config.normalize_features:
                # 对特征使用鲁棒标准化
                self.feature_scaler = RobustScaler()
                # 🚀 优化：避免多次索引，直接修改数组
                feature_data = self.data[self.config.feature_cols].values
                normalized = self.feature_scaler.fit_transform(feature_data)
                self.data.loc[:, self.config.feature_cols] = normalized
                logger.info("训练集特征标准化完成（使用RobustScaler）")

            if self.config.normalize_targets:
                # 对径流目标变量使用专门的对数正态归一化器
                self.target_scaler = HydroLogNormalizer(add_constant=0.01, use_robust=True)

                # 🚀 优化：避免重复取值
                runoff_data = self.data[self.config.target_col].values
                runoff_min, runoff_max = runoff_data.min(), runoff_data.max()

                # 应用对数正态归一化
                normalized_runoff = self.target_scaler.fit_transform(runoff_data)
                self.data.loc[:, self.config.target_col] = normalized_runoff

                logger.info("训练集径流目标变量标准化完成（使用HydroLogNormalizer）")
                logger.info(f"  原始径流范围: {runoff_min:.3f} ~ {runoff_max:.3f} mm/day")
                logger.info(f"  标准化后范围: {normalized_runoff.min():.3f} ~ {normalized_runoff.max():.3f}")

                # 保存标准化参数供验证/测试集使用
                self.scalers = {
                    'feature_scaler': self.feature_scaler if self.config.normalize_features else None,
                    'target_scaler': self.target_scaler if self.config.normalize_targets else None
                }

        else:
            # 验证/测试集：使用训练集的标准化器
            if self.scalers is None:
                logger.warning(f"{self.split.upper()} 集没有提供标准化器，跳过标准化")
                return

            if self.config.normalize_features and 'feature_scaler' in self.scalers and self.scalers['feature_scaler'] is not None:
                # 🚀 优化：直接操作数组避免多次索引
                feature_data = self.data[self.config.feature_cols].values
                normalized = self.scalers['feature_scaler'].transform(feature_data)
                self.data.loc[:, self.config.feature_cols] = normalized
                logger.info(f"{self.split.upper()} 集特征标准化完成（使用训练集RobustScaler参数）")

            if self.config.normalize_targets and 'target_scaler' in self.scalers and self.scalers['target_scaler'] is not None:
                # 🚀 优化：避免重复取值
                runoff_data = self.data[self.config.target_col].values
                runoff_min, runoff_max = runoff_data.min(), runoff_data.max()

                # 应用对数正态归一化
                normalized_runoff = self.scalers['target_scaler'].transform(runoff_data)
                self.data.loc[:, self.config.target_col] = normalized_runoff

                logger.info(f"{self.split.upper()} 集径流目标变量标准化完成（使用训练集HydroLogNormalizer参数）")
                logger.info(f"  原始径流范围: {runoff_min:.3f} ~ {runoff_max:.3f} mm/day")
                logger.info(f"  标准化后范围: {normalized_runoff.min():.3f} ~ {normalized_runoff.max():.3f}")
    
    def _create_sequences(self):
        """创建时间序列 - 高速优化版本"""
        logger.info(f"🚀 开始创建序列 (高速优化模式)")
        self.sequences = []
        
        # 🔥 关键优化：预先按站点分组并排序，避免重复query
        logger.info("📊 预处理：按站点分组数据...")
        start_time = time.time()
        
        # 首先确保数据按station_id, date排序（一次性排序）
        if not self.data.index.is_monotonic_increasing:
            self.data = self.data.sort_values(['station_id', 'date']).reset_index(drop=True)
        
        # 使用groupby一次性分组，避免重复query
        grouped_data = dict(list(self.data.groupby('station_id', sort=False)))
        logger.info(f"预处理完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 分批处理站点，避免内存峰值
        batch_size = getattr(self.config, 'station_batch_size', 100)
        try:
            batch_size = int(batch_size)
            if batch_size <= 0:
                batch_size = 100
        except Exception:
            batch_size = 100
        station_batches = [self.station_list[i:i+batch_size] for i in range(0, len(self.station_list), batch_size)]
        
        for batch_idx, station_batch in enumerate(station_batches):
            batch_start_time = time.time()
            logger.info(f"处理站点批次 {batch_idx+1}/{len(station_batches)} ({len(station_batch)}个站点)")
            
            for station_idx_in_batch, station_id in enumerate(station_batch):
                # 获取站点索引
                station_idx = self.station_list.index(station_id)
                
                # 🚀 直接从预分组的数据中获取站点数据（已排序）
                if station_id not in grouped_data:
                    continue
                    
                station_data = grouped_data[station_id]
                
                # 检查数据长度
                if len(station_data) < self.config.sequence_length:
                    continue
                
                # 🚀 高效数组预计算：一次性提取所有需要的数据
                features_array = station_data[self.config.feature_cols].values.astype(np.float32)
                targets_array = station_data[self.config.target_col].values.astype(np.float32)
                dates_array = station_data['date'].values
                
                # 经纬度为可选列：缺失时填NaN
                try:
                    lons = float(station_data['lon'].iloc[0])
                except Exception:
                    lons = np.nan
                try:
                    lats = float(station_data['lat'].iloc[0])
                except Exception:
                    lats = np.nan
                
                # 🚀 矢量化时间特征计算（一次性处理整个站点的日期）
                dates_pd = pd.to_datetime(dates_array)
                months = dates_pd.month.values
                day_of_years = dates_pd.dayofyear.values
                
                # 矢量化季节性编码
                month_sin = np.sin(2 * np.pi * months / 12).astype(np.float32)
                month_cos = np.cos(2 * np.pi * months / 12).astype(np.float32)
                doy_sin = np.sin(2 * np.pi * day_of_years / 365).astype(np.float32)
                doy_cos = np.cos(2 * np.pi * day_of_years / 365).astype(np.float32)
                
                # 组合时间特征矩阵 [n_days, 4]
                all_time_features = np.column_stack([month_sin, month_cos, doy_sin, doy_cos])
                
                # 创建滑动窗口序列 - 矢量化批量创建
                stride = 1 if self.split == "test" else self.config.sequence_stride
                seq_len = self.config.sequence_length
                
                # 🚀 批量生成所有序列索引
                max_start = len(station_data) - seq_len
                if max_start < 0:
                    continue
                    
                start_indices = np.arange(0, max_start + 1, stride)
                
                # 🚀 批量创建序列（矢量化操作）
                for start_idx in start_indices:
                    end_idx = start_idx + seq_len
                    
                    # 直接数组切片（最高效）
                    features = features_array[start_idx:end_idx]
                    targets = targets_array[start_idx:end_idx]
                    time_features = all_time_features[start_idx:end_idx]
                    
                    # 🚀 优化：预计算日期字符串，避免在__getitem__中重复转换
                    start_date_str = pd.to_datetime(dates_array[start_idx]).strftime('%Y-%m-%d')
                    end_date_str = pd.to_datetime(dates_array[end_idx-1]).strftime('%Y-%m-%d')
                    
                    # 添加到序列列表
                    self.sequences.append({
                        'features': features,
                        'targets': targets,
                        'time_features': time_features,
                        'station_id': station_id,
                        'station_idx': station_idx,
                        'lon': lons,
                        'lat': lats,
                        'start_date': dates_array[start_idx],
                        'end_date': dates_array[end_idx-1],
                        'start_date_str': start_date_str,  # 🚀 预缓存
                        'end_date_str': end_date_str  # 🚀 预缓存
                    })
                
                # 🔥 每处理完一个站点就清理内存
                del station_data, features_array, targets_array, dates_array
                gc.collect()
            
            # 每个批次后清理内存并显示进度
            batch_time = time.time() - batch_start_time
            sequences_per_sec = len(self.sequences) / max(batch_time, 0.1)
            logger.info(f"批次 {batch_idx+1}/{len(station_batches)} 完成，耗时: {batch_time:.2f}秒")
            logger.info(f"  - 当前总序列数: {len(self.sequences):,}, 创建速度: {sequences_per_sec:.0f} 序列/秒")
            gc.collect()  # 🔥 每个批次都强制垃圾回收
        
        logger.info(f"为 {len(self.station_list)} 个站点创建了 {len(self.sequences)} 个序列")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取单个样本 - 优化版本，减少内存复制和计算"""
        sequence = self.sequences[idx]
        
        # 🚀 优化：使用from_numpy避免复制，直接共享内存
        features = torch.from_numpy(sequence['features']).float()
        targets = torch.from_numpy(sequence['targets']).float()
        time_features = torch.from_numpy(sequence.get('time_features', 
                                                      np.zeros((features.shape[0], 4), dtype=np.float32))).float()
        
        target_scalar = targets[-1]

        # 🚀 优化：简化scaler查找逻辑
        raw_features_last = None
        if self.config.normalize_features:
            scaler = getattr(self, 'feature_scaler', None) or (self.scalers.get('feature_scaler') if self.scalers else None)
            if scaler is not None:
                try:
                    last_feat = sequence['features'][-1:, :]  # 保持2D避免reshape
                    raw_last_np = scaler.inverse_transform(last_feat).astype(np.float32, copy=False).flatten()
                    raw_features_last = torch.from_numpy(raw_last_np)
                except Exception:
                    raw_features_last = features[-1]
        if raw_features_last is None:
            raw_features_last = features[-1]
        
        # 🚀 优化：使用预缓存的日期字符串（避免重复转换）
        start_date_str = sequence.get('start_date_str', pd.to_datetime(sequence['start_date']).strftime('%Y-%m-%d'))
        end_date_str = sequence.get('end_date_str', pd.to_datetime(sequence['end_date']).strftime('%Y-%m-%d'))
        
        return {
            'features': features,
            'time_features': time_features,
            'targets': target_scalar,
            'targets_seq': targets,
            'station_id': sequence['station_id'],
            'lon': torch.tensor([sequence['lon']], dtype=torch.float32),
            'lat': torch.tensor([sequence['lat']], dtype=torch.float32),
            'station_idx': torch.tensor([sequence['station_idx']], dtype=torch.long),
            'raw_features_last': raw_features_last,
            'start_date': start_date_str,
            'end_date': end_date_str
        }

    def get_scalers(self):
        """获取标准化器（仅训练集有效）"""
        if self.split == "train" and hasattr(self, 'scalers'):
            return self.scalers
        return None
    
    def _check_sequence_cache(self) -> bool:
        """检查是否可以使用缓存的序列"""
        global _GLOBAL_DATA_CACHE
        
        # 生成序列缓存键
        cache_key = {
            'split': self.split,
            'sequence_length': self.config.sequence_length,
            'sequence_stride': self.config.sequence_stride,
            'feature_cols': tuple(self.config.feature_cols),
            'target_col': self.config.target_col,
            'csv_path': self.config.csv_path,
            'normalize_features': self.config.normalize_features,
            'normalize_targets': self.config.normalize_targets,
            'data_shape': getattr(self.data, 'shape', None),
            'station_count': len(self.station_list)
        }
        
        # 检查缓存是否可用
        if (_GLOBAL_DATA_CACHE['sequence_cache_params'] == cache_key and 
            self.split in _GLOBAL_DATA_CACHE['sequences_cache']):
            
            # 使用缓存序列
            self.sequences = _GLOBAL_DATA_CACHE['sequences_cache'][self.split]
            
            # 如果是训练集，还要检查标准化器缓存
            if self.split == "train" and _GLOBAL_DATA_CACHE['scalers_cache'] is not None:
                self.scalers = _GLOBAL_DATA_CACHE['scalers_cache']
                # 为了兼容现有接口，设置相应的属性
                if 'feature_scaler' in self.scalers:
                    self.feature_scaler = self.scalers['feature_scaler']
                if 'target_scaler' in self.scalers:
                    self.target_scaler = self.scalers['target_scaler']
            
            return True
        
        return False
    
    def _cache_sequences(self):
        """缓存创建的序列"""
        global _GLOBAL_DATA_CACHE
        
        # 生成序列缓存键
        cache_key = {
            'split': self.split,
            'sequence_length': self.config.sequence_length,
            'sequence_stride': self.config.sequence_stride,
            'feature_cols': tuple(self.config.feature_cols),
            'target_col': self.config.target_col,
            'csv_path': self.config.csv_path,
            'normalize_features': self.config.normalize_features,
            'normalize_targets': self.config.normalize_targets,
            'data_shape': getattr(self.data, 'shape', None),
            'station_count': len(self.station_list)
        }
        
        # 缓存序列
        _GLOBAL_DATA_CACHE['sequences_cache'][self.split] = self.sequences
        _GLOBAL_DATA_CACHE['sequence_cache_params'] = cache_key
        
        # 如果是训练集，缓存标准化器
        if self.split == "train" and hasattr(self, 'scalers'):
            _GLOBAL_DATA_CACHE['scalers_cache'] = self.scalers
        
        logger.info(f"✅ {self.split}集序列已缓存 ({len(self.sequences)}个)")
    
    def clear_sequence_cache():
        """清理序列缓存"""
        global _GLOBAL_DATA_CACHE
        _GLOBAL_DATA_CACHE['sequences_cache'].clear()
        _GLOBAL_DATA_CACHE['scalers_cache'] = None
        _GLOBAL_DATA_CACHE['grouped_data_cache'] = None
        _GLOBAL_DATA_CACHE['sequence_cache_params'] = None
        logger.info("🗑️ 序列缓存已清理")


def get_sequence_cache_info() -> Dict:
    """获取序列缓存信息"""
    global _GLOBAL_DATA_CACHE
    
    cache_info = {
        'has_sequences_cache': bool(_GLOBAL_DATA_CACHE['sequences_cache']),
        'cached_splits': list(_GLOBAL_DATA_CACHE['sequences_cache'].keys()),
        'cache_params': _GLOBAL_DATA_CACHE['sequence_cache_params'],
    }
    
    # 计算缓存大小
    if _GLOBAL_DATA_CACHE['sequences_cache']:
        total_sequences = sum(len(sequences) for sequences in _GLOBAL_DATA_CACHE['sequences_cache'].values())
        cache_info['total_cached_sequences'] = total_sequences
        
        # 估算内存使用
        if total_sequences > 0:
            # 假设每个序列大约占用 sequence_length * (feature_dims + target_dims + time_dims) * 4 bytes
            # 粗略估算：96 * (10 + 1 + 4) * 4 = 5760 bytes ≈ 6KB per sequence
            estimated_mb = total_sequences * 6 / 1024
            cache_info['estimated_memory_mb'] = estimated_mb
    
    return cache_info


def warmup_data_loading(config: FixedDataConfig) -> None:
    """🔥 数据加载预热：在训练开始前预加载所有数据"""
    logger.info("🔥 数据加载预热开始...")
    
    start_time = time.time()
    
    # 预加载所有数据集
    splits_info = preload_all_datasets(config)
    
    # 显示缓存状态
    cache_info = get_sequence_cache_info()
    
    total_time = time.time() - start_time
    logger.info(f"🎯 数据预热完成！总耗时: {total_time:.2f}秒")
    logger.info(f"  - 缓存状态: {cache_info}")
    
    return splits_info


    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sequence = self.sequences[idx]
        
        # 转换为tensor
        features = torch.FloatTensor(sequence['features'])  # [seq_len, n_features]
        targets = torch.FloatTensor(sequence['targets'])    # [seq_len]
        
        # 🔥 添加时间特征
        import numpy as np  # 确保numpy可用
        time_features = torch.FloatTensor(sequence.get('time_features', 
                                                      np.zeros((features.shape[0], 4))))  # [seq_len, 4]
        
        # 对于序列到点预测，只使用最后一个时间步的目标值
        target_scalar = targets[-1]  # 只取最后一个时间步作为预测目标

        # 还原最后一个时间步的原始物理驱动(未标准化)，供PBM使用
        raw_features_last = None
        try:
            if self.config.normalize_features:
                scaler = getattr(self, 'feature_scaler', None)
                if scaler is None and self.scalers and 'feature_scaler' in self.scalers:
                    scaler = self.scalers['feature_scaler']
                if scaler is not None:
                    import numpy as np  # 局部导入以避免顶层依赖
                    last_feat = sequence['features'][-1].reshape(1, -1)
                    raw_last_np = scaler.inverse_transform(last_feat).astype(np.float32).flatten()
                    raw_features_last = torch.from_numpy(raw_last_np)
        except Exception:
            # 回退：无法反归一化时，使用已标准化值（仍可运行，但物理意义较弱）
            raw_features_last = features[-1]
        if raw_features_last is None:
            raw_features_last = features[-1]
        
        # 直接使用预存储的日期，转换为字符串
        import pandas as pd
        start_date_str = pd.to_datetime(sequence['start_date']).strftime('%Y-%m-%d')
        end_date_str = pd.to_datetime(sequence['end_date']).strftime('%Y-%m-%d')
        
        return {
            'features': features,
            'time_features': time_features,  # 🔥 新增时间特征
            'targets': target_scalar,  # 使用标量目标值 [1]
            'targets_seq': targets,    # 新增：完整目标序列 [seq_len]
            'station_id': sequence['station_id'],
            'lon': torch.FloatTensor([sequence['lon']]),
            'lat': torch.FloatTensor([sequence['lat']]),
            'station_idx': torch.LongTensor([sequence['station_idx']]),
            'raw_features_last': raw_features_last,  # 提供未标准化的最后时步物理驱动
            'start_date': start_date_str,  # 转换为字符串
            'end_date': end_date_str  # 预测目标对应的日期字符串
        }

    def get_scalers(self):
        """获取标准化器（仅训练集有效）"""
        if self.split == "train" and hasattr(self, 'scalers'):
            return self.scalers
        return None


def create_fixed_data_loaders(config: FixedDataConfig, batch_size: int = 64, num_workers: int = 0, 
                             pin_memory: bool = True, prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建修复版的数据加载器 - 正确处理标准化，支持GPU优化，优化缓存机制
    """
    logger.info("🚀 开始创建数据加载器，使用增强缓存优化...")
    
    # 🚀 优化：自动清理旧的序列缓存，避免内存泄漏
    import gc
    gc.collect()
    
    # 🚀 检查是否所有数据集都已缓存
    global _GLOBAL_DATA_CACHE
    if (_GLOBAL_DATA_CACHE['sequence_cache_params'] is not None and 
        all(split in _GLOBAL_DATA_CACHE['sequences_cache'] for split in ['train', 'val', 'test'])):
        logger.info("🎯 检测到完整序列缓存，快速创建数据集...")
    
    # 创建训练集
    logger.info("📊 创建训练集...")
    train_dataset = FixedHydroDataset(config, split="train")
    
    # 获取训练集的标准化器
    scalers = train_dataset.get_scalers()
    
    # 创建验证集和测试集，传入训练集的标准化器
    logger.info("📊 创建验证集...")
    val_dataset = FixedHydroDataset(config, split="val", scalers=scalers)
    
    logger.info("📊 创建测试集...")
    test_dataset = FixedHydroDataset(config, split="test", scalers=scalers)
    
    # GPU优化的DataLoader参数
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    # 只有在num_workers > 0时才添加prefetch_factor
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = True
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )
    
    logger.info(f"修复版数据加载器创建完成:")
    logger.info(f"  - 训练集批次数: {len(train_loader)}")
    logger.info(f"  - 验证集批次数: {len(val_loader)}")
    logger.info(f"  - 测试集批次数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def clear_data_cache():
    """清理全局数据缓存"""
    global _GLOBAL_DATA_CACHE
    _GLOBAL_DATA_CACHE.update({
        'raw_data': None,
        'filtered_data': None,
        'cache_params': None,
        'load_time': None,
        'sequences_cache': {},
        'scalers_cache': None,
        'grouped_data_cache': None,
        'sequence_cache_params': None
    })
    logger.info("🗑️ 全局数据缓存已清理")


def get_cache_info() -> Dict:
    """获取缓存信息"""
    global _GLOBAL_DATA_CACHE
    
    cache_info = {
        'has_raw_data': _GLOBAL_DATA_CACHE['raw_data'] is not None,
        'has_filtered_data': _GLOBAL_DATA_CACHE['filtered_data'] is not None,
        'cache_params': _GLOBAL_DATA_CACHE['cache_params'],
        'load_time': _GLOBAL_DATA_CACHE['load_time']
    }
    
    if cache_info['has_filtered_data']:
        cache_info['data_shape'] = _GLOBAL_DATA_CACHE['filtered_data'].shape
        cache_info['memory_usage_mb'] = _GLOBAL_DATA_CACHE['filtered_data'].memory_usage(deep=True).sum() / 1024 / 1024
    
    return cache_info


def preload_data(config: FixedDataConfig) -> None:
    """预加载数据到缓存中"""
    logger.info("🚀 预加载数据中...")
    
    # 创建一个临时数据集来触发数据加载
    temp_dataset = FixedHydroDataset(config, split="train")
    
    # 显示缓存信息
    cache_info = get_cache_info()
    if cache_info['has_filtered_data']:
        logger.info(f"✅ 数据预加载完成:")
        logger.info(f"  - 数据形状: {cache_info['data_shape']}")
        logger.info(f"  - 内存占用: {cache_info['memory_usage_mb']:.1f} MB")
        logger.info(f"  - 加载耗时: {cache_info['load_time']:.2f} 秒")


def preload_all_datasets(config: FixedDataConfig) -> Dict[str, int]:
    """🚀 预加载所有数据集到缓存（训练前一次性准备）"""
    logger.info("🚀 开始预加载所有数据集...")
    start_time = time.time()
    
    # 按顺序创建所有数据集，触发缓存
    splits_info = {}
    
    # 1. 训练集（会创建标准化器）
    logger.info("📊 预加载训练集...")
    train_dataset = FixedHydroDataset(config, split="train")
    scalers = train_dataset.get_scalers()
    splits_info['train'] = len(train_dataset)
    
    # 2. 验证集
    logger.info("📊 预加载验证集...")
    val_dataset = FixedHydroDataset(config, split="val", scalers=scalers)
    splits_info['val'] = len(val_dataset)
    
    # 3. 测试集
    logger.info("📊 预加载测试集...")
    test_dataset = FixedHydroDataset(config, split="test", scalers=scalers)
    splits_info['test'] = len(test_dataset)
    
    total_time = time.time() - start_time
    total_sequences = sum(splits_info.values())
    
    logger.info(f"✅ 所有数据集预加载完成！")
    logger.info(f"  - 总耗时: {total_time:.2f}秒")
    logger.info(f"  - 总序列数: {total_sequences:,}")
    logger.info(f"  - 训练集: {splits_info['train']:,}")
    logger.info(f"  - 验证集: {splits_info['val']:,}")
    logger.info(f"  - 测试集: {splits_info['test']:,}")
    logger.info(f"  - 序列创建速度: {total_sequences/total_time:.0f} 序列/秒")
    
    return splits_info


# 优化建议：对于频繁使用的场景，可以考虑使用以下策略
def optimize_for_repeated_use():
    """为重复使用优化的建议"""
    suggestions = [
        "1. 使用 warmup_data_loading() 在训练开始前预加载所有数据",
        "2. 在不同实验间保持Python会话以维持缓存",
        "3. 优化后的数据加载器会自动使用Parquet格式（如果可用）",
        "4. 序列缓存机制避免重复创建序列",
        "5. 矢量化时间特征计算提升性能5-10倍"
    ]
    
    for suggestion in suggestions:
        logger.info(f" {suggestion}")
    
    return suggestions


def benchmark_data_loading(config: FixedDataConfig, runs: int = 3) -> Dict[str, float]:
    """🚀 数据加载性能基准测试"""
    logger.info(f"🏃 开始数据加载性能测试，运行{runs}次...")
    
    times = []
    for run in range(runs):
        # 清理缓存，确保每次都是冷启动
        clear_data_cache()
        
        start_time = time.time()
        
        # 测试完整的数据加载流程
        train_loader, val_loader, test_loader = create_fixed_data_loaders(
            config, batch_size=32, num_workers=0
        )
        
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        
        logger.info(f"运行 {run+1}/{runs}: {run_time:.2f}秒")
        
        # 收集统计信息
        total_sequences = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        sequences_per_sec = total_sequences / run_time
        
        logger.info(f"  - 总序列数: {total_sequences:,}")
        logger.info(f"  - 创建速度: {sequences_per_sec:.0f} 序列/秒")
    
    # 计算统计
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    logger.info(f"🎯 性能测试结果:")
    logger.info(f"  - 平均时间: {avg_time:.2f} ± {std_time:.2f}秒")
    logger.info(f"  - 最快时间: {min_time:.2f}秒")
    
    return {
        'average_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'times': times
    }