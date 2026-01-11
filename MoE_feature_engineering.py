"""
特征工程增强模块 - 从基础特征生成丰富的衍生特征
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class HydroFeatureEngineer:
    """水文特征工程器 - 从基础特征生成丰富的衍生特征"""
    
    def __init__(self, 
                 window_sizes: List[int] = [3, 7, 14, 30],
                 seasonal_features: bool = True,
                 extreme_features: bool = True,
                 interaction_features: bool = True):
        """
        初始化特征工程器
        
        Args:
            window_sizes: 滑动窗口大小列表
            seasonal_features: 是否生成季节性特征
            extreme_features: 是否生成极值特征
            interaction_features: 是否生成交互特征
        """
        self.window_sizes = window_sizes
        self.seasonal_features = seasonal_features
        self.extreme_features = extreme_features
        self.interaction_features = interaction_features
        
        # 特征名称映射
        self.base_features = ['precip', 'temp', 'pet']
        self.feature_names = []
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有衍生特征
        
        Args:
            data: 包含基础特征的DataFrame，需要有时间索引
            
        Returns:
            包含所有特征的DataFrame
        """
        result_df = data.copy()
        
        # 1. 滑动窗口统计特征
        result_df = self._add_rolling_features(result_df)
        
        # 2. 季节性特征
        if self.seasonal_features:
            result_df = self._add_seasonal_features(result_df)
        
        # 3. 极值特征
        if self.extreme_features:
            result_df = self._add_extreme_features(result_df)
        
        # 4. 交互特征
        if self.interaction_features:
            result_df = self._add_interaction_features(result_df)
        
        # 5. 趋势特征
        result_df = self._add_trend_features(result_df)
        
        # 6. 水文指数特征
        result_df = self._add_hydrological_indices(result_df)
        
        return result_df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加滑动窗口统计特征"""
        
        for feature in self.base_features:
            if feature not in df.columns:
                continue
                
            for window in self.window_sizes:
                # 基础统计
                df[f'{feature}_mean_{window}d'] = df[feature].rolling(window, min_periods=1).mean()
                df[f'{feature}_std_{window}d'] = df[feature].rolling(window, min_periods=1).std()
                df[f'{feature}_max_{window}d'] = df[feature].rolling(window, min_periods=1).max()
                df[f'{feature}_min_{window}d'] = df[feature].rolling(window, min_periods=1).min()
                
                # 高级统计
                df[f'{feature}_skew_{window}d'] = df[feature].rolling(window, min_periods=3).skew()
                df[f'{feature}_kurt_{window}d'] = df[feature].rolling(window, min_periods=4).kurtosis()
                
                # 变化率
                df[f'{feature}_change_{window}d'] = (df[feature] - df[f'{feature}_mean_{window}d']) / (df[f'{feature}_std_{window}d'] + 1e-8)
                
                # 累积特征
                df[f'{feature}_sum_{window}d'] = df[feature].rolling(window, min_periods=1).sum()
        
        return df
    
    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加季节性特征"""
        
        # 确保有时间索引
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df.index = pd.to_datetime(df['date'])
            else:
                print("警告：无法生成季节性特征，缺少时间信息")
                return df
        
        # 基础时间特征
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        df['season'] = df.index.month % 12 // 3 + 1
        
        # 周期性编码
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 季节性统计
        for feature in self.base_features:
            if feature not in df.columns:
                continue
            
            # 月度统计
            monthly_stats = df.groupby(df.index.month)[feature].agg(['mean', 'std']).add_prefix(f'{feature}_monthly_')
            df = df.join(monthly_stats, on=df.index.month)
            
            # 季节性异常
            df[f'{feature}_seasonal_anomaly'] = df[feature] - df[f'{feature}_monthly_mean']
        
        return df
    
    def _add_extreme_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加极值特征"""
        
        for feature in self.base_features:
            if feature not in df.columns:
                continue
            
            # 分位数特征
            for q in [0.1, 0.25, 0.75, 0.9, 0.95, 0.99]:
                threshold = df[feature].quantile(q)
                df[f'{feature}_above_p{int(q*100)}'] = (df[feature] > threshold).astype(int)
            
            # 极值检测
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            df[f'{feature}_outlier'] = ((df[feature] < Q1 - 1.5*IQR) | 
                                       (df[feature] > Q3 + 1.5*IQR)).astype(int)
            
            # 连续极值天数
            extreme_threshold = df[feature].quantile(0.9)
            df[f'{feature}_extreme'] = (df[feature] > extreme_threshold).astype(int)
            df[f'{feature}_extreme_days'] = df[f'{feature}_extreme'].groupby(
                (df[f'{feature}_extreme'] != df[f'{feature}_extreme'].shift()).cumsum()
            ).cumsum() * df[f'{feature}_extreme']
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交互特征"""
        
        # 基础交互
        if 'precip' in df.columns and 'temp' in df.columns:
            df['precip_temp_ratio'] = df['precip'] / (df['temp'] + 273.15)  # 考虑绝对温度
            df['precip_temp_product'] = df['precip'] * np.maximum(df['temp'], 0)  # 只考虑正温度
        
        if 'temp' in df.columns and 'pet' in df.columns:
            df['temp_pet_ratio'] = df['temp'] / (df['pet'] + 1e-8)
            df['temp_pet_diff'] = df['temp'] - df['pet']
        
        if 'precip' in df.columns and 'pet' in df.columns:
            df['precip_pet_ratio'] = df['precip'] / (df['pet'] + 1e-8)
            df['water_balance'] = df['precip'] - df['pet']  # 简单水量平衡
        
        # 高级交互
        if all(f in df.columns for f in ['precip', 'temp', 'pet']):
            # 有效降水（考虑温度影响）
            df['effective_precip'] = df['precip'] * (1 + 0.1 * np.maximum(df['temp'] - 5, 0))
            
            # 蒸发压力指数
            df['evap_stress'] = df['pet'] / (df['precip'] + 1e-8)
            
            # 综合水文指数
            df['hydro_index'] = (df['precip'] - df['pet']) / (df['temp'] + 10)
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势特征"""
        
        for feature in self.base_features:
            if feature not in df.columns:
                continue
            
            # 短期趋势
            for window in [3, 7, 14]:
                df[f'{feature}_trend_{window}d'] = df[feature].rolling(window, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            
            # 变化率
            df[f'{feature}_pct_change_1d'] = df[feature].pct_change(1)
            df[f'{feature}_pct_change_7d'] = df[feature].pct_change(7)
            
            # 动量指标
            df[f'{feature}_momentum_3d'] = df[feature] - df[feature].shift(3)
            df[f'{feature}_momentum_7d'] = df[feature] - df[feature].shift(7)
        
        return df
    
    def _add_hydrological_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加水文学指数特征"""
        
        if 'precip' in df.columns:
            # 干旱指数
            for window in [30, 60, 90]:
                precip_sum = df['precip'].rolling(window, min_periods=1).sum()
                precip_mean = df['precip'].rolling(window*4, min_periods=1).mean() * window  # 长期平均
                df[f'drought_index_{window}d'] = (precip_sum - precip_mean) / (precip_mean + 1e-8)
            
            # 降水强度指数
            df['precip_intensity'] = df['precip'] / (df['precip'].rolling(7, min_periods=1).count() + 1e-8)
            
            # 连续无雨天数
            no_rain = (df['precip'] <= 0.1).astype(int)
            df['dry_spell_length'] = no_rain.groupby(
                (no_rain != no_rain.shift()).cumsum()
            ).cumsum() * no_rain
        
        if 'temp' in df.columns:
            # 度日指数
            df['heating_degree_days'] = np.maximum(18 - df['temp'], 0)
            df['cooling_degree_days'] = np.maximum(df['temp'] - 18, 0)
            df['growing_degree_days'] = np.maximum(df['temp'] - 5, 0)
            
            # 冰点天数
            df['freezing_days'] = (df['temp'] <= 0).astype(int)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """获取所有特征名称"""
        return [col for col in df.columns if col not in ['date', 'runoff', 'station_id']]


class AdaptiveFeatureSelector:
    """自适应特征选择器"""
    
    def __init__(self, max_features: int = 50, correlation_threshold: float = 0.95):
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.selected_features = []
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        基于相关性和重要性选择特征
        
        Args:
            X: 特征DataFrame
            y: 目标变量
            
        Returns:
            选择的特征名称列表
        """
        # 1. 移除高相关性特征
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > self.correlation_threshold)]
        
        X_filtered = X.drop(columns=to_drop)
        
        # 2. 基于与目标变量的相关性排序
        target_corr = X_filtered.corrwith(y).abs().sort_values(ascending=False)
        
        # 3. 选择top特征
        selected = target_corr.head(self.max_features).index.tolist()
        
        self.selected_features = selected
        return selected


if __name__ == "__main__":
    # 测试特征工程
    print("🧪 测试水文特征工程...")
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    test_data = pd.DataFrame({
        'precip': np.random.exponential(2, len(dates)),
        'temp': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + np.random.normal(0, 3, len(dates)),
        'pet': 3 + 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + np.random.normal(0, 0.5, len(dates)),
        'runoff': np.random.exponential(1, len(dates))
    }, index=dates)
    
    # 特征工程
    engineer = HydroFeatureEngineer()
    enhanced_data = engineer.engineer_features(test_data)
    
    print(f"原始特征数: {len(test_data.columns)}")
    print(f"增强后特征数: {len(enhanced_data.columns)}")
    print(f"新增特征数: {len(enhanced_data.columns) - len(test_data.columns)}")
    
    # 特征选择
    selector = AdaptiveFeatureSelector(max_features=30)
    feature_cols = [col for col in enhanced_data.columns if col != 'runoff']
    selected_features = selector.select_features(
        enhanced_data[feature_cols], 
        enhanced_data['runoff']
    )
    
    print(f"选择的特征数: {len(selected_features)}")
    print("✅ 特征工程测试完成！")
