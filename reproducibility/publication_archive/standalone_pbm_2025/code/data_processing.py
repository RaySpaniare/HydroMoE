#!/usr/bin/env python3
"""
HydroPy数据处理模块
包含CSV数据加载、验证和预处理功能
"""

import os
import pandas as pd
import numpy as np
import warnings

# 导入配置
from config import CSV_FILES, DATA_VALIDATION

warnings.filterwarnings('ignore')


def load_csv_forcing_data(csv_path):
    """加载CSV强迫数据"""
    print(f"加载CSV文件: {os.path.basename(csv_path)}")

    # 读取CSV文件
    df = pd.read_csv(csv_path, header=None, encoding='utf-8')

    # 解析站点信息 - 从第2行开始，每行是一个站点
    station_names = df.iloc[1:, 0].values  # 从第2行开始，第1列是站点名
    longitudes = pd.to_numeric(df.iloc[1:, 1].values, errors='coerce')  # 第2列是经度
    latitudes = pd.to_numeric(df.iloc[1:, 2].values, errors='coerce')   # 第3列是纬度

    # 解析日期 - 第1行，从第4列开始是日期（浮点数格式）
    date_values = df.iloc[0, 3:].values  # 第1行，从第4列开始
    # 转换为整数再解析日期
    int_dates = [int(x) for x in date_values if not pd.isna(x)]
    dates = pd.to_datetime(int_dates, format='%Y%m%d', errors='coerce')

    # 解析数据值 - 从第2行开始，从第4列开始
    data_values = df.iloc[1:, 3:].values.astype(float).T  # 转置，使时间为第一维

    # 创建站点信息DataFrame
    stations_df = pd.DataFrame({
        'station_name': station_names,
        'longitude': longitudes,
        'latitude': latitudes
    })

    # 清理站点信息
    stations_df = stations_df.dropna()

    print(f"加载完成: {len(stations_df)} 个站点, {len(dates)} 天数据")
    print(f"   时间范围: {dates[0].strftime('%Y-%m-%d')} 到 {dates[-1].strftime('%Y-%m-%d')}")

    return {
        'data': data_values,
        'dates': dates,
        'stations': stations_df
    }


def validate_data_quality(data_dict, data_type="unknown"):
    """验证数据质量"""
    print(f"🔍 验证{data_type}数据质量...")
    
    data = data_dict['data']
    stations = data_dict['stations']
    
    n_times, n_stations = data.shape
    print(f"   数据维度: {n_times} 天 × {n_stations} 个站点")
    
    # 检查缺失值
    missing_ratio = np.isnan(data).sum() / data.size
    print(f"   缺失值比例: {missing_ratio:.2%}")
    
    if missing_ratio > DATA_VALIDATION['max_missing_ratio']:
        print(f"   - 缺失值比例过高 (>{DATA_VALIDATION['max_missing_ratio']:.1%})")
    
    # 检查每个站点的数据质量
    valid_stations = []
    for i, (idx, station) in enumerate(stations.iterrows()):
        station_data = data[:, i]
        valid_count = np.sum(~np.isnan(station_data))
        valid_ratio = valid_count / len(station_data)
        
        if valid_count >= DATA_VALIDATION['min_valid_points'] and valid_ratio >= 0.7:
            valid_stations.append(i)
        else:
            print(f"   - 站点 {station['station_name']}: 有效数据不足 ({valid_count}/{len(station_data)})")
    
    print(f"   - 有效站点: {len(valid_stations)}/{n_stations}")
    
    return valid_stations


def detect_outliers(data, threshold=None):
    """检测异常值"""
    if threshold is None:
        threshold = DATA_VALIDATION['outlier_threshold']
    
    # 计算Z分数
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)
    
    if std_val == 0:
        return np.zeros_like(data, dtype=bool)
    
    z_scores = np.abs((data - mean_val) / std_val)
    outliers = z_scores > threshold
    
    return outliers


def fix_data_quality_issues(data_dict, data_type):
    """修复数据质量问题"""
    print(f"修复{data_type}数据质量问题...")

    data = data_dict['data'].copy()
    n_fixes = 0

    if data_type == 'pet':
        # 修复PET负值问题
        negative_mask = data < 0
        data[negative_mask] = 0.0  # 将负值设为0
        n_fixes += np.sum(negative_mask)
        print(f"   修复PET负值: {n_fixes} 个")

    elif data_type == 'runoff':
        # 修复径流数据问题
        # 1. 处理极端异常值
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            q99 = np.percentile(valid_data, 99)
            extreme_mask = data > q99 * 10  # 超过99分位数10倍的视为极端异常
            data[extreme_mask] = np.nan
            n_fixes += np.sum(extreme_mask)

        # 2. 处理负值
        negative_mask = data < 0
        data[negative_mask] = 0.0
        n_fixes += np.sum(negative_mask)
        print(f"   修复径流异常值: {n_fixes} 个")

    elif data_type == 'precip':
        # 修复降水负值
        negative_mask = data < 0
        data[negative_mask] = 0.0
        n_fixes += np.sum(negative_mask)
        if n_fixes > 0:
            print(f"   修复降水负值: {n_fixes} 个")

    # 更新数据字典
    processed_dict = data_dict.copy()
    processed_dict['data'] = data

    return processed_dict


def preprocess_data(data_dict, remove_outliers=True):
    """预处理数据"""
    print("预处理数据...")

    data = data_dict['data'].copy()

    if remove_outliers:
        n_outliers = 0
        for i in range(data.shape[1]):
            station_data = data[:, i]
            outliers = detect_outliers(station_data)
            data[outliers, i] = np.nan
            n_outliers += np.sum(outliers)

        print(f"   移除异常值: {n_outliers} 个")

    # 更新数据字典
    processed_dict = data_dict.copy()
    processed_dict['data'] = data

    return processed_dict


def load_all_csv_data(data_dir=None):
    """加载所有CSV数据文件"""
    print("📂 加载所有CSV数据文件...")

    # 如果没有指定数据目录，尝试多个可能的路径
    if data_dir is None:
        possible_data_paths = [
            r"D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理",
            r".\data",
            r".\美国已处理",
            r".\数据",
            r".",  # 当前目录
        ]

        print("🔍 在以下路径中查找数据文件:")
        for path in possible_data_paths:
            print(f"   - {path}")

        # 尝试每个路径
        for test_dir in possible_data_paths:
            if os.path.exists(test_dir):
                print(f"   - 检查路径: {test_dir}")
                # 检查是否所有文件都存在
                all_files_exist = True
                for data_type, filename in CSV_FILES.items():
                    file_path = os.path.join(test_dir, filename)
                    if not os.path.exists(file_path):
                        all_files_exist = False
                        break

                if all_files_exist:
                    print(f"   - 在 {test_dir} 找到所有数据文件")
                    data_dir = test_dir
                    break
                else:
                    print(f"   - {test_dir} 中缺少部分文件")

        if data_dir is None:
            print("在所有路径中都未找到完整的数据文件")
            return None

    data_files = {}
    file_paths = {}

    # 检查文件存在性
    for data_type, filename in CSV_FILES.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            file_paths[data_type] = file_path
            print(f"   - 找到 {data_type}: {filename}")
        else:
            print(f"   - 未找到 {data_type}: {filename}")

    if len(file_paths) == 0:
        print("未找到任何数据文件")
        return None
    
    # 加载数据并修复质量问题
    for data_type, file_path in file_paths.items():
        try:
            data_dict = load_csv_forcing_data(file_path)
            # 修复数据质量问题
            data_dict = fix_data_quality_issues(data_dict, data_type)
            data_files[data_type] = data_dict
            print(f"   - {data_type}数据加载成功")
        except Exception as e:
            print(f"   - {data_type}数据加载失败: {e}")
    
    return data_files


def find_common_stations(data_files):
    """找到所有数据文件中的共同站点"""
    print("🔍 查找共同站点...")
    
    if not data_files:
        return []
    
    # 获取所有数据类型的站点名称
    station_sets = []
    for data_type, data_dict in data_files.items():
        stations = set(data_dict['stations']['station_name'].values)
        station_sets.append(stations)
        print(f"   {data_type}: {len(stations)} 个站点")
    
    # 找交集
    common_stations = set.intersection(*station_sets)
    common_stations = list(common_stations)
    
    print(f"   - 共同站点: {len(common_stations)} 个")
    
    return common_stations


def filter_data_by_stations(data_files, common_stations):
    """根据共同站点过滤数据"""
    print("根据共同站点过滤数据...")
    
    filtered_data = []
    
    for data_type, data_dict in data_files.items():
        stations_df = data_dict['stations']
        data_array = data_dict['data']
        
        # 找到共同站点的索引
        station_indices = []
        filtered_stations = []
        
        for station_name in common_stations:
            mask = stations_df['station_name'] == station_name
            if mask.any():
                idx = stations_df[mask].index[0]
                original_idx = stations_df.index.get_loc(idx)
                station_indices.append(original_idx)
                filtered_stations.append(stations_df.loc[idx])
        
        # 过滤数据
        filtered_data_array = data_array[:, station_indices]
        filtered_stations_df = pd.DataFrame(filtered_stations).reset_index(drop=True)
        
        filtered_dict = {
            'data': filtered_data_array,
            'dates': data_dict['dates'],
            'stations': filtered_stations_df
        }
        
        filtered_data.append(filtered_dict)
        print(f"   - {data_type}: {filtered_data_array.shape[1]} 个站点")
    
    return filtered_data


def validate_time_consistency(data_files):
    """验证时间序列一致性"""
    print("🕒 验证时间序列一致性...")
    
    if not data_files:
        return False
    
    # 获取第一个数据文件的日期作为参考
    reference_dates = None
    reference_type = None
    
    for data_type, data_dict in data_files.items():
        if reference_dates is None:
            reference_dates = data_dict['dates']
            reference_type = data_type
            break
    
    # 检查所有数据文件的日期是否一致
    all_consistent = True
    for data_type, data_dict in data_files.items():
        dates = data_dict['dates']
        if not dates.equals(reference_dates):
            print(f"   - {data_type}的时间序列与{reference_type}不一致")
            all_consistent = False
        else:
            print(f"   - {data_type}时间序列一致")
    
    if all_consistent:
        print(f"   - 所有数据文件时间序列一致: {len(reference_dates)} 天")
        print(f"   时间范围: {reference_dates[0].strftime('%Y-%m-%d')} 到 {reference_dates[-1].strftime('%Y-%m-%d')}")
    
    return all_consistent


def get_data_summary(data_files):
    """获取数据摘要信息"""
    print("\n数据摘要:")
    
    for data_type, data_dict in data_files.items():
        data = data_dict['data']
        stations = data_dict['stations']
        dates = data_dict['dates']
        
        print(f"\n   {data_type.upper()}数据:")
        print(f"     站点数: {len(stations)}")
        print(f"     时间长度: {len(dates)} 天")
        print(f"     数据范围: {np.nanmin(data):.2f} - {np.nanmax(data):.2f}")
        print(f"     平均值: {np.nanmean(data):.2f}")
        print(f"     缺失值: {np.isnan(data).sum()} ({np.isnan(data).sum()/data.size:.1%})")


def prepare_model_data(data_files, common_stations):
    """准备模型输入数据"""
    print("准备模型输入数据...")
    
    # 过滤数据
    filtered_data = filter_data_by_stations(data_files, common_stations)
    
    # 验证数据顺序
    data_types = ['precip', 'temp', 'pet', 'runoff']
    if len(filtered_data) != len(data_types):
        print(f"数据文件数量不匹配: 期望{len(data_types)}个，实际{len(filtered_data)}个")
        return None, None
    
    # 构建强迫数据和观测数据
    forcing_data = {
        'precip': filtered_data[0]['data'],
        'temp': filtered_data[1]['data'],
        'pet': filtered_data[2]['data'],
        'stations': filtered_data[0]['stations'],
        'dates': filtered_data[0]['dates']
    }
    
    obs_data = {
        'data': filtered_data[3]['data']
    }
    
    print(f"模型数据准备完成:")
    print(f"   站点数: {len(common_stations)}")
    print(f"   时间长度: {forcing_data['precip'].shape[0]} 天")
    
    return forcing_data, obs_data
