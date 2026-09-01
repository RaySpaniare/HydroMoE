#!/usr/bin/env python3
"""
HydroPy结果管理模块
包含结果保存、加载和统计分析功能
"""

import os
import json
import pandas as pd
import numpy as np

# 导入配置
from config import (
    RESULT_FILES, QUALITY_THRESHOLDS, STATISTICS_CONFIG,
    OPTIMAL_PROCESSES, CPU_COUNT, CMA_AVAILABLE
)
from optimization import StaticParameterOptimizer
def _resolve_output_path(requested_path: str) -> str:
    """Ensure parent directory exists or fallback to Desktop with same filename."""
    try:
        requested_path = os.path.normpath(requested_path)
        parent_dir = os.path.dirname(requested_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        if not parent_dir or os.path.exists(parent_dir):
            return os.path.abspath(requested_path)
    except Exception:
        pass
    filename = os.path.basename(requested_path) if requested_path else 'output.csv'
    desktop_dir = os.path.join(os.path.expanduser('~'), 'Desktop')
    try:
        os.makedirs(desktop_dir, exist_ok=True)
    except Exception:
        return os.path.abspath(filename)
    return os.path.abspath(os.path.join(desktop_dir, filename))



def get_quality_level(r2_value):
    """根据R²值判断优化质量等级"""
    if r2_value > QUALITY_THRESHOLDS['excellent']:
        return "excellent"
    elif r2_value > QUALITY_THRESHOLDS['good']:
        return "good"
    elif r2_value > QUALITY_THRESHOLDS['fair']:
        return "fair"
    elif r2_value > QUALITY_THRESHOLDS['poor']:
        return "poor"
    else:
        return "failed"


def save_optimization_results(optimization_results, filename=None, max_evaluations=150):
    """保存优化结果，包含时序纠偏信息和完整的优化详情"""
    if filename is None:
        filename = RESULT_FILES['optimization_results']
    
    try:
        # 准备保存的数据结构
        save_data = {
            'optimization_summary': {
                'total_stations': len(optimization_results),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_method': 'CMA-ES with Lag Correction',
                'lag_correction_enabled': True,
                'cpu_cores_used': OPTIMAL_PROCESSES,
                'max_cpu_cores': CPU_COUNT,
                'optimization_config': {
                    'max_evaluations_per_station': max_evaluations,
                    'cma_es_available': CMA_AVAILABLE,
                    'parameter_count': len(StaticParameterOptimizer(include_pet_correction=True, include_runoff_correction=True).param_names),
                    'includes_mechanism_params': True,
                    'includes_pet_correction': True,
                    'includes_runoff_correction': True
                }
            },
            'station_results': {}
        }

        # 计算最终统计信息
        r2_values = [result['best_r2'] for result in optimization_results.values()]
        r2_array = np.array(r2_values)

        save_data['optimization_summary']['statistics'] = {
            'mean_r2': float(r2_array.mean()),
            'median_r2': float(np.median(r2_array)),
            'std_r2': float(r2_array.std()),
            'max_r2': float(r2_array.max()),
            'min_r2': float(r2_array.min()),
            'r2_distribution': {
                f'r2_gt_{threshold}': int(np.sum(r2_array > threshold))
                for threshold in STATISTICS_CONFIG['r2_thresholds']
            },
            'quality_assessment': {
                'excellent_stations': int(np.sum(r2_array > QUALITY_THRESHOLDS['excellent'])),
                'good_stations': int(np.sum((r2_array > QUALITY_THRESHOLDS['good']) & (r2_array <= QUALITY_THRESHOLDS['excellent']))),
                'fair_stations': int(np.sum((r2_array > QUALITY_THRESHOLDS['fair']) & (r2_array <= QUALITY_THRESHOLDS['good']))),
                'poor_stations': int(np.sum((r2_array > QUALITY_THRESHOLDS['poor']) & (r2_array <= QUALITY_THRESHOLDS['fair']))),
                'failed_stations': int(np.sum(r2_array <= QUALITY_THRESHOLDS['poor']))
            }
        }

        # 保存每个站点的详细结果
        lag_corrected_count = 0
        total_improvement = 0.0
        successful_optimizations = 0
        failed_optimizations = 0

        for station_name, result in optimization_results.items():
            # 确保所有数值都是可序列化的类型
            best_params = {}
            for key, value in result['best_params'].items():
                if isinstance(value, (np.integer, np.floating)):
                    best_params[key] = float(value)
                elif isinstance(value, (int, float)):
                    best_params[key] = value
                else:
                    best_params[key] = str(value)

            # 基础站点数据
            station_data = {
                'station_idx': int(result['station_idx']),
                'best_r2': float(result['best_r2']),
                'optimization_method': str(result['optimization_method']),
                'best_params': best_params,
                'optimization_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'parameter_count': len(best_params),
                'quality_level': get_quality_level(float(result['best_r2'])),
                'optimization_success': float(result['best_r2']) > QUALITY_THRESHOLDS['poor']
            }

            # 统计成功/失败的优化
            if float(result['best_r2']) > QUALITY_THRESHOLDS['poor']:
                successful_optimizations += 1
            else:
                failed_optimizations += 1

            # 如果有错误信息，也保存
            if 'error' in result:
                station_data['error'] = str(result['error'])
                station_data['optimization_success'] = False

            # 如果有时序纠偏信息，详细保存
            if 'lag_correction' in result:
                lag_info = result['lag_correction']
                station_data['lag_correction'] = {
                    'lag_days': int(lag_info['lag_days']),
                    'original_r2': float(lag_info['original_r2']),
                    'corrected_r2': float(lag_info['corrected_r2']),
                    'improvement': float(lag_info['improvement']),
                    'has_improvement': bool(float(lag_info['improvement']) > 0.01),
                    'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # 统计纠偏效果
                if float(lag_info['improvement']) > 0:
                    lag_corrected_count += 1
                    total_improvement += float(lag_info['improvement'])

            save_data['station_results'][station_name] = station_data

        # 添加时序纠偏汇总统计
        if lag_corrected_count > 0:
            save_data['optimization_summary']['lag_correction_summary'] = {
                'total_analyzed_stations': int(len([r for r in optimization_results.values() if 'lag_correction' in r])),
                'improved_stations': int(lag_corrected_count),
                'average_improvement': float(total_improvement / lag_corrected_count),
                'total_improvement': float(total_improvement),
                'improvement_rate': float(lag_corrected_count / len(optimization_results) * 100)
            }

        # 添加优化成功率统计
        save_data['optimization_summary']['optimization_performance'] = {
            'successful_optimizations': successful_optimizations,
            'failed_optimizations': failed_optimizations,
            'success_rate': float(successful_optimizations / len(optimization_results) * 100),
            'total_processed': len(optimization_results),
            'average_r2': float(r2_array.mean()),
            'optimization_efficiency': {
                'stations_per_core': float(len(optimization_results) / OPTIMAL_PROCESSES),
                'estimated_total_evaluations': len(optimization_results) * max_evaluations,
                'parameter_space_size': len(StaticParameterOptimizer(include_pet_correction=True, include_runoff_correction=True).param_names)
            }
        }

        # 保存到文件（覆盖原文件）
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"优化结果已保存到: {filename}")
            print(f"   包含 {len(optimization_results)} 个站点的详细参数")

            # 验证文件完整性
            with open(filename, 'r', encoding='utf-8') as f:
                test_load = json.load(f)
                # 验证关键字段是否存在
                assert 'optimization_summary' in test_load
                assert 'station_results' in test_load
                assert len(test_load['station_results']) == len(optimization_results)
            print(f"JSON文件完整性验证通过，包含 {len(test_load['station_results'])} 个站点数据")

        except Exception as save_error:
            print(f"保存JSON文件失败: {save_error}")
            # 尝试保存备份文件
            backup_filename = RESULT_FILES['backup_results']
            try:
                with open(backup_filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                print(f"已保存备份文件: {backup_filename}")
            except Exception as backup_error:
                print(f"保存备份文件也失败: {backup_error}")

    except Exception as e:
        print(f"保存优化结果失败: {e}")
        import traceback
        traceback.print_exc()


def load_optimization_results(filename=None):
    """加载优化结果"""
    if filename is None:
        filename = RESULT_FILES['optimization_results']
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                optimization_results = json.load(f)
            print(f"加载优化结果: {filename}")
            return optimization_results
        else:
            print(f"优化结果文件不存在: {filename}")
            return {}
    except Exception as e:
        print(f"加载优化结果失败: {e}")
        return {}


def print_optimization_summary(optimization_results):
    """打印优化结果摘要"""
    if not optimization_results:
        print("没有优化结果可显示")
        return
    
    print("\n" + "="*60)
    print("优化结果摘要")
    print("="*60)


def export_simulated_runoff_csv(forcing_data, optimization_results, output_path):
    """根据最优参数为所有站点生成模拟径流CSV，格式与“美国径流.csv”一致。

    要求forcing_data包含：'stations' (DataFrame: station_name, longitude, latitude), 'dates' (DatetimeIndex)
    optimization_results: { station_name: { best_params, best_r2, ... } }
    """
    try:
        import pandas as pd
        import numpy as np
        from optimization import StaticParameterOptimizer

        stations_df = forcing_data['stations']
        dates = forcing_data.get('dates')
        precip = forcing_data['precip']
        temp = forcing_data['temp']
        pet = forcing_data['pet']

        # 构建输出表头：第一行前三列空/标签，后续为YYYYMMDD整数
        date_headers = [int(d.strftime('%Y%m%d')) for d in dates]

        # 逐站模拟
        optimizer = StaticParameterOptimizer()
        n_days = precip.shape[0]
        n_stations = stations_df.shape[0]
        out_matrix = np.zeros((n_days, n_stations), dtype=float)

        name_to_idx = {stations_df.iloc[i]['station_name']: i for i in range(n_stations)}

        for station_name, result in optimization_results.items():
            if station_name not in name_to_idx:
                continue
            j = name_to_idx[station_name]
            best_params = result['best_params']
            # 取各自列的强迫
            qsim = optimizer._run_hydro_simulation(
                precip[:, j], temp[:, j], pet[:, j], best_params
            )
            out_matrix[:, j] = qsim

        # 组装DataFrame为导出格式
        header_row = ['station_name', 'longitude', 'latitude'] + date_headers
        rows = []
        for j in range(n_stations):
            row = [
                stations_df.iloc[j]['station_name'],
                stations_df.iloc[j]['longitude'],
                stations_df.iloc[j]['latitude'],
            ] + list(out_matrix[:, j])
            rows.append(row)

        df_out = pd.DataFrame(rows, columns=header_row)
        # Ensure output path exists or fallback to Desktop
        safe_path = _resolve_output_path(output_path)
        df_out.to_csv(safe_path, header=False, index=False, encoding='utf-8')
        print(f"已导出模拟径流: {safe_path}")
    except Exception as e:
        print(f"导出模拟径流失败: {e}")
    
    # 基本统计
    if 'optimization_summary' in optimization_results:
        summary = optimization_results['optimization_summary']
        print(f"总站点数: {summary.get('total_stations', 0)}")
        print(f"优化时间: {summary.get('timestamp', 'Unknown')}")
        print(f"优化方法: {summary.get('optimization_method', 'Unknown')}")
        
        if 'statistics' in summary:
            stats = summary['statistics']
            print(f"\nR²统计:")
            print(f"  平均值: {stats.get('mean_r2', 0):.4f}")
            print(f"  中位数: {stats.get('median_r2', 0):.4f}")
            print(f"  最大值: {stats.get('max_r2', 0):.4f}")
            print(f"  最小值: {stats.get('min_r2', 0):.4f}")
        
        if 'quality_assessment' in summary:
            quality = summary['quality_assessment']
            print(f"\n质量评估:")
            print(f"  优秀 (R²>0.7): {quality.get('excellent_stations', 0)} 个站点")
            print(f"  良好 (0.5<R²≤0.7): {quality.get('good_stations', 0)} 个站点")
            print(f"  一般 (0.3<R²≤0.5): {quality.get('fair_stations', 0)} 个站点")
            print(f"  较差 (0.1<R²≤0.3): {quality.get('poor_stations', 0)} 个站点")
            print(f"  失败 (R²≤0.1): {quality.get('failed_stations', 0)} 个站点")
    
    print("="*60)
