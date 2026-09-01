#!/usr/bin/env python3
"""
HydroPy主程序入口
整合所有模块，提供统一的程序接口
"""

import sys
import os

# 导入所有模块
from config import CSV_FILES, OPTIMAL_PROCESSES, CPU_COUNT
from data_processing import (
    load_all_csv_data, find_common_stations, validate_time_consistency,
    get_data_summary, prepare_model_data
)
from optimization import optimize_stations_with_cmaes, optimize_on_test_and_export
from results_manager import save_optimization_results, print_optimization_summary, export_simulated_runoff_csv


def print_welcome_message():
    """打印欢迎信息"""
    print("HydroPy CSV核心版本 - 模块化智能优化版本")
    print("=" * 60)
    print("核心功能:")
    print("   - 模块化架构，易于维护和扩展")
    print("   - 智能加载已有优化结果")
    print("   - 时序纠偏分析和校正")
    print("   - 使用CMA-ES优化连续静态参数")
    print("   - CPU多进程并行加速")
    print("   - 专注美国流域数据适配")
    print("=" * 60)


def print_parameter_info():
    """打印参数信息"""
    print("\nCMA-ES优化的连续静态参数包括:")
    print("  - 土壤水文参数: wcap(20-1200), wava(10-800), wmin(1-300), wmax(50-1500), beta(0.01-10)")
    print("   土地覆盖参数: fveg(0-0.95), fbare(0.05-1.0), flake(0-0.7)")
    print("  - 地形参数: slope_avg(0.0001-1.5), topo_std(0.01-2000), perm(0-1.0), lai_annual(0-10)")
    print("  - 校正系数: pet_correction(0.001-1.0), runoff_correction(0.2-3.0)")
    print("   多年冻土参数: perm(0.0-0.8)")
    print("   植被参数: lai_annual(0.5-5.0)")


def print_usage_info():
    """打印使用方法"""
    print("\n使用方法:")
    print("  1. 准备CSV数据文件:")
    for data_type, filename in CSV_FILES.items():
        print(f"     - {filename}")
    print("  2. 运行程序:")
    print("     python main.py")
    print("  3. 直接运行全时段优化（已集成机理内部时序纠正）")


def get_user_choice():
    """兼容保留，但不再询问，统一返回False（走完整优化）。"""
    return False


def load_and_validate_data():
    """加载和验证数据"""
    print("\n📂 开始加载数据...")

    # 加载所有CSV数据
    data_files = load_all_csv_data()
    if not data_files:
        print("数据加载失败")
        print("\n解决方案:")
        print("   1. 确保以下CSV文件存在于当前目录或指定路径:")
        for data_type, filename in CSV_FILES.items():
            print(f"      - {filename}")
        print("   2. 或者运行以下命令生成测试数据:")
        print("      python generate_test_data.py")
        print("   3. 或者将数据文件放在以下任一路径:")
        print("      - ./data/")
        print("      - ./美国已处理/")
        print("      - ./数据/")
        return None, None, None
    
    # 验证时间序列一致性
    if not validate_time_consistency(data_files):
        print("时间序列不一致，程序退出")
        return None, None, None
    
    # 找到共同站点
    common_stations = find_common_stations(data_files)
    if len(common_stations) == 0:
        print("未找到共同站点，程序退出")
        return None, None, None
    
    # 获取数据摘要
    get_data_summary(data_files)
    
    # 准备模型数据
    forcing_data, obs_data = prepare_model_data(data_files, common_stations)
    if forcing_data is None or obs_data is None:
        print("模型数据准备失败，程序退出")
        return None, None, None
    
    return forcing_data, obs_data, common_stations


def run_optimization(forcing_data, obs_data, use_smart_mode=False):
    """在测试集(自2008-起)直接运行优化并导出结果"""
    print(f"\n系统配置:")
    print(f"   CPU核心数: {CPU_COUNT}")
    print(f"   并行进程数: {OPTIMAL_PROCESSES}")

    # 直接在测试集执行优化，并导出达标站点的全时段模拟与测试指标
    print("\n开始在测试集(>=2008-01-01)上优化并导出...")
    results, metrics_df = optimize_on_test_and_export(
        forcing_data,
        obs_data,
        test_start_year=2008,
        r2_threshold=0.3,
        metrics_output_path='output/test_metrics.csv',
        simulated_runoff_output_path='output/simulated_runoff_full.csv',
        max_evaluations=150
    )

    return results


def save_and_summarize_results(optimization_results):
    """保存和总结结果"""
    if not optimization_results:
        print("没有优化结果可保存")
        return
    
    # 保存结果（按用户要求使用备份文件名与格式）
    print("\n保存优化结果...")
    save_optimization_results(optimization_results, filename='cmaes_optimal_params备份.json', max_evaluations=150)
    
    # 打印结果摘要
    print_optimization_summary(optimization_results)

    # 导出全时段模拟与测试集指标已在优化过程中完成
    
    # 统计信息
    r2_values = [result['best_r2'] for result in optimization_results.values()]
    successful_count = sum(1 for r2 in r2_values if r2 > 0.1)
    
    print(f"\n优化完成!")
    print(f"   总站点数: {len(optimization_results)}")
    print(f"   成功优化: {successful_count} 个站点")
    print(f"   成功率: {successful_count/len(optimization_results)*100:.1f}%")
    print(f"   平均R²: {sum(r2_values)/len(r2_values):.4f}")


def main():
    """主函数"""
    try:
        # 打印欢迎信息
        print_welcome_message()
        print_parameter_info()
        print_usage_info()
        
        # 获取用户选择
        use_smart_mode = get_user_choice()
        
        # 加载和验证数据
        forcing_data, obs_data, common_stations = load_and_validate_data()
        if forcing_data is None:
            return 1
        
        print(f"\n数据加载完成:")
        print(f"   共同站点数: {len(common_stations)}")
        print(f"   时间序列长度: {forcing_data['precip'].shape[0]} 天")
        
        # 运行优化
        optimization_results = run_optimization(forcing_data, obs_data, use_smart_mode)
        
        # 保存和总结结果
        save_and_summarize_results(optimization_results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
        return 1
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    """程序入口点"""
    exit_code = main()
    sys.exit(exit_code)
