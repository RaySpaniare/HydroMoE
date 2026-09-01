#!/usr/bin/env python3
"""
HydroPy配置模块
包含所有常量、参数范围和系统配置
"""

import multiprocessing as mp

# ======================================================================================================
# 系统配置
# ======================================================================================================

# CPU配置
CPU_COUNT = mp.cpu_count()
OPTIMAL_PROCESSES = max(1, CPU_COUNT - 2)

# 依赖库可用性检查
try:
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("WARNING: sklearn not available, R² calculation will be skipped")
    r2_score = None
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNING: scipy not available, lag correction will be limited")
    pearsonr = None
    SCIPY_AVAILABLE = False

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    print("WARNING: CMA-ES not available, will use random search")
    CMA_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("WARNING: scikit-optimize not available, will skip Bayesian optimization")
    BAYESIAN_AVAILABLE = False

# ======================================================================================================
# 参数优化配置
# ======================================================================================================

# 静态参数边界定义
PARAMETER_BOUNDS = {
    # 土壤水文参数（核心静态参数，逐站点得到单值；下列为优化搜索边界）
    'wcap': (50.0, 6000.0),     # 土壤持水容量 [mm]
    'wava': (10.0, 5000.0),     # 土壤有效水分 [mm]
    'wmin': (0.0, 1000.0),      # 最小土壤水分 [mm]
    'wmax': (200.0, 8000.0),    # 最大土壤水分 [mm]
    'beta': (0.1, 5.0),         # ARNO模型β参数 [/]

    # 土地覆盖参数（逐站点静态）
    'fveg': (0.0, 1.0),         # 植被覆盖比例 [/]
    'fbare': (0.0, 1.0),        # 裸土比例 [/]
    'flake': (0.0, 0.7),        # 湖泊/水面比例 [/]

    # 植被结构
    'lai_annual': (0.0, 8.0),   # 年叶面积指数 [/]

    # PET与观测校正
    'pet_correction': (0.1, 5.0),   # PET校正系数 [/]
    'runoff_correction': (0.2, 3.0),# 径流校正系数 [/]

    # 蒸散发过程关键参数
    'transp_fraction': (0.4, 1.0),  # 蒸腾分配比例 [/]
    'et_alpha': (0.5, 5.0),         # 蒸散发总体系数 [/]
    'sevap_alpha': (0.5, 4.0),      # 土壤蒸发系数 [/]

    # 土壤水分阈值
    'rm_crit': (0.2, 0.9),          # 临界土壤水分比例 [/]
    'sevap_low': (0.01, 0.5),       # 土壤蒸发下限比例 [/]

    # 地下水过程参数
    'groundwater_recession': (2.0, 120.0),  # 地下水衰减系数 [day]
    'baseflow_threshold': (0.01, 0.8),      # 基流阈值 [/]
    'gw_recharge_rate': (0.05, 0.95),      # 地下水补给率 [/]

    # 时序-路由（机理内部简单纠正，可选）
    'internal_lag_days': (0.0, 365.0),      # 模型输出整天滞后 [day]
    'routing_k_days': (0.0, 60.0),          # 线性水库路由时间常数 [day]
}

# （已移除）扩展“科学参数边界”集合，避免与当前简化机理混淆

# 固定模型参数（不参与优化）
FIXED_MODEL_PARAMS = {
    # 雪过程参数
    'rainf_lower': -1.1,      # 降雨下限温度 [°C]
    'snowf_upper': 3.3,       # 降雪上限温度 [°C]
    'melt_crit': 0.0,         # 融雪临界温度 [°C]
    'frc_liquid': 0.06,       # 雪中液态水比例 [/]
    # 土壤过程参数
    'qsb_min': 1.15741e-05,   # 最小排水参数 [kg m-2 s-1] - 约1.0 mm/day
    'qsb_max': 1.15741e-04,   # 最大排水参数 [kg m-2 s-1] - 约10.0 mm/day
    'qsb_exp': 1.5,           # ECHAM排水指数 [/]
    'qsb_low': 0.05,          # 排水最小土壤湿度 [/]
    'qsb_hig': 0.90,          # 排水最大土壤湿度 [/]
    'sevap_low': 0.05,        # 土壤蒸发最小土壤湿度 [/]
    'rm_crit': 0.75,          # 临界根区土壤湿度比例 [/]
    # 地下水参数
    'initial_groundwater': 10.0,        # 初始地下水储量
    'groundwater_retention_time': 30.0, # 地下水滞留时间
    # ARNO模型参数
    'beta_e': 0.45,           # PET修正因子
}

# ======================================================================================================
# 优化算法配置
# ======================================================================================================

# CMA-ES配置
CMAES_CONFIG = {
    'max_evaluations': 2000,     # 每个站点最大评估次数（显著增加）
    'population_size_factor': 100,# 种群大小因子（显著增加）
    'sigma_factor': 0.01,         # 初始步长因子（更小→更细致搜索）
    'tolerance': 1e-10,          # 收敛容差（更严格）
}

# 随机搜索配置
RANDOM_SEARCH_CONFIG = {
    'max_evaluations': 200,      # 每个站点最大评估次数
    'n_samples': 100,            # 随机样本数量
}

# 时序纠偏配置
LAG_CORRECTION_CONFIG = {
    'max_lag_days': 365,      # 最大滞后天数（不再使用复杂分析）
}

# ======================================================================================================
# 数据处理配置
# ======================================================================================================

# CSV数据文件名
CSV_FILES = {
    'precip': r'D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理\美国降水.csv',
    'temp':   r'D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理\美国温度.csv',
    'pet':    r'D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理\美国潜在蒸发.csv',
    'runoff': r'D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理\美国径流.csv'
}

# 数据验证配置
DATA_VALIDATION = {
    'min_valid_points': 100,     # 最少有效数据点
    'max_missing_ratio': 0.3,    # 最大缺失比例
    'outlier_threshold': 5.0,    # 异常值阈值（标准差倍数）
}

# ======================================================================================================
# 结果管理配置
# ======================================================================================================

# 文件名配置
RESULT_FILES = {
    'optimization_results': 'cmaes_optimal_params.json',
    'backup_results': 'cmaes_optimal_params_backup.json',
    'log_file': 'hydropy_optimization.log'
}

# 质量评估阈值
QUALITY_THRESHOLDS = {
    'excellent': 0.7,    # R² > 0.7
    'good': 0.5,         # 0.5 < R² <= 0.7
    'fair': 0.3,         # 0.3 < R² <= 0.5
    'poor': 0.1,         # 0.1 < R² <= 0.3
    'failed': 0.0        # R² <= 0.1
}

# 统计报告配置
STATISTICS_CONFIG = {
    'r2_thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'distribution_intervals': [
        (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
    ]
}

# （已移除）默认参数配置：强制从真实数据读取经纬度与时间信息

# ======================================================================================================
# 调试和日志配置
# ======================================================================================================

# 日志级别
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# 调试配置
DEBUG_CONFIG = {
    'verbose_optimization': False,  # 详细优化输出
    'save_intermediate': False,     # 保存中间结果
    'plot_results': False,          # 绘制结果图表
}

# 进度显示配置
PROGRESS_CONFIG = {
    'use_tqdm': True,            # 使用进度条
    'update_frequency': 10,      # 更新频率
    'show_eta': True,            # 显示预计完成时间
}
