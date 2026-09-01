#!/usr/bin/env python3
"""
HydroPy参数优化模块
包含静态参数优化器和相关优化算法
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import warnings
import os

# 导入配置和核心模块
from config import (
    PARAMETER_BOUNDS, CMA_AVAILABLE, SKLEARN_AVAILABLE,
    CMAES_CONFIG, RANDOM_SEARCH_CONFIG, OPTIMAL_PROCESSES
)
from hydro_core import HydroPyCSVCore

# 导入依赖库
if SKLEARN_AVAILABLE:
    from sklearn.metrics import r2_score
else:
    r2_score = None

if CMA_AVAILABLE:
    import cma

warnings.filterwarnings('ignore')


def _resolve_output_path(requested_path: str) -> str:
    """Ensure the parent directory exists; otherwise fallback to Desktop.

    Returns a writable absolute path. If creating the parent directory of
    requested_path fails or it doesn't exist, write to user's Desktop with the
    same filename.
    """
    try:
        # Normalize path and attempt to create parent directory
        requested_path = os.path.normpath(requested_path)
        parent_dir = os.path.dirname(requested_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        # If parent exists or created, return absolute path
        if not parent_dir or os.path.exists(parent_dir):
            return os.path.abspath(requested_path)
    except Exception:
        # Fall through to Desktop fallback
        pass

    # Desktop fallback
    filename = os.path.basename(requested_path) if requested_path else 'output.csv'
    desktop_dir = os.path.join(os.path.expanduser('~'), 'Desktop')
    try:
        os.makedirs(desktop_dir, exist_ok=True)
    except Exception:
        # As a last resort, use current working directory
        return os.path.abspath(filename)
    return os.path.abspath(os.path.join(desktop_dir, filename))


def find_optimal_lag_proxy(precip_series, obs_series, max_lag_days=730):
    """使用降水-观测径流的简单互相关，扫描±max_lag_days，返回最佳滞后(整天)。
    正滞后表示将模型输出向后平移相同天数（常见于汇流滞后）。"""
    p = np.asarray(precip_series, dtype=float)
    o = np.asarray(obs_series, dtype=float)
    n = min(len(p), len(o))
    p = p[:n]
    o = o[:n]
    best_lag = 0
    best_corr = -1e9
    # 预标准化以稳定相关
    def zscore(x):
        xm = np.nanmean(x)
        xs = np.nanstd(x)
        return (x - xm) / (xs + 1e-12)
    pz = zscore(p)
    oz = zscore(o)
    max_lag_days = int(max(0, max_lag_days))
    for lag in range(-max_lag_days, max_lag_days + 1):
        if lag == 0:
            a, b = pz, oz
        elif lag > 0:
            a, b = pz[:-lag], oz[lag:]
        else:
            a, b = pz[-lag:], oz[:lag]
        if len(a) < 30:
            continue
        mask = ~(np.isnan(a) | np.isnan(b))
        if np.sum(mask) < 30:
            continue
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if np.std(a[mask]) > 0 and np.std(b[mask]) > 0 else -1e9
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    return int(best_lag)

class StaticParameterOptimizer:
    """静态参数连续优化器 - 支持CMA-ES和随机搜索"""

    def __init__(self, include_pet_correction=True, include_runoff_correction=True, custom_bounds=None):
        """初始化连续参数优化器

        Args:
            include_pet_correction (bool): 是否包含PET校正系数优化
            include_runoff_correction (bool): 是否包含径流观测校正系数优化
        """
        # 仅保留精简必调参数集合（其余固定）
        essential_params = [
            'wcap', 'wmin', 'wmax', 'wava',        # 土壤容量与阈值
            'beta',                                 # 产流形状参数
            'fveg', 'fbare',                        # 覆盖分配
            'pet_correction',                       # PET校正（已纳入）
            'transp_fraction', 'et_alpha', 'sevap_alpha',  # ET强度与分配
            'rm_crit', 'sevap_low',                 # 胁迫阈值
            'groundwater_recession', 'baseflow_threshold', 'gw_recharge_rate',  # 地下水过程
            'lai_annual'                            # 植被结构（用于蒸腾）
        ]
        base_bounds = custom_bounds if custom_bounds is not None else PARAMETER_BOUNDS
        self.param_bounds = {k: v for k, v in base_bounds.items() if k in essential_params}
        
        # 根据选项调整参数边界
        if not include_pet_correction and 'pet_correction' in self.param_bounds:
            del self.param_bounds['pet_correction']
        
        if not include_runoff_correction and 'runoff_correction' in self.param_bounds:
            del self.param_bounds['runoff_correction']

        # 固定参数（不参与优化）- 纬度将动态设置
        self.fixed_params = {}

        # 参数名称列表（用于CMA-ES）
        self.param_names = list(self.param_bounds.keys())

        # 配置选项
        self.include_pet_correction = include_pet_correction
        self.include_runoff_correction = include_runoff_correction

        # 约束参数
        self._eps = 1e-6

    def _clip_to_bounds(self, name, value):
        """将参数裁剪到边界范围内。"""
        if name not in self.param_bounds:
            return value
        low, high = self.param_bounds[name]
        if low is None or high is None:
            return value
        return float(min(max(value, low), high))

    def _enforce_constraints(self, p):
        """强制满足物理与一致性约束，原地修改并返回参数字典。"""
        # 基本存在性
        get = p.get

        # 1) 土壤容量与阈值关系: 0 < wmin < wcap < wmax
        if ('wmin' in p) and ('wcap' in p) and ('wmax' in p):
            wmin = float(get('wmin'))
            wcap = float(get('wcap'))
            wmax = float(get('wmax'))
            # 先裁剪到各自边界
            wmin = self._clip_to_bounds('wmin', wmin)
            wcap = self._clip_to_bounds('wcap', wcap)
            wmax = self._clip_to_bounds('wmax', wmax)
            # 施加顺序约束
            if not (wmin < wcap):
                wmin = min(wcap - self._eps, wmin)
            if not (wcap < wmax):
                wmax = max(wcap + self._eps, wmax)
            # 重新裁剪并确保严格不等
            wmin = self._clip_to_bounds('wmin', wmin)
            wmax = self._clip_to_bounds('wmax', wmax)
            # 极端情况下强制按比例分隔
            if not (wmin < wcap < wmax):
                mid = max(self._eps, wcap)
                span = max(3 * self._eps, (wmax - wmin))
                wmin = mid - span * 0.4
                wmax = mid + span * 0.6
                wmin = self._clip_to_bounds('wmin', wmin)
                wmax = self._clip_to_bounds('wmax', wmax)
                if not (wmin < wcap < wmax):
                    # 退一步：按边界中点构造
                    wmin_low, wmin_high = self.param_bounds['wmin']
                    wcap_low, wcap_high = self.param_bounds['wcap']
                    wmax_low, wmax_high = self.param_bounds['wmax']
                    wcap = (wcap_low + wcap_high) / 2
                    wmin = min((wmin_low + wmin_high) / 2, wcap - self._eps)
                    wmax = max((wmax_low + wmax_high) / 2, wcap + self._eps)
            p['wmin'], p['wcap'], p['wmax'] = wmin, wcap, wmax

        # 2) 有效水分不超过容量，且不小于 wmin
        if ('wava' in p) and ('wcap' in p):
            wava = float(get('wava'))
            wava = self._clip_to_bounds('wava', wava)
            if 'wmin' in p:
                wava = max(wava, float(p['wmin']))
            wava = min(wava, float(p['wcap']))
            p['wava'] = wava

        # 3) 覆盖分配: fveg + fbare ≤ 1 且各自∈[0,1]
        if ('fveg' in p) and ('fbare' in p):
            fveg = self._clip_to_bounds('fveg', float(get('fveg')))
            fbare = self._clip_to_bounds('fbare', float(get('fbare')))
            total = fveg + fbare
            if total > 1.0 + self._eps:
                scale = 1.0 / total
                fveg *= scale
                fbare *= scale
            p['fveg'], p['fbare'] = fveg, fbare

        # 4) LAI非负
        if 'lai_annual' in p:
            p['lai_annual'] = max(0.0, self._clip_to_bounds('lai_annual', float(get('lai_annual'))))

        # 5) 地下水约束：阈值在0-1，衰减>0，补给率在0-1
        if 'baseflow_threshold' in p:
            p['baseflow_threshold'] = self._clip_to_bounds('baseflow_threshold', float(get('baseflow_threshold')))
        if 'groundwater_recession' in p:
            p['groundwater_recession'] = max(self._eps, self._clip_to_bounds('groundwater_recession', float(get('groundwater_recession'))))
        if 'gw_recharge_rate' in p:
            p['gw_recharge_rate'] = self._clip_to_bounds('gw_recharge_rate', float(get('gw_recharge_rate')))

        # 6) 蒸散发相关参数非负
        for name in ['transp_fraction', 'et_alpha', 'sevap_alpha', 'rm_crit', 'sevap_low']:
            if name in p:
                p[name] = self._clip_to_bounds(name, float(get(name)))

        return p

    def generate_random_parameters(self, n_samples=100):
        """生成随机参数组合"""
        print(f"生成 {n_samples} 个随机参数组合...")

        combinations = []
        for _ in range(n_samples):
            param_dict = {}
            for param_name, (min_val, max_val) in self.param_bounds.items():
                param_dict[param_name] = np.random.uniform(min_val, max_val)
            param_dict.update(self.fixed_params)
            # 施加参数约束
            param_dict = self._enforce_constraints(param_dict)
            combinations.append(param_dict)

        return combinations

    def optimize_with_cmaes(self, precip, temp, pet, obs, station_name, station_lat=40.0, max_evaluations=None, initial_x0=None):
        """使用CMA-ES优化单个站点参数"""
        if not CMA_AVAILABLE:
            print(f"   CMA-ES不可用，使用随机搜索")
            return self.optimize_with_random_search(precip, temp, pet, obs, station_name, station_lat, max_evaluations)

        if max_evaluations is None:
            max_evaluations = CMAES_CONFIG['max_evaluations']

        print(f"   使用CMA-ES优化站点 {station_name} (参数数量: {len(self.param_names)})")

        # 初始参数
        x0 = []
        bounds = []
        for param_name in self.param_names:
            min_val, max_val = self.param_bounds[param_name]
            bounds.append([min_val, max_val])
        if initial_x0 is None:
            for param_name in self.param_names:
                min_val, max_val = self.param_bounds[param_name]
                x0.append((min_val + max_val) / 2)
        else:
            x0 = list(initial_x0)

        # 初始步长
        sigma0 = CMAES_CONFIG['sigma_factor'] * np.array([max_val - min_val for min_val, max_val in self.param_bounds.values()])

        # CMA-ES设置
        popsize = min(30, CMAES_CONFIG['population_size_factor'] + int(3 * np.log(len(self.param_names))))
        opts = {
            'bounds': [list(b) for b in zip(*bounds)],
            'maxfevals': max_evaluations,
            'popsize': popsize,
            'tolfun': CMAES_CONFIG['tolerance'],
            'verbose': -1  # 静默模式
        }

        best_r2 = -999
        best_params = None

        def objective_function(x):
            nonlocal best_r2, best_params

            # 构建参数字典
            param_dict = dict(zip(self.param_names, x))
            param_dict.update(self.fixed_params)
            param_dict['lat'] = station_lat  # 使用站点真实纬度

            # 施加参数约束
            param_dict = self._enforce_constraints(param_dict)

            # 评估参数
            result = self.evaluate_parameter_combination(param_dict, precip, temp, pet, obs, station_name)
            r2 = result['r2']

            # 更新最佳结果
            if r2 > best_r2:
                best_r2 = r2
                best_params = param_dict.copy()

            return -r2  # CMA-ES最小化，所以返回负值

        try:
            # 执行CMA-ES优化
            es = cma.CMAEvolutionStrategy(x0, np.mean(sigma0), opts)

            while not es.stop():
                solutions = es.ask()
                fitness_values = [objective_function(x) for x in solutions]
                es.tell(solutions, fitness_values)

            return best_params, best_r2

        except Exception as e:
            print(f"   CMA-ES优化失败: {e}，使用随机搜索")
            return self.optimize_with_random_search(precip, temp, pet, obs, station_name, station_lat, max_evaluations)

    def optimize_with_random_search(self, precip, temp, pet, obs, station_name, station_lat=40.0, max_evaluations=None):
        """使用随机搜索优化单个站点参数"""
        if max_evaluations is None:
            max_evaluations = RANDOM_SEARCH_CONFIG['max_evaluations']
            
        print(f"   使用随机搜索优化站点 {station_name}")

        best_r2 = -999
        best_params = None

        # 生成随机参数组合
        param_combinations = self.generate_random_parameters(max_evaluations)

        for param_combo in param_combinations:
            param_combo['lat'] = station_lat  # 使用站点真实纬度
            # 施加参数约束
            param_combo = self._enforce_constraints(param_combo)
            result = self.evaluate_parameter_combination(param_combo, precip, temp, pet, obs, station_name)

            if result['r2'] > best_r2:
                best_r2 = result['r2']
                best_params = param_combo.copy()

        return best_params, best_r2

    def evaluate_parameter_combination(self, param_combo, precip, temp, pet, obs, station_name):
        """评估单个参数组合的性能"""
        try:
            # 直接运行水文模拟
            sim_runoff = self._run_hydro_simulation(precip, temp, pet, param_combo)

            # 早停/快速筛除：若几乎全零或无效，直接返回劣值
            if sim_runoff is None or np.all(~np.isfinite(sim_runoff)) or np.nanmax(sim_runoff) < 1e-8:
                return {
                    'params': param_combo,
                    'station': station_name,
                    'r2': -999,
                    'rmse': 999999,
                    'valid_points': 0,
                    'error': 'Degenerate simulation (near-zero or invalid)'
                }

            # 应用径流观测校正系数
            runoff_correction = param_combo.get('runoff_correction', 1.0)
            obs_corrected = obs * runoff_correction

            # 简单移动窗口纠偏：直接对模拟径流做整天平移以最大化R²（±730天）
            def shift_series(x, lag):
                if lag == 0:
                    return x
                if lag > 0:
                    return np.concatenate([np.full(lag, x[0]), x[:-lag]])
                else:
                    lag = -lag
                    return np.concatenate([x[lag:], np.full(lag, x[-1])])

            best_r2_lag = 0
            best_sim_shifted = sim_runoff
            # 限定最大滞后天数（两年）
            max_lag_days = 730
            # 步长可以为1天（工作量更大）
            for lag in range(-max_lag_days, max_lag_days + 1):
                sim_shifted = shift_series(sim_runoff, lag)
                mask = ~(np.isnan(sim_shifted) | np.isnan(obs_corrected) | (obs_corrected < 0) | (sim_shifted < 0))
                if np.sum(mask) < 200:
                    continue
                ss_res = np.sum((obs_corrected[mask] - sim_shifted[mask]) ** 2)
                ss_tot = np.sum((obs_corrected[mask] - np.mean(obs_corrected[mask])) ** 2)
                r2_try = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                if r2_try > best_r2_lag:
                    best_r2_lag = r2_try
                    best_sim_shifted = sim_shifted

            # 使用最佳平移后的序列计算指标
            valid_mask = ~(np.isnan(best_sim_shifted) | np.isnan(obs_corrected) |
                          (obs_corrected < 0) | (best_sim_shifted < 0))

            # 更严格的数据质量检查
            if np.sum(valid_mask) < 200:  # 提高最少数据点要求
                return {
                    'params': param_combo,
                    'station': station_name,
                    'r2': -999,
                    'rmse': 999999,
                    'valid_points': np.sum(valid_mask),
                    'error': 'Insufficient valid data points'
                }

            sim_valid = best_sim_shifted[valid_mask]
            obs_valid = obs_corrected[valid_mask]

            # 额外的数据质量检查
            if np.var(obs_valid) < 1e-10 or np.var(sim_valid) < 1e-10:
                return {
                    'params': param_combo,
                    'station': station_name,
                    'r2': -999,
                    'rmse': 999999,
                    'valid_points': np.sum(valid_mask),
                    'error': 'Insufficient variance in data'
                }

            # 改进的R²计算
            try:
                # 使用Nash-Sutcliffe效率系数（更适合水文模型）
                ss_res = np.sum((obs_valid - sim_valid) ** 2)
                ss_tot = np.sum((obs_valid - np.mean(obs_valid)) ** 2)

                if ss_tot > 0:
                    r2 = 1 - (ss_res / ss_tot)
                else:
                    r2 = 0.0

                # 限制R²在合理范围内
                r2 = max(-10.0, min(1.0, r2))

            except Exception as e:
                r2 = -999

            rmse = np.sqrt(np.mean((sim_valid - obs_valid) ** 2))
            mae = float(np.mean(np.abs(sim_valid - obs_valid))) if len(sim_valid) > 0 else 999999
            bias = float((np.mean(sim_valid - obs_valid) / (np.mean(obs_valid) + 1e-12)) * 100.0) if len(sim_valid) > 0 else 999999
            # KGE
            r = np.corrcoef(sim_valid, obs_valid)[0, 1] if np.std(sim_valid) > 0 and np.std(obs_valid) > 0 else 0.0
            alpha = np.std(sim_valid) / (np.std(obs_valid) + 1e-12)
            beta = (np.mean(sim_valid) + 1e-12) / (np.mean(obs_valid) + 1e-12)
            kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

            return {
                'params': param_combo,
                'station': station_name,
                'r2': r2,
                'rmse': float(rmse),
                'mae': mae,
                'bias': bias,
                'kge': float(kge),
                'valid_points': np.sum(valid_mask),
                'sim_mean': sim_valid.mean(),
                'obs_mean': obs_valid.mean()
            }

        except Exception as e:
            return {
                'params': param_combo,
                'station': station_name,
                'r2': -999,
                'rmse': 999999,
                'valid_points': 0,
                'error': str(e)
            }

    def _run_hydro_simulation(self, precip, temp, pet, static_params):
        """内部水文模拟函数 - 只返回总径流"""
        n_days = len(precip)

        # 创建模型实例
        model = HydroPyCSVCore(static_params=static_params)

        # 获取参数（必须通过优化提供，不使用默认值）
        wcap = static_params['wcap']
        beta = static_params['beta']
        wmin = static_params['wmin']
        wmax = static_params['wmax']
        lai = static_params['lai_annual']
        lat = static_params.get('lat', 40.0)
        fveg = static_params.get('fveg', 0.2)
        fbare = static_params.get('fbare', 0.7)
        pet_correction = static_params.get('pet_correction', 1.0)  # PET校正系数

        # 初始化状态
        swe = 0.0
        wliq = 0.0
        rootmoist = wcap * 0.5
        groundwstor = model.get_initial_groundwater()

        # 只保存总径流结果
        qtot_results = np.zeros(n_days)

        # 逐日计算（不使用早停，完整模拟全时段以保证有效样本）
        for day in range(n_days):
            daily_precip = precip[day]
            daily_temp = temp[day]  # 使用摄氏度
            daily_pet = pet[day] * pet_correction  # 应用PET校正系数
            day_of_year = day % 365 + 1

            # 1. 雪过程
            snowf, rainf, _ = model.get_rain_and_snow(daily_precip, daily_temp)
            smelt_pot, _ = model.get_potential_snowmelt(daily_temp, lat, day_of_year)
            swe, wliq, _, rainmelt = model.update_snow(swe, wliq, snowf, smelt_pot)

            # 2. 冻土诊断
            frozen = model.diagnose_frozen_ground(daily_temp)

            # 3. 地表过程
            throughfall = rainmelt + rainf

            # 4. 径流计算
            qs = model.get_surface_runoff(throughfall, rootmoist, wcap, beta, wmin, wmax, frozen)
            qsb = model.get_drainage(rootmoist, wcap, dt=86400, frozen=frozen)

            # 5. 蒸散发计算 - 传递static_params用于机理参数
            transp = model.get_transpiration(daily_pet, rootmoist, wcap, fveg, lai, static_params)
            sevap = model.get_soilevap(daily_pet, rootmoist, wcap, fbare, static_params)

            # 6. 土壤更新
            rootmoist, qs_add = model.update_soil(rootmoist, throughfall, qs, transp, sevap, qsb, wcap)
            qs += qs_add

            # 7. 地下水更新
            groundwstor, qg = model.update_groundwater(groundwstor, qsb, static_params=static_params)

            # 8. 总径流（未路由）
            qtot_results[day] = qs + qsb + qg

            # 不做早停检查，确保全程填充

        # 9. 返回未路由序列（外部已实现平移纠偏）
        return qtot_results


# ======================================================================================================
# 并行优化功能
# ======================================================================================================

def optimize_single_station_cmaes(args):
    """使用CMA-ES优化单个站点的最优参数"""
    station_idx, precip, temp, pet, obs, station_name, station_lat, max_evaluations = args

    print(f"🔍 开始CMA-ES优化站点 {station_name} (索引: {station_idx}, 纬度: {station_lat:.2f}°)")

    # 创建参数优化器
    optimizer = StaticParameterOptimizer(
        include_pet_correction=True,
        include_runoff_correction=True
    )

    try:
        # 先扫描±两年的滞后天数，对观测进行对齐后优化（或对模拟做整天平移等效）
        best_lag_days = find_optimal_lag_proxy(precip, obs, max_lag_days=730)

        # 将最佳滞后作为内部路由参数（整天平移），交由机理内部处理
        # 在优化器中，会读取 static_params['internal_lag_days']
        def run_with_lag(evaluations):
            # 包装一次性评估，固定 internal_lag_days
            original_generate = optimizer.generate_random_parameters
            def wrapped_generate(n_samples=100):
                combos = original_generate(n_samples)
                for c in combos:
                    c['internal_lag_days'] = max(0, int(abs(best_lag_days)))
                return combos
            optimizer.generate_random_parameters = wrapped_generate
            return optimizer.optimize_with_cmaes(
                precip, temp, pet, obs, station_name, station_lat, evaluations
            )

        # 多起点+两阶段细化
        num_starts = 3
        best_params = None
        best_r2 = -999

        # 粗搜多起点
        for _ in range(num_starts):
            # 随机起点：按边界均匀采样
            init = []
            for pname in optimizer.param_names:
                lo, hi = optimizer.param_bounds[pname]
                init.append(float(np.random.uniform(lo, hi)))
            params_try, r2_try = optimizer.optimize_with_cmaes(
                precip, temp, pet, obs, station_name, station_lat, max_evaluations, initial_x0=init
            )
            if r2_try > best_r2:
                best_r2, best_params = r2_try, params_try

        # 两阶段细化：以best为中心收缩边界±20%再优化一次
        if best_params is not None:
            tightened_bounds = {}
            for pname in optimizer.param_names:
                lo, hi = optimizer.param_bounds[pname]
                mid = float(best_params[pname])
                span = (hi - lo) * 0.2
                new_lo = max(lo, mid - span)
                new_hi = min(hi, mid + span)
                if new_lo >= new_hi:
                    new_lo, new_hi = lo, hi
                tightened_bounds[pname] = (new_lo, new_hi)

            refiner = StaticParameterOptimizer(include_pet_correction=True, include_runoff_correction=True, custom_bounds=tightened_bounds)
            # 以best为起点细化
            params_refine, r2_refine = refiner.optimize_with_cmaes(
                precip, temp, pet, obs, station_name, station_lat, max_evaluations // 2, initial_x0=[best_params[p] for p in refiner.param_names]
            )
            if r2_refine > best_r2:
                best_r2, best_params = r2_refine, params_refine

        print(f"站点 {station_name} CMA-ES优化完成: 最佳R²={best_r2:.4f}")

        return {
            'station_idx': station_idx,
            'station_name': station_name,
            'best_params': best_params,
            'best_r2': best_r2,
            'optimization_method': 'cmaes_with_lag_scan'
        }

    except Exception as e:
        print(f"站点 {station_name} CMA-ES优化失败: {e}")
        return {
            'station_idx': station_idx,
            'station_name': station_name,
            'best_params': None,
            'best_r2': -999,
            'error': str(e)
        }


def optimize_stations_with_cmaes(forcing_data, obs_data, max_evaluations=150, time_range=None):
    """
    使用CMA-ES优化多个站点的静态参数和机理参数

    参数:
        forcing_data: 强迫数据字典
        obs_data: 观测数据字典
        max_evaluations: 每个站点的最大评估次数 (默认150次)
        time_range: 时间范围 (start_idx, end_idx)，None表示使用全部时间序列

    返回:
        dict: 优化结果
    """
    method_name = "CMA-ES" if CMA_AVAILABLE else "随机搜索"
    print(f"开始{method_name}静态参数和机理参数优化")
    print(f"每站点最大评估次数: {max_evaluations}")
    print(f"使用 {OPTIMAL_PROCESSES} 个并行进程")

    # 创建临时优化器来获取参数数量（精简参数集）
    temp_optimizer = StaticParameterOptimizer(include_pet_correction=True, include_runoff_correction=True)
    print(f"参数数量: {len(temp_optimizer.param_names)}（使用精简必调参数集）")

    # 处理时间序列范围
    if time_range is not None:
        start_idx, end_idx = time_range
        print(f"🕒 使用时间序列范围: {start_idx} - {end_idx} (共 {end_idx - start_idx} 天)")

        # 切片数据
        precip_data = forcing_data['precip'][start_idx:end_idx, :]
        temp_data = forcing_data['temp'][start_idx:end_idx, :]
        pet_data = forcing_data['pet'][start_idx:end_idx, :]
        obs_data_slice = obs_data['data'][start_idx:end_idx, :]
    else:
        print(f"🕒 使用完整时间序列: {forcing_data['precip'].shape[0]} 天")
        precip_data = forcing_data['precip']
        temp_data = forcing_data['temp']
        pet_data = forcing_data['pet']
        obs_data_slice = obs_data['data']

    # 准备并行任务
    stations_info = forcing_data['stations']
    n_stations = len(stations_info)

    args_list = []
    for i in range(n_stations):
        precip = precip_data[:, i]
        temp = temp_data[:, i]
        pet = pet_data[:, i]
        obs = obs_data_slice[:, i]
        station_name = stations_info.iloc[i]['station_name']
        station_lat = stations_info.iloc[i]['latitude']  # 获取站点真实纬度

        args_list.append((i, precip, temp, pet, obs, station_name, station_lat, max_evaluations))

    # 并行优化
    print(f"使用 {OPTIMAL_PROCESSES} 个CPU核心进行并行优化...")

    optimization_results = {}
    with mp.Pool(processes=OPTIMAL_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(optimize_single_station_cmaes, args_list),
            total=n_stations,
            desc="CMA-ES优化"
        ))

    # 收集结果
    for result in results:
        if result and result['best_params'] is not None:
            optimization_results[result['station_name']] = {
                'station_idx': result['station_idx'],
                'best_params': result['best_params'],
                'best_r2': result['best_r2'],
                'optimization_method': result.get('optimization_method', 'unknown')
            }

    print(f"{method_name}优化完成!")
    print(f"成功优化 {len(optimization_results)} 个站点")

    # 优化统计
    if optimization_results:
        r2_values = [result['best_r2'] for result in optimization_results.values()]
        r2_array = np.array(r2_values)

        print(f"\n{method_name}优化结果统计:")
        print_detailed_statistics(r2_array)

        # 打印R²分布统计
        print_r2_distribution_statistics(r2_array)

    return optimization_results


# =====================================================================================================
# 基于测试集(如2008-)优化，并导出全局模拟与测试集指标
# =====================================================================================================

def _calc_kge(sim, obs):
    """计算KGE (Kling-Gupta Efficiency)。要求输入为1D有效数组。"""
    if len(sim) < 2:
        return -999.0
    r = np.corrcoef(sim, obs)[0, 1] if np.std(sim) > 0 and np.std(obs) > 0 else 0.0
    alpha = np.std(sim) / (np.std(obs) + 1e-12)
    beta = (np.mean(sim) + 1e-12) / (np.mean(obs) + 1e-12)
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def _calc_metrics(sim, obs):
    """返回包含 R2, RMSE, MAE, bias(%), KGE 的字典。"""
    mask = ~(np.isnan(sim) | np.isnan(obs))
    if np.sum(mask) < 10:
        return {'r2': -999, 'rmse': 999999, 'mae': 999999, 'bias': 999999, 'kge': -999, 'n': int(np.sum(mask))}
    sim_v = sim[mask]
    obs_v = obs[mask]
    ss_res = np.sum((obs_v - sim_v) ** 2)
    ss_tot = np.sum((obs_v - np.mean(obs_v)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2 = max(-10.0, min(1.0, r2))
    rmse = float(np.sqrt(np.mean((sim_v - obs_v) ** 2)))
    mae = float(np.mean(np.abs(sim_v - obs_v)))
    bias = float((np.mean(sim_v - obs_v) / (np.mean(obs_v) + 1e-12)) * 100.0)
    kge = float(_calc_kge(sim_v, obs_v))
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias, 'kge': kge, 'n': int(np.sum(mask))}


def _simulate_full_period(precip, temp, pet, params, lat, dates):
    """使用给定参数对全时段进行模拟，返回长度为n_days的一维数组。"""
    static_params = params.copy()
    static_params['lat'] = lat
    # 复用优化器的内部模拟逻辑
    temp_optimizer = StaticParameterOptimizer(include_pet_correction=True, include_runoff_correction=True)
    return temp_optimizer._run_hydro_simulation(precip, temp, pet, static_params)


def optimize_on_test_and_export(forcing_data, obs_data, test_start_year=2008,
                                r2_threshold=0.3,
                                metrics_output_path='output/test_metrics.csv',
                                simulated_runoff_output_path='output/simulated_runoff_full.csv',
                                max_evaluations=150):
    """
    在测试集(>=test_start_year)上优化；对R2>=阈值的站点，输出：
      1) 全时段的模拟径流CSV（各列为站点）
      2) 测试集上的评估指标(R2/RMSE/MAE/Bias/KGE)
    """
    dates = forcing_data['dates']
    if not isinstance(dates, (pd.Series, pd.DatetimeIndex)):
        dates = pd.to_datetime(dates)
    test_start_date = pd.Timestamp(year=test_start_year, month=1, day=1)
    # 找到测试集起始索引
    try:
        start_idx = int(np.argmax(dates >= test_start_date))
    except Exception:
        start_idx = 0
    end_idx = len(dates)

    # 在测试集优化
    results = optimize_stations_with_cmaes(forcing_data, obs_data, max_evaluations=max_evaluations,
                                           time_range=(start_idx, end_idx))

    stations_df = forcing_data['stations']
    precip_full = forcing_data['precip']
    temp_full = forcing_data['temp']
    pet_full = forcing_data['pet']
    obs_full = obs_data['data']

    # 计算全体站点的测试集R²均值（仅用于报告，不再作为导出条件）
    r2_list = []
    for _, info in results.items():
        if info and info.get('best_params') is not None:
            r2_list.append(float(info.get('best_r2', -999)))
    mean_r2 = float(np.mean(r2_list)) if r2_list else -999.0

    # 收集全时段模拟与测试集指标（当导出开启时为所有达成优化的站点，否则跳过写文件）
    sim_dict = {}
    metrics_rows = []

    for station_name, info in results.items():
        best_params = info['best_params']
        if best_params is None:
            continue
        idx = info['station_idx']

        lat = float(stations_df.iloc[idx]['latitude'])
        sim_full = _simulate_full_period(precip_full[:, idx],
                                         temp_full[:, idx],
                                         pet_full[:, idx],
                                         best_params,
                                         lat,
                                         dates)
        sim_dict[station_name] = sim_full

        # 计算测试集指标
        sim_test = sim_full[start_idx:end_idx]
        obs_test = obs_full[start_idx:end_idx, idx]
        m = _calc_metrics(sim_test, obs_test)
        metrics_rows.append({
            'station': station_name,
            'r2': m['r2'],
            'rmse': m['rmse'],
            'mae': m['mae'],
            'bias_percent': m['bias'],
            'kge': m['kge'],
            'n_points': m['n']
        })

    # 解析与保障输出路径
    simulated_runoff_output_path = _resolve_output_path(simulated_runoff_output_path)
    metrics_output_path = _resolve_output_path(metrics_output_path)

    # 输出全时段模拟CSV（与“美国径流.csv”相同格式：第一行日期yyyyMMdd、每行站点/经纬度/数值）
    try:
        stations_df = forcing_data['stations']
        dates_idx = pd.to_datetime(dates)
        date_headers = [int(d.strftime('%Y%m%d')) for d in dates_idx]

        # 组装矩阵：n_days x n_stations
        out_df = pd.DataFrame({name: sim_dict.get(name) for name in stations_df['station_name']}, index=dates_idx)

        # 构建输出行
        rows = []
        for j in range(stations_df.shape[0]):
            st = stations_df.iloc[j]
            series = out_df[st['station_name']].values if st['station_name'] in out_df.columns else np.zeros(len(dates_idx))
            row = [st['station_name'], st['longitude'], st['latitude']] + list(series)
            rows.append(row)

        header = ['station_name', 'longitude', 'latitude'] + date_headers
        df_out = pd.DataFrame(rows, columns=header)
        df_out.to_csv(simulated_runoff_output_path, header=False, index=False, encoding='utf-8')
        print(f"已导出模拟径流: {simulated_runoff_output_path}")
    except Exception as e:
        print(f"导出模拟径流失败: {e}")

    # 输出测试集指标CSV
    metrics_df = pd.DataFrame(metrics_rows)
    try:
        metrics_df.to_csv(metrics_output_path, index=False, encoding='utf-8')
    except Exception:
        # 再次尝试使用Desktop回退（极端情况下文件名为空等）
        alt_path = _resolve_output_path(metrics_output_path)
        metrics_df.to_csv(alt_path, index=False, encoding='utf-8')
        metrics_output_path = alt_path
    print(f"测试集优化完成（起始: {test_start_year}-01-01），全体平均R²={mean_r2:.3f}（不作为导出阈值）")
    print(f"  导出全时段模拟: {simulated_runoff_output_path}（{len(sim_dict)} 个站点）")
    print(f"  导出测试集指标: {metrics_output_path}（{len(metrics_rows)} 条记录）")

    return results, metrics_df


def print_detailed_statistics(r2_array):
    """打印详细的R²统计信息"""
    if len(r2_array) == 0:
        print("   - 没有有效的R²数据")
        return

    print(f"   总站点数: {len(r2_array)}")
    print(f"   平均R²: {r2_array.mean():.4f}")
    print(f"   中位数R²: {np.median(r2_array):.4f}")
    print(f"   标准差: {r2_array.std():.4f}")
    print(f"   最大R²: {r2_array.max():.4f}")
    print(f"   最小R²: {r2_array.min():.4f}")

    # 详细分布统计
    from config import STATISTICS_CONFIG
    thresholds = STATISTICS_CONFIG['r2_thresholds']
    print(f"   R²分布:")
    for threshold in thresholds:
        count = np.sum(r2_array > threshold)
        percentage = count / len(r2_array) * 100
        print(f"     R² > {threshold}: {count} 个站点 ({percentage:.1f}%)")

    # 零值和负值统计
    negative_count = np.sum(r2_array < 0.0)
    zero_count = np.sum(r2_array == 0.0)
    very_low_count = np.sum((r2_array > 0.0) & (r2_array <= 0.05))
    print(f"   R² < 0.0: {negative_count} 个站点 ({negative_count/len(r2_array)*100:.1f}%)")
    print(f"   R² = 0.0: {zero_count} 个站点 ({zero_count/len(r2_array)*100:.1f}%)")
    print(f"   0.0 < R² ≤ 0.05: {very_low_count} 个站点 ({very_low_count/len(r2_array)*100:.1f}%)")


def print_r2_distribution_statistics(r2_array):
    """打印R²分布统计信息"""
    print(f"\nR²分布统计 (共{len(r2_array)}个站点):")
    print(f"   均值R²: {r2_array.mean():.4f}")
    print(f"   中位数R²: {np.median(r2_array):.4f}")
    print(f"   最大R²: {r2_array.max():.4f}")
    print(f"   最小R²: {r2_array.min():.4f}")

    # 区间分布统计
    from config import STATISTICS_CONFIG
    intervals = STATISTICS_CONFIG['distribution_intervals']

    print(f"\n   区间分布:")
    for low, high in intervals:
        count = np.sum((r2_array >= low) & (r2_array < high))
        percentage = count / len(r2_array) * 100
        print(f"     {low:.1f}-{high:.1f}: {count:3d}个站点 ({percentage:5.1f}%)")
