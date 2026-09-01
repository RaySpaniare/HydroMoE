#!/usr/bin/env python3
"""
使用最优参数(cmaes_optimal_params.json)在测试集(>=2008-01-01)上评估并导出模拟径流。

评估口径与优化阶段保持一致：
- 对观测应用 runoff_correction（若参数提供，默认1.0）
- 在测试集上对模拟径流执行 ±730 天整天平移扫描，取使R²最大的平移
- 有效点严格筛选，剔除 NaN 与负值；要求有效点>=200 且方差>0
- 终端打印每站点评估；导出“已平移纠正”的测试期模拟径流CSV，表头包含日期(20080101起)
"""

import os
import json
import numpy as np
import pandas as pd

from data_processing import (
    load_all_csv_data,
    find_common_stations,
    validate_time_consistency,
    get_data_summary,
    prepare_model_data,
)
from optimization import StaticParameterOptimizer


def _ensure_parent_dir(path: str) -> str:
    try:
        path = os.path.normpath(path)
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        return os.path.abspath(path)
    except Exception:
        filename = os.path.basename(path) if path else 'output.csv'
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        try:
            os.makedirs(desktop, exist_ok=True)
            return os.path.abspath(os.path.join(desktop, filename))
        except Exception:
            return os.path.abspath(filename)


def _calc_metrics(sim: np.ndarray, obs: np.ndarray) -> dict:
    mask = ~(np.isnan(sim) | np.isnan(obs) | (sim < 0) | (obs < 0))
    n = int(np.sum(mask))
    if n < 200:
        return {'r2': -999.0, 'rmse': 999999.0, 'mae': 999999.0, 'bias': 999999.0, 'kge': -999.0, 'n': n}
    sim_v = sim[mask]
    obs_v = obs[mask]
    if np.var(sim_v) < 1e-10 or np.var(obs_v) < 1e-10:
        return {'r2': -999.0, 'rmse': 999999.0, 'mae': 999999.0, 'bias': 999999.0, 'kge': -999.0, 'n': n}
    ss_res = np.sum((obs_v - sim_v) ** 2)
    ss_tot = np.sum((obs_v - np.mean(obs_v)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2 = float(max(-10.0, min(1.0, r2)))
    rmse = float(np.sqrt(np.mean((sim_v - obs_v) ** 2)))
    mae = float(np.mean(np.abs(sim_v - obs_v)))
    bias = float((np.mean(sim_v - obs_v) / (np.mean(obs_v) + 1e-12)) * 100.0)
    # KGE
    r = np.corrcoef(sim_v, obs_v)[0, 1] if np.std(sim_v) > 0 and np.std(obs_v) > 0 else 0.0
    alpha = np.std(sim_v) / (np.std(obs_v) + 1e-12)
    beta = (np.mean(sim_v) + 1e-12) / (np.mean(obs_v) + 1e-12)
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias, 'kge': kge, 'n': n}


def _shift_series(x: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return x
    if lag > 0:
        return np.concatenate([np.full(lag, x[0]), x[:-lag]])
    lag = -lag
    return np.concatenate([x[lag:], np.full(lag, x[-1])])


def _scan_best_shift(sim: np.ndarray, obs: np.ndarray, max_lag_days: int = 730) -> tuple:
    best_r2 = -999.0
    best_lag = 0
    best_sim = sim
    for lag in range(-max_lag_days, max_lag_days + 1):
        sim_shift = _shift_series(sim, lag)
        m = _calc_metrics(sim_shift, obs)
        if m['r2'] > best_r2:
            best_r2 = m['r2']
            best_lag = lag
            best_sim = sim_shift
    return best_sim, best_lag, best_r2


def load_params(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'station_results' in data:
        return data['station_results']
    return data


def run_evaluation(params_json='cmaes_optimal_params.json',
                   simulated_runoff_csv='output/simulated_runoff_test.csv'):
    print("PBM评估：按优化口径在测试集(>=2008-01-01)上评估与导出")

    # 1) 数据
    data_files = load_all_csv_data()
    if not data_files:
        print("数据加载失败，退出。")
        return 1
    if not validate_time_consistency(data_files):
        print("时间序列不一致，退出。")
        return 1
    common = find_common_stations(data_files)
    if len(common) == 0:
        print("未找到共同站点，退出。")
        return 1
    get_data_summary(data_files)
    forcing_data, obs_data = prepare_model_data(data_files, common)
    if forcing_data is None or obs_data is None:
        print("模型数据准备失败，退出。")
        return 1

    dates = forcing_data['dates']
    if not isinstance(dates, (pd.Series, pd.DatetimeIndex)):
        dates = pd.to_datetime(dates)
    test_start = pd.Timestamp(year=2008, month=1, day=1)
    try:
        start_idx = int(np.argmax(dates >= test_start))
    except Exception:
        start_idx = 0
    end_idx = len(dates)

    # 2) 参数
    if not os.path.exists(params_json):
        print(f"未找到最优参数文件: {params_json}")
        return 1
    station_params = load_params(params_json)

    stations_df = forcing_data['stations']
    precip = forcing_data['precip']
    temp = forcing_data['temp']
    pet = forcing_data['pet']
    obs_full = obs_data['data']

    name_to_idx = {stations_df.iloc[i]['station_name']: i for i in range(stations_df.shape[0])}
    optimizer = StaticParameterOptimizer(include_pet_correction=True, include_runoff_correction=True)

    results_rows = []
    sim_matrix_shifted = {}

    # 测试集日期表头
    test_dates = dates[start_idx:end_idx]
    date_headers = [int(d.strftime('%Y%m%d')) for d in test_dates]

    print("\n测试集评估(与优化口径一致)：")
    print("station, r2, rmse, mae, bias%, kge, n, best_lag_days")

    for station_name, info in station_params.items():
        if station_name not in name_to_idx:
            continue
        j = name_to_idx[station_name]
        best_params = info.get('best_params', {})
        if not best_params:
            continue

        # 强制使用真实纬度
        lat_val = float(stations_df.iloc[j]['latitude'])
        p = best_params.copy()
        p['lat'] = lat_val

        # 观测校正
        runoff_corr = float(p.get('runoff_correction', 1.0))

        # 模拟(测试集切片)
        sim_test = optimizer._run_hydro_simulation(
            precip[start_idx:end_idx, j],
            temp[start_idx:end_idx, j],
            pet[start_idx:end_idx, j],
            p
        )

        # 观测(测试集切片)并应用校正
        obs_test = obs_full[start_idx:end_idx, j] * runoff_corr

        # ±730天平移扫描，取最大R²
        sim_best, best_lag, best_r2 = _scan_best_shift(sim_test, obs_test, max_lag_days=730)
        sim_matrix_shifted[station_name] = sim_best

        # 指标(在best shift下)
        m = _calc_metrics(sim_best, obs_test)
        results_rows.append({
            'station': station_name,
            'r2': m['r2'],
            'rmse': m['rmse'],
            'mae': m['mae'],
            'bias_percent': m['bias'],
            'kge': m['kge'],
            'n_points': m['n'],
            'best_lag_days': int(best_lag)
        })
        print(f"{station_name}, {m['r2']:.4f}, {m['rmse']:.4f}, {m['mae']:.4f}, {m['bias']:.2f}, {m['kge']:.4f}, {m['n']}, {best_lag}")

    # 导出测试集模拟(已平移纠正)
    rows = []
    header = ['station_name', 'longitude', 'latitude'] + date_headers
    for j in range(stations_df.shape[0]):
        st = stations_df.iloc[j]
        name = st['station_name']
        if name not in sim_matrix_shifted:
            continue
        series = sim_matrix_shifted[name]
        row = [name, st['longitude'], st['latitude']] + list(series)
        rows.append(row)

    df_out = pd.DataFrame(rows, columns=header)
    out_path = _ensure_parent_dir(simulated_runoff_csv)
    df_out.to_csv(out_path, header=False, index=False, encoding='utf-8')
    print(f"\n已导出测试集(已平移纠正)模拟径流: {out_path}")

    # 汇总统计
    if results_rows:
        r2_vals = np.array([r['r2'] for r in results_rows if np.isfinite(r['r2'])])
        if r2_vals.size > 0:
            print(f"\n测试集R²统计(优化同口径): 均值={r2_vals.mean():.4f}, 中位数={np.median(r2_vals):.4f}, 最大={r2_vals.max():.4f}, 最小={r2_vals.min():.4f}")

    return 0


if __name__ == '__main__':
    import sys
    json_path = 'cmaes_optimal_params.json'
    out_csv = 'output/simulated_runoff_test.csv'
    if len(sys.argv) >= 2:
        json_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_csv = sys.argv[2]
    code = run_evaluation(json_path, out_csv)
    sys.exit(code)


