"""
低值站点增强模块（新建）
目标：
1) 从 cmaes_optimal_params.json 读取站点 best_r2，标记 R² < 0.2 的低值站点
2) 基于原始长表 CSV 离线生成“滞后/滚动”的径流特征（不使用当日 y(t)）
3) 提供独立管道：生成增强CSV → 构造数据加载器 → 训练/评估（可选）

注意：不修改现有大文件；通过新模块独立运行或被主程序调用。
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional


def compute_low_r2_stations(cmaes_json_path: str, threshold: float = 0.2) -> Set[str]:
    """
    从 cmaes_optimal_params.json 中筛选低R²站点。

    兼容两种结构：
      1) 旧版: { 'camels_XXXX': { 'best_r2': ... }, ... }
      2) 新版: { 'optimization_summary': {...}, 'station_results': { sid: {'best_r2': ...}, ... } }
    """
    low_set: Set[str] = set()
    # 环境变量优先
    env_override = os.getenv('CMAES_PARAMS_FILE', '').strip()
    if env_override:
        cmaes_json_path = env_override
    # 解析候选路径：绝对路径优先，其次 CWD，相对模块目录
    candidates = []
    p = Path(cmaes_json_path)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / cmaes_json_path)
        candidates.append(Path(__file__).resolve().parent / cmaes_json_path)
    file_to_open = None
    for cand in candidates:
        if cand.exists():
            file_to_open = cand
            break
    if file_to_open is None:
        print(f"⚠️ 未找到CMA-ES参数文件: {cmaes_json_path} (工作目录: {os.getcwd()})")
        return low_set

    try:
        with open(file_to_open, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception as e:
        print(f"❌ 读取CMA-ES文件失败: {e}")
        return low_set

    # 新版结构
    if isinstance(raw, dict) and 'station_results' in raw:
        station_dict = raw.get('station_results', {}) or {}
        for sid, rec in station_dict.items():
            try:
                r2 = float(rec.get('best_r2', rec.get('r2', rec.get('R2', 0.0))))
            except Exception:
                r2 = 0.0
            if r2 < threshold:
                low_set.add(str(sid))
        return low_set

    # 旧版结构
    if isinstance(raw, dict):
        for sid, rec in raw.items():
            if not isinstance(rec, dict):
                continue
            try:
                r2 = float(rec.get('best_r2', rec.get('r2', rec.get('R2', 0.0))))
            except Exception:
                r2 = 0.0
            if r2 < threshold:
                low_set.add(str(sid))
    return low_set


def _build_runoff_lag_features(group: pd.DataFrame,
                               lags: List[int],
                               roll_windows: List[int]) -> pd.DataFrame:
    """
    针对单个站点分组生成滞后与滚动特征（只使用历史信息）。
    - 原始列要求：['date','runoff'] 至少；其余原样保留
    - 滞后特征空值填充：仅首日为空，填0.0以避免NaN
    - 滚动统计使用 runoff.shift(1) 的历史值，min_periods=1 保证无NaN
    """
    grp = group.sort_values('date').copy()

    # 基础：前一日径流，供滚动统计使用（只含历史）
    prev = grp['runoff'].shift(1)

    # 滞后特征
    max_lag = max(lags) if lags else 0
    for k in lags:
        col = f'runoff_lag_{k}d'
        grp[col] = grp['runoff'].shift(k)
        # 仅开头不足 k 天的样本会是NaN；用0.0填充以避免下游Scaler报错
        grp[col] = grp[col].fillna(0.0)

    # 滚动统计（使用 prev，确保只依赖历史）
    for w in roll_windows:
        grp[f'runoff_mean_{w}d'] = prev.rolling(window=w, min_periods=1).mean()
        grp[f'runoff_std_{w}d'] = prev.rolling(window=w, min_periods=1).std().fillna(0.0)

    return grp


def augment_csv_with_runoff_lags(src_csv_path: str,
                                 dst_csv_path: str,
                                 lags: List[int] = [1, 3, 7, 14, 30],
                                 roll_windows: List[int] = [7, 30],
                                 station_col: str = 'station_id',
                                 date_col: str = 'date') -> Tuple[str, List[str]]:
    """
    读取原CSV，分站点生成径流的滞后/滚动特征，写出增强CSV。

    Returns:
        (输出路径, 新增特征列列表)
    """
    if not os.path.exists(src_csv_path):
        raise FileNotFoundError(f"找不到源CSV: {src_csv_path}")

    df = pd.read_csv(src_csv_path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        raise ValueError(f"CSV缺少日期列: {date_col}")

    required = {station_col, date_col, 'runoff'}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少必要列: {missing}")

    # 分站点生成
    groups = []
    for sid, g in df.groupby(station_col, sort=False):
        groups.append(_build_runoff_lag_features(g, lags=lags, roll_windows=roll_windows))

    out = pd.concat(groups, axis=0, ignore_index=True)
    out = out.sort_values([station_col, date_col])

    new_cols = [f'runoff_lag_{k}d' for k in lags]
    for w in roll_windows:
        new_cols += [f'runoff_mean_{w}d', f'runoff_std_{w}d']

    # 确保没有NaN（仅来自首日std），统一填0
    out[new_cols] = out[new_cols].fillna(0.0)

    # 写出
    os.makedirs(os.path.dirname(dst_csv_path) or '.', exist_ok=True)
    out.to_csv(dst_csv_path, index=False)
    print(f"✅ 已生成增强CSV: {dst_csv_path}  (新增列: {len(new_cols)})")
    return dst_csv_path, new_cols


def export_low_station_list(cmaes_json_path: str, threshold: float, save_path: str) -> Set[str]:
    """
    导出低R²站点清单CSV，包含 station_id 与 flag。
    返回低R²站点集合。
    """
    low_set = compute_low_r2_stations(cmaes_json_path, threshold)
    rows = [{'station_id': sid, 'low_r2_flag': 1} for sid in sorted(low_set)]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"💾 已导出低R²站点清单({len(low_set)}): {save_path}")
    return low_set


def build_augmented_feature_cols(base_cols: Optional[List[str]], new_cols: List[str], max_cols: int = 20) -> List[str]:
    """
    生成最终用于模型的特征列清单。
    - base_cols: 原特征列（如 ['pet','precip','temp']）
    - new_cols: 新增径流滞后/滚动特征
    - max_cols: 控制规模，默认加入较精简的一部分以稳健起步
    """
    base_cols = base_cols or ['pet', 'precip', 'temp']
    # 精简挑选：lag(1,3,7) + mean/std(7,30)
    preferred = []
    for k in [1, 3, 7]:
        col = f'runoff_lag_{k}d'
        if col in new_cols:
            preferred.append(col)
    for w in [7, 30]:
        for suf in ['mean', 'std']:
            col = f'runoff_{suf}_{w}d'
            if col in new_cols:
                preferred.append(col)

    final = base_cols + preferred
    if len(final) > max_cols:
        final = final[:max_cols]
    print(f"🧩 最终特征列数: {len(final)} -> {final}")
    return final


def run_pipeline(
    src_csv: str,
    cmaes_json: str = 'cmaes_optimal_params.json',
    out_dir: str = './outputs/augmented',
    r2_threshold: float = 0.2,
    lags: List[int] = [1, 3, 7, 14, 30],
    roll_windows: List[int] = [7, 30],
) -> Dict[str, str]:
    """
    一键离线增强：生成带滞后/滚动的CSV + 导出低R²站点清单。
    不直接训练，只准备数据与清单。返回路径字典。
    """
    os.makedirs(out_dir, exist_ok=True)
    dst_csv = os.path.join(out_dir, '特征合并长表_with_lags.csv')
    low_csv = os.path.join(out_dir, 'low_r2_stations.csv')

    dst_csv, new_cols = augment_csv_with_runoff_lags(
        src_csv_path=src_csv,
        dst_csv_path=dst_csv,
        lags=lags,
        roll_windows=roll_windows,
    )
    low_set = export_low_station_list(cmaes_json, threshold=r2_threshold, save_path=low_csv)

    # 同步返回推荐的 feature_cols（给主程序/配置使用）
    feature_cols = build_augmented_feature_cols(['pet', 'precip', 'temp'], new_cols)

    meta = {
        'augmented_csv': dst_csv,
        'low_r2_list': low_csv,
        'recommended_feature_cols': ','.join(feature_cols),
        'low_r2_count': str(len(low_set))
    }
    print(f"✅ 增强准备完成: {meta}")
    return meta


if __name__ == '__main__':
    # 示例独立运行：
    # python MoE_lowflow_augment.py
    # 将在 ./outputs/augmented 下生成增强CSV与低R²清单，并打印推荐特征列
    default_src = r"D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理\特征合并长表.csv"
    try:
        run_pipeline(src_csv=default_src,
                     cmaes_json='cmaes_optimal_params.json',
                     out_dir='./outputs/augmented',
                     r2_threshold=0.2)
    except Exception as e:
        print(f"❌ 运行失败: {e}")


