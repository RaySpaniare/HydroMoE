# -*- coding: utf-8 -*-
"""
MoE_cmaes_loader.py
CMA-ES参数加载器：加载站点特定的优化参数
"""

import json
import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Dict, Any, Optional
from MoE_config import CMAES_CONFIG, PBM_RESULTS_CONFIG

# 全局缓存，避免重复加载占用内存
_GLOBAL_CMAES_CACHE = {
    'params_data': None,
    'optimization_summary': None,
    'pbm_results': None,
    '_pbm_wide': None,
    '_pbm_time_cols': None,
    '_pbm_sid_col': None,
    '_pbm_row_index_map': None,
    '_pbm_time_values': None,
}


class CMAESParamLoader:
    """CMA-ES参数加载器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化CMA-ES参数加载器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or CMAES_CONFIG
        self.params_data = None
        self.optimization_summary = {}
        self.pbm_results = None
        self._pbm_wide = False
        self._pbm_time_cols = []
        self._pbm_sid_col = PBM_RESULTS_CONFIG.get('station_id_col', 'station_id')
        self._pbm_row_index_map = None  # station_id -> row index (for wide)
        self._pbm_time_values = None    # np.ndarray [n_stations, n_time]
        # 若已有全局缓存则直接复用
        if _GLOBAL_CMAES_CACHE['params_data'] is not None:
            self.params_data = _GLOBAL_CMAES_CACHE['params_data']
            self.optimization_summary = _GLOBAL_CMAES_CACHE['optimization_summary'] or {}
        else:
            self._load_params_data()
            _GLOBAL_CMAES_CACHE['params_data'] = self.params_data
            _GLOBAL_CMAES_CACHE['optimization_summary'] = self.optimization_summary

        if _GLOBAL_CMAES_CACHE['pbm_results'] is not None:
            self.pbm_results = _GLOBAL_CMAES_CACHE['pbm_results']
            self._pbm_wide = bool(_GLOBAL_CMAES_CACHE['_pbm_wide'])
            self._pbm_time_cols = list(_GLOBAL_CMAES_CACHE['_pbm_time_cols'] or [])
            self._pbm_sid_col = _GLOBAL_CMAES_CACHE['_pbm_sid_col'] or self._pbm_sid_col
            self._pbm_row_index_map = _GLOBAL_CMAES_CACHE.get('_pbm_row_index_map', None)
            self._pbm_time_values = _GLOBAL_CMAES_CACHE.get('_pbm_time_values', None)
        else:
            self._load_pbm_results()
            _GLOBAL_CMAES_CACHE['pbm_results'] = self.pbm_results
            _GLOBAL_CMAES_CACHE['_pbm_wide'] = self._pbm_wide
            _GLOBAL_CMAES_CACHE['_pbm_time_cols'] = self._pbm_time_cols
            _GLOBAL_CMAES_CACHE['_pbm_sid_col'] = self._pbm_sid_col
            _GLOBAL_CMAES_CACHE['_pbm_row_index_map'] = self._pbm_row_index_map
            _GLOBAL_CMAES_CACHE['_pbm_time_values'] = self._pbm_time_values
    
    def _load_params_data(self):
        """加载CMA-ES参数数据"""
        try:
            params_file = self.config['params_file']
            # 允许通过环境变量覆盖
            env_override = os.getenv('CMAES_PARAMS_FILE', '').strip()
            if env_override:
                params_file = env_override
            # 路径解析：支持工作目录与模块目录
            candidate_paths = []
            p = Path(params_file)
            if p.is_absolute():
                candidate_paths.append(p)
            else:
                candidate_paths.append(Path.cwd() / params_file)
                candidate_paths.append(Path(__file__).resolve().parent / params_file)
            file_to_open = None
            for cand in candidate_paths:
                if cand.exists():
                    file_to_open = cand
                    break
            if file_to_open is None:
                raise FileNotFoundError(params_file)

            with open(file_to_open, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # 兼容两种结构：
            # 1) 旧版: { 'camels_XXXX': {...}, ... }
            # 2) 新版: { 'optimization_summary': {...}, 'station_results': {...} }
            if isinstance(raw, dict) and 'station_results' in raw:
                self.optimization_summary = raw.get('optimization_summary', {}) or {}
                self.params_data = raw.get('station_results', {}) or {}
            else:
                self.params_data = raw if isinstance(raw, dict) else {}
            total = len(self.params_data) if isinstance(self.params_data, dict) else 0
            print(f" 成功加载CMA-ES参数文件: {str(file_to_open)}")
            print(f" 总站点数: {total}")
        except FileNotFoundError:
            print(f" 未找到CMA-ES参数文件: {self.config['params_file']} (工作目录: {os.getcwd()})")
            self.params_data = {}
        except Exception as e:
            print(f" 加载CMA-ES参数文件失败: {e}")
            self.params_data = {}
    
    def _load_pbm_results(self):
        """
        加载预计算PBM结果
        如果不使用预计算结果，则创建虚拟数据结构
        """
        # 检查是否需要加载预计算结果
        if not PBM_RESULTS_CONFIG.get('use_precomputed_results', False):
            print("🔧 配置为不使用预计算PBM结果，跳过加载")
            self.pbm_results = self._create_dummy_pbm_results()
            return
            
        # 检查是否配置了结果文件
        if 'results_file' not in PBM_RESULTS_CONFIG:
            print("⚠️ 未配置PBM结果文件路径，使用虚拟数据")
            self.pbm_results = self._create_dummy_pbm_results()
            return
            
        try:
            # 尝试不同的编码方式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(PBM_RESULTS_CONFIG['results_file'], encoding=encoding)
                    # 丢弃全空列，重置索引样式列
                    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed:')]
                    # 降低内存占用：数值列转为 float32
                    for c in df.columns:
                        if c != self._pbm_sid_col and pd.api.types.is_numeric_dtype(df[c]):
                            try:
                                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
                            except Exception:
                                pass
                    self.pbm_results = df
                    print(f"✅ 成功加载PBM结果文件: {PBM_RESULTS_CONFIG['results_file']} (编码: {encoding})")
                    print(f"📊 PBM结果数据形状: {self.pbm_results.shape}")
                    # 自动识别站点列与格式
                    self._auto_detect_pbm_schema()
                    # 为宽表构建快速索引
                    self._build_pbm_fast_index()
                    return
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败，创建虚拟数据
            print("⚠️ 无法读取PBM结果文件，使用虚拟数据")
            self.pbm_results = self._create_dummy_pbm_results()
            
        except FileNotFoundError:
            print(f"❌ 未找到PBM结果文件: {PBM_RESULTS_CONFIG.get('results_file', '未配置')}")
            print("⚠️ 使用虚拟PBM结果数据")
            self.pbm_results = self._create_dummy_pbm_results()
        except Exception as e:
            print(f"❌ 加载PBM结果文件失败: {e}")
            print("⚠️ 使用虚拟PBM结果数据")
            self.pbm_results = self._create_dummy_pbm_results()
            self._pbm_wide = False
    
    def _create_dummy_pbm_results(self):
        """创建虚拟PBM结果数据"""
        import numpy as np
        
        # 创建虚拟数据
        n_stations = 10
        n_time_steps = 1000
        
        data = []
        for i in range(n_stations):
            station_id = f"camels_{9378630 + i:08d}"
            for t in range(n_time_steps):
                data.append({
                    'station_id': station_id,
                    'time_step': t,
                    'snow_output': np.random.randn() * 0.1,
                    'runoff_output': np.random.randn() * 0.1,
                    'et_output': np.random.randn() * 0.1,
                    'groundwater_output': np.random.randn() * 0.1
                })
        
        return pd.DataFrame(data)

    def _auto_detect_pbm_schema(self):
        """自动检测PBM结果文件的列模式（长表/宽表）、站点列与时间列"""
        if self.pbm_results is None or self.pbm_results.empty:
            return
        df = self.pbm_results
        cfg_sid = PBM_RESULTS_CONFIG.get('station_id_col', 'station_id')
        cfg_time = PBM_RESULTS_CONFIG.get('time_col', 'time_step')

        def _find_sid_col(frame: pd.DataFrame) -> Optional[str]:
            import re
            # 1) 直接使用配置
            if cfg_sid in frame.columns:
                return cfg_sid
            # 2) 常见别名
            aliases = ['station_id', 'site_id', 'gauge_id', 'station', 'site_no', '站点', '站点名']
            for name in aliases:
                if name in frame.columns:
                    return name
            # 3) 通过值模式匹配（包含 camels_########）
            pattern = re.compile(r"^camels_\d{8}$")
            object_cols = [c for c in frame.columns if frame[c].dtype == 'object']
            best_col = None
            best_ratio = 0.0
            for c in object_cols:
                vals = frame[c].astype(str).head(1000)
                ratio = (vals.str.match(pattern).sum()) / max(len(vals), 1)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_col = c
            if best_col is not None and best_ratio >= 0.5:
                return best_col
            # 4) 如果第一列看起来是递增索引，忽略它，尝试第二列
            cols = list(frame.columns)
            if len(cols) >= 2 and frame[cols[0]].dtype != 'object' and frame[cols[1]].dtype == 'object':
                return cols[1]
            # 5) 回退：选第一个object列
            return object_cols[0] if object_cols else None

        # 检测是否为长表
        is_long = cfg_time in df.columns
        sid_col = _find_sid_col(df)
        if sid_col is None:
            # 无法识别站点列，按长表处理（可能只作为占位）
            print("⚠️ 未能自动识别PBM站点列，按长表处理")
            self._pbm_wide = False
            self._pbm_sid_col = cfg_sid
            return
        self._pbm_sid_col = sid_col

        if is_long:
            # 长表：仅记录站点列
            self._pbm_wide = False
            print(f"🗂️ 检测到PBM长表格式：站点列='{self._pbm_sid_col}', 时间列='{cfg_time}'")
            return

        # 宽表：构建时间列集合
        self._pbm_wide = True
        cols = [c for c in df.columns if c != sid_col]
        # 尝试基于列名模式筛选时间列
        time_cols_by_name = [
            c for c in cols
            if (str(c).isdigit() or str(c).startswith('t_') or str(c).startswith('day_'))
        ]
        time_cols = time_cols_by_name.copy()
        # 如果基于名字没有找到，使用数值列并排除元数据列
        if not time_cols:
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            meta_names = set(['lon', 'lat', 'longitude', 'latitude', 'elev', 'elevation', 'area', 'drainage_area', 'x', 'y'])
            # 先按名字排除
            numeric_cols = [c for c in numeric_cols if str(c).lower() not in meta_names]
            # 再按取值范围排除经纬度
            def _is_lon(series: pd.Series) -> bool:
                s = series.dropna()
                return not s.empty and (s.between(-180, 180).mean() > 0.99) and (s.abs().mean() > 1)
            def _is_lat(series: pd.Series) -> bool:
                s = series.dropna()
                return not s.empty and (s.between(-90, 90).mean() > 0.99) and (s.abs().mean() > 1)
            filtered = []
            for c in numeric_cols:
                s = df[c]
                if _is_lon(s) or _is_lat(s):
                    continue
                filtered.append(c)
            time_cols = filtered
        self._pbm_time_cols = list(time_cols)
        print(f"🧭 检测到PBM宽表格式：站点列='{self._pbm_sid_col}', 时间列数={len(self._pbm_time_cols)}")

    def _build_pbm_fast_index(self):
        """为宽表构建快速索引（station_id -> 行索引，时间列为 NumPy 矩阵）。"""
        try:
            if self.pbm_results is None or not self._pbm_wide:
                return
            df = self.pbm_results
            sid_col = self._pbm_sid_col
            if sid_col not in df.columns or not self._pbm_time_cols:
                return
            # 行索引映射
            self._pbm_row_index_map = {str(sid): i for i, sid in enumerate(df[sid_col].astype(str).values)}
            # 时间列矩阵（float32）
            time_vals = df[self._pbm_time_cols].to_numpy(copy=False)
            if not np.issubdtype(time_vals.dtype, np.floating):
                time_vals = time_vals.astype(np.float32, copy=False)
            self._pbm_time_values = time_vals
        except Exception as _:
            # 回退：不影响功能，仅无法加速
            self._pbm_row_index_map = None
            self._pbm_time_values = None
    
    def get_station_params(self, station_id: str) -> Dict[str, Any]:
        """
        获取站点特定的优化参数
        
        Args:
            station_id: 站点ID，格式如 'camels_09378630'
            
        Returns:
            站点特定的参数字典
        """
        if not self.params_data or station_id not in self.params_data:
            print(f"⚠️ 未找到站点 {station_id} 的CMA-ES参数，使用默认参数")
            return self._get_default_params()
        
        station_data = self.params_data[station_id]
        
        # 检查是否有best_params字段
        if 'best_params' not in station_data:
            print(f"⚠️ 站点 {station_id} 没有best_params字段，使用默认参数")
            return self._get_default_params()
        
        best_params = station_data['best_params']
        
        # 根据映射关系转换参数
        converted_params = {}
        for category, mapping in self.config['param_mapping'].items():
            converted_params[category] = {}
            for moe_param, cmaes_param in mapping.items():
                if cmaes_param in best_params:
                    converted_params[category][moe_param] = best_params[cmaes_param]
                else:
                    print(f"⚠️ 参数 {cmaes_param} 在站点 {station_id} 中不存在")
        
        # 添加固定参数
        converted_params['snow_params'].update({
            'snowf_upper': 3.3,
            'rainf_lower': -1.1
        })
        
        # 确保所有必需的参数都存在，用默认值填充缺失项
        default_params = self._get_default_params()
        for category in default_params:
            if category not in converted_params:
                converted_params[category] = {}
            for param_name, default_value in default_params[category].items():
                if param_name not in converted_params[category]:
                    converted_params[category][param_name] = default_value
        
        return converted_params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'runoff_params': {
                'c_max': 100.0,
                'beta_e': 2.0,
                'b': 0.5,
                'k': 0.1,
                'alpha': 0.5
            },
            'et_params': {
                'transp_fraction': 0.5,
                'et_alpha': 1.0,
                'rm_crit': 0.5,
                'et_beta': 1.0
            },
            'snow_params': {
                'melt_factor': 3.0,
                'melt_temp': 0.0,
                'snowf_upper': 3.3,
                'rainf_lower': -1.1
            },
            'groundwater_params': {
                'k_drainage': 0.05,
                'drainage_exp': 1.5,
                'baseflow_factor': 0.3,
                'groundwater_decay': 0.95
            }
        }
    
    def get_pbm_results(self, station_id: str, time_step: int) -> Optional[Dict[str, float]]:
        """
        获取预计算的PBM结果
        
        Args:
            station_id: 站点ID
            time_step: 时间步
            
        Returns:
            PBM结果字典，如果未找到则返回None
        """
        if self.pbm_results is None:
            return None
        
        try:
            sid_col = self._pbm_sid_col or PBM_RESULTS_CONFIG.get('station_id_col', 'station_id')
            if not self._pbm_wide:
                # 长表：按列筛选
                station_data = self.pbm_results[
                    (self.pbm_results[sid_col] == station_id) &
                    (self.pbm_results[PBM_RESULTS_CONFIG['time_col']] == time_step)
                ]
                if station_data.empty:
                    return None
                results = {}
                for output_name, col_name in PBM_RESULTS_CONFIG['output_cols'].items():
                    if col_name in station_data.columns:
                        results[output_name] = station_data[col_name].iloc[0]
                return results if results else None
            else:
                # 宽表：使用预构建索引快速访问
                if self._pbm_row_index_map is not None and self._pbm_time_values is not None:
                    row_idx = self._pbm_row_index_map.get(str(station_id), None)
                    if row_idx is None:
                        # 尝试根据数字后缀模糊匹配
                        import re
                        m = re.search(r"(\d+)$", str(station_id))
                        if m:
                            last_digits = m.group(1)
                            # 构造一次性反查表（仅在需要时）
                            for k, v in self._pbm_row_index_map.items():
                                if k.endswith(last_digits):
                                    row_idx = v
                                    break
                    if row_idx is None:
                        return None
                    t = int(time_step)
                    if t < 0 or t >= self._pbm_time_values.shape[1]:
                        return None
                    val = self._pbm_time_values[row_idx, t]
                    if not np.isfinite(val):
                        return None
                    return {'runoff_output': float(val)}
                # 回退：使用原先的 Pandas 路径（较慢）
                df = self.pbm_results
                row = df[df[sid_col] == station_id]
                if row.empty:
                    import re
                    m = re.search(r"(\d+)$", str(station_id))
                    if m:
                        last_digits = m.group(1)
                        row = df[df[sid_col].astype(str).str.contains(last_digits, na=False)]
                if row.empty:
                    return None
                t = int(time_step)
                if 0 <= t < len(self._pbm_time_cols):
                    col_name = self._pbm_time_cols[t]
                    try:
                        return {'runoff_output': float(row.iloc[0][col_name])}
                    except Exception:
                        return None
                return None
        except Exception as e:
            print(f"❌ 获取PBM结果失败: {e}")
            return None
    
    def get_station_info(self, station_id: str) -> Dict[str, Any]:
        """
        获取站点信息
        
        Args:
            station_id: 站点ID
            
        Returns:
            站点信息字典
        """
        if not self.params_data or station_id not in self.params_data:
            return {'station_idx': -1, 'best_r2': 0.0, 'success': False}
        
        station_data = self.params_data[station_id]
        return {
            'station_idx': station_data.get('station_idx', -1),
            'best_r2': station_data.get('best_r2', 0.0),
            'success': station_data.get('success', station_data.get('optimization_success', False))
        }
    
    def list_available_stations(self) -> list:
        """获取所有可用的站点ID列表"""
        if not self.params_data:
            return []
        return list(self.params_data.keys())
    
    def get_params_summary(self) -> Dict[str, Any]:
        """获取参数统计摘要"""
        if not self.params_data:
            return {'total_stations': 0, 'successful_stations': 0, 'success_rate': 0.0, 'avg_r2': 0.0}
        # 优先使用文件中的摘要
        if self.optimization_summary:
            perf = self.optimization_summary.get('optimization_performance', {})
            stats = self.optimization_summary.get('statistics', {})
            total = int(perf.get('total_processed', len(self.params_data)))
            succ = int(perf.get('successful_optimizations', 0))
            rate = float(perf.get('success_rate', (succ / total * 100 if total else 0.0)))
            avg_r2 = float(perf.get('average_r2', stats.get('mean_r2', 0.0)))
            # 将 success_rate 统一为小数（0-1）
            if rate > 1.0:
                rate = rate / 100.0
            return {
                'total_stations': total,
                'successful_stations': succ,
                'success_rate': rate,
                'avg_r2': avg_r2
            }
        # 否则基于逐站统计
        total_stations = len(self.params_data)
        successful_stations = 0
        r2_sum = 0.0
        for data in self.params_data.values():
            if data.get('success', data.get('optimization_success', False)):
                successful_stations += 1
            r2_sum += float(data.get('best_r2', 0.0))
        success_rate = successful_stations / total_stations if total_stations > 0 else 0.0
        return {
            'total_stations': total_stations,
            'successful_stations': successful_stations,
            'success_rate': success_rate,
            'avg_r2': r2_sum / total_stations if total_stations > 0 else 0.0
        }


def test_cmaes_loader():
    """测试CMA-ES参数加载器"""
    print("🧪 测试CMA-ES参数加载器...")
    
    loader = CMAESParamLoader()
    
    # 测试参数摘要
    summary = loader.get_params_summary()
    print(f"📊 参数摘要: {summary}")
    
    # 测试站点列表
    stations = loader.list_available_stations()
    print(f"🏪 可用站点数: {len(stations)}")
    
    if stations:
        # 测试第一个站点
        test_station = stations[0]
        print(f"🔍 测试站点: {test_station}")
        
        # 获取站点信息
        station_info = loader.get_station_info(test_station)
        print(f"📋 站点信息: {station_info}")
        
        # 获取站点参数
        params = loader.get_station_params(test_station)
        print(f"⚙️ 站点参数: {params}")
        
        # 测试PBM结果获取
        pbm_results = loader.get_pbm_results(test_station, 0)
        print(f"🌊 PBM结果: {pbm_results}")


if __name__ == "__main__":
    test_cmaes_loader()
