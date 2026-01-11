# -*- coding: utf-8 -*-
"""
MoE_pbm.py
优化后的PBM模块：支持预计算结果加载和CMA-ES参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from MoE_cmaes_loader import CMAESParamLoader


class OptimizedPBM(nn.Module):
    """优化后的物理机理模型"""
    
    def __init__(self, config: Dict[str, Any], cmaes_loader: CMAESParamLoader = None):
        """
        初始化优化后的PBM模块
        
        Args:
            config: 配置字典
            cmaes_loader: CMA-ES参数加载器
        """
        super().__init__()
        self.config = config
        self.cmaes_loader = cmaes_loader or CMAESParamLoader()
        self.use_precomputed = config.get('use_precomputed_pbm', False)  # 默认关闭预计算
        
        # 初始化默认参数
        self.default_params = self._get_default_params()
        
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'snow_params': {
                'snowf_upper': 3.3,
                'rainf_lower': -1.1,
                'melt_crit': 0.0,
                'frc_liquid': 0.06,
                'melt_factor': 3.0,
                'melt_temp': 0.0
            },
            'runoff_params': {
                'beta_e': 0.75,
                'wmin_ratio': 0.1,
                'wmax_ratio': 1.0,
                'c_max': 100.0,
                'b': 0.5,
                'k': 0.1,
                'alpha': 0.5
            },
            'drainage_params': {
                'qsb_min': 1.15741e-05,
                'qsb_max': 1.15741e-04,
                'qsb_low': 0.9,
                'qsb_hig': 0.9,
                'qsb_exp': 1.5,
                'gw_recharge': 0.2
            },
            'et_params': {
                'rm_crit': 0.7,
                'wilting_ratio': 0.1,
                'sevap_low': 0.1,
                'et_alpha': 1.0,
                'transp_fraction': 1.0,
                'et_beta': 1.0
            },
            'groundwater_params': {
                'retention_time': 30.0,
                'baseflow_threshold': 0.3,
                'k_drainage': 0.05,
                'drainage_exp': 1.5,
                'baseflow_factor': 0.3,
                'groundwater_decay': 0.95
            }
        }
    
    def get_station_params(self, station_id: str) -> Dict[str, Any]:
        """获取站点特定参数"""
        if self.cmaes_loader:
            return self.cmaes_loader.get_station_params(station_id)
        return self.default_params
    
    def forward(self, inputs: Dict[str, torch.Tensor], station_ids: torch.Tensor, station_ids_str: list = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            inputs: 输入数据字典
            station_ids: 站点ID张量
            
        Returns:
            输出结果字典
        """
        batch_size = inputs['precip'].shape[0]
        device = inputs['precip'].device
        
        # 初始化输出
        outputs = {
            'snow_output': torch.zeros(batch_size, device=device),
            'runoff_output': torch.zeros(batch_size, device=device),
            'et_output': torch.zeros(batch_size, device=device),
            'groundwater_output': torch.zeros(batch_size, device=device)
        }
        
        # 如果使用预计算结果，直接加载
        if self.use_precomputed:
            for i in range(batch_size):
                station_id = f"camels_{station_ids[i].item():08d}"
                time_step = inputs.get('time_step', torch.tensor(0, device=device))[i].item()
                
                pbm_results = self.cmaes_loader.get_pbm_results(station_id, time_step)
                if pbm_results:
                    for key, value in pbm_results.items():
                        if key in outputs:
                            outputs[key][i] = value
                else:
                    # 如果预计算结果不可用，使用实时计算
                    outputs = self._compute_realtime_pbm(inputs, station_ids, i, outputs)
        else:
            # 使用实时计算
            outputs = self._compute_realtime_pbm(inputs, station_ids, 0, outputs, station_ids_str)
        
        return outputs
    
    def _compute_realtime_pbm(self, inputs: Dict[str, torch.Tensor], 
                            station_ids: torch.Tensor, 
                            batch_idx: int, 
                            outputs: Dict[str, torch.Tensor],
                            station_ids_str: list = None) -> Dict[str, torch.Tensor]:
        """实时计算PBM（批量优化版本）"""
        batch_size = inputs['precip'].shape[0]
        device = inputs['precip'].device
        
        # 🚀 优化：批量提取驱动数据，避免逐个item()调用
        precip_batch = inputs['precip']  # [batch_size]
        temp_batch = inputs['temp']      # [batch_size]
        pet_batch = inputs['pet']        # [batch_size]
        
        # 🚀 优化：批量获取参数（使用默认参数作为基准，避免逐个查询）
        # 对于大多数站点使用默认参数，特殊站点可以逐个覆盖
        snow_melt_factor = 3.0
        c_max = 100.0
        beta_e = 2.0
        et_alpha = 1.0
        transp_fraction = 0.5
        k_drainage = 0.05
        baseflow_factor = 0.3
        
        # 🚀 批量计算雪水过程
        snowf_upper = 3.3
        rainf_lower = -1.1
        temp_range = snowf_upper - rainf_lower
        snow_fraction = torch.clamp((snowf_upper - temp_batch) / temp_range, 0, 1)
        snowf = precip_batch * snow_fraction
        rainf = F.softplus(precip_batch - snowf - 1e-6)
        
        # 批量融雪计算
        temp_diff = temp_batch - 0.0  # melt_temp = 0
        smelt_pot = torch.where(temp_diff > 0, snow_melt_factor * temp_diff, torch.zeros_like(temp_batch))
        smelt_pot = F.softplus(smelt_pot)
        
        snow_output = snowf * snow_melt_factor + smelt_pot
        
        #  批量径流计算
        effective_precip = rainf + smelt_pot
        runoff_output = effective_precip * (1.0 - torch.exp(-effective_precip / (c_max * beta_e)))
        
        #  批量ET计算
        et_output = pet_batch * et_alpha * transp_fraction
        
        #  批量地下水计算
        groundwater_output = effective_precip * k_drainage + et_output * baseflow_factor
        
        # 确保非负输出（批量操作）
        outputs['snow_output'] = torch.clamp(snow_output, min=0.0)
        outputs['runoff_output'] = torch.clamp(runoff_output, min=0.0)
        outputs['et_output'] = torch.clamp(et_output, min=0.0)
        outputs['groundwater_output'] = torch.clamp(groundwater_output, min=0.0)
        
        return outputs
    
    def get_snow_process(self, precip: torch.Tensor, temp: torch.Tensor, 
                        station_id: str, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """雪水过程计算"""
        # 使用站点特定参数
        snow_params = params.get('snow_params', self.default_params['snow_params'])
        
        # 降雨降雪分离
        snowf, rainf = self._get_rain_and_snow(precip, temp, snow_params)
        
        # 融雪计算
        smelt_pot = self._get_potential_snowmelt(temp, snow_params)
        
        return snowf, rainf
    
    def _get_rain_and_snow(self, precip: torch.Tensor, temp: torch.Tensor, 
                          snow_params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """降雨降雪分离"""
        snowf_upper = snow_params['snowf_upper']
        rainf_lower = snow_params['rainf_lower']
        
        temp_range = snowf_upper - rainf_lower
        if temp_range > 1e-8:
            snow_fraction = torch.clamp((snowf_upper - temp) / temp_range, 0, 1)
        else:
            snow_fraction = torch.where(temp <= snowf_upper, 
                                      torch.ones_like(temp), 
                                      torch.zeros_like(temp))
        
        snowf = precip * snow_fraction
        rainf = F.softplus(precip - snowf - 1e-6)
        
        return snowf, rainf
    
    def _get_potential_snowmelt(self, temp: torch.Tensor, 
                               snow_params: Dict[str, Any]) -> torch.Tensor:
        """潜在融雪计算"""
        melt_temp = snow_params['melt_temp']
        melt_factor = snow_params['melt_factor']
        
        temp_diff = temp - melt_temp
        smelt_pot = torch.where(temp_diff > 0, 
                               melt_factor * temp_diff, 
                               torch.zeros_like(temp))
        
        return F.softplus(smelt_pot)
    
    def get_runoff_process(self, throughfall: torch.Tensor, rootmoist: torch.Tensor,
                          station_id: str, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """径流生成过程"""
        runoff_params = params.get('runoff_params', self.default_params['runoff_params'])
        
        # 地表径流计算
        qs = self._get_surface_runoff(throughfall, rootmoist, runoff_params)
        
        # 地下径流计算
        qsb = self._get_drainage(rootmoist, runoff_params)
        
        return qs, qsb
    
    def _get_surface_runoff(self, throughfall: torch.Tensor, rootmoist: torch.Tensor,
                           runoff_params: Dict[str, Any]) -> torch.Tensor:
        """地表径流计算"""
        beta = runoff_params['beta_e']
        c_max = runoff_params['c_max']
        b = runoff_params['b']
        
        wmin = c_max * runoff_params.get('wmin_ratio', 0.1)
        wmax = c_max * runoff_params.get('wmax_ratio', 1.0)
        
        # 简化的地表径流计算
        if c_max > wmin and rootmoist > wmin:
            rm_sub = wmax - (wmax - wmin) * (1 - (rootmoist - wmin) / (c_max - wmin))**(1 / (1 + beta))
            rm_sub = torch.max(rootmoist, rm_sub)
        else:
            rm_sub = rootmoist
        
        # 计算径流
        if wmax > wmin:
            c1 = torch.clamp(((wmax - rm_sub) / (wmax - wmin))**(1 + beta), 0, 1)
            if rm_sub + throughfall <= wmax:
                c2 = torch.clamp(((wmax - rm_sub - throughfall) / (wmax - wmin))**(1 + beta), 0, 1)
            else:
                c2 = torch.zeros_like(c1)
        else:
            c1 = c2 = torch.zeros_like(throughfall)
        
        qs = throughfall - torch.max(0, wmin - rootmoist) - ((wmax - wmin) / (1 + beta)) * (c1 - c2)
        
        return F.softplus(qs)
    
    def _get_drainage(self, rootmoist: torch.Tensor, runoff_params: Dict[str, Any]) -> torch.Tensor:
        """地下径流计算"""
        c_max = runoff_params['c_max']
        qsb_min = 1.15741e-05
        qsb_max = 1.15741e-04
        qsb_low = 0.9
        qsb_hig = 0.9
        qsb_exp = 1.5
        
        if c_max <= 1e-10:
            return torch.zeros_like(rootmoist)
        
        no_qsb = (c_max <= 1e-10) | (rootmoist <= c_max * qsb_low)
        full_qsb = (c_max > 1e-10) & (rootmoist >= c_max * qsb_hig)
        
        qsb = torch.where(no_qsb, 
                         torch.zeros_like(rootmoist),
                         qsb_min * (rootmoist / c_max))
        
        if full_qsb.any():
            max_qsb = qsb + (qsb_max - qsb_min) * \
                     ((rootmoist - c_max * qsb_hig) / (c_max - c_max * qsb_hig))**qsb_exp
            qsb = torch.where(full_qsb, max_qsb, qsb)
        
        qsb = torch.min(qsb, rootmoist)
        
        return F.softplus(qsb)
    
    def get_et_process(self, potevap: torch.Tensor, rootmoist: torch.Tensor,
                      station_id: str, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """蒸散发过程"""
        et_params = params.get('et_params', self.default_params['et_params'])
        
        # 植物蒸腾
        transp = self._get_transpiration(potevap, rootmoist, et_params)
        
        # 土壤蒸发
        sevap = self._get_soilevap(potevap, rootmoist, et_params)
        
        return transp, sevap
    
    def _get_transpiration(self, potevap: torch.Tensor, rootmoist: torch.Tensor,
                          et_params: Dict[str, Any]) -> torch.Tensor:
        """植物蒸腾计算"""
        rm_crit = et_params['rm_crit']
        wilting_ratio = et_params.get('wilting_ratio', 0.1)
        transp_frac = et_params['transp_fraction']
        et_alpha = et_params['et_alpha']
        
        # 简化的蒸腾计算
        transp_stress = torch.clamp((rootmoist - wilting_ratio) / (rm_crit - wilting_ratio), 0, 1)
        transp = potevap * transp_stress * transp_frac * et_alpha
        
        return F.softplus(transp)
    
    def _get_soilevap(self, potevap: torch.Tensor, rootmoist: torch.Tensor,
                     et_params: Dict[str, Any]) -> torch.Tensor:
        """土壤蒸发计算"""
        sevap_low = et_params.get('sevap_low', 0.1)
        et_alpha = et_params['et_alpha']
        
        # 简化的土壤蒸发计算
        sevap_stress = torch.clamp((rootmoist - sevap_low) / (1 - sevap_low), 0, 1)
        sevap = potevap * sevap_stress * et_alpha
        
        return F.softplus(sevap)
    
    def get_groundwater_process(self, groundwstor_old: torch.Tensor, qsb: torch.Tensor,
                               station_id: str, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """地下水过程"""
        gw_params = params.get('groundwater_params', self.default_params['groundwater_params'])
        
        retention_time = gw_params['retention_time']
        baseflow_thresh = gw_params['baseflow_threshold']
        gw_recharge = gw_params.get('gw_recharge', 0.2)
        
        # 基流计算
        qg = torch.where(groundwstor_old > baseflow_thresh,
                        groundwstor_old / retention_time,
                        torch.zeros_like(groundwstor_old))
        
        # 地下水更新
        effective_qsb = qsb * gw_recharge
        groundwstor_new = groundwstor_old + effective_qsb - qg
        
        return F.softplus(groundwstor_new), F.softplus(qg)


def test_optimized_pbm():
    """测试优化后的PBM模块"""
    print("🧪 测试优化后的PBM模块...")
    
    # 创建测试数据
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inputs = {
        'precip': torch.randn(batch_size, device=device),
        'temp': torch.randn(batch_size, device=device),
        'pet': torch.randn(batch_size, device=device),
        'time_step': torch.arange(batch_size, device=device)
    }
    
    station_ids = torch.tensor([9378630, 9378640, 9378650, 9378660], device=device)
    
    # 创建优化后的PBM模块
    config = {'use_precomputed_pbm': True}
    pbm = OptimizedPBM(config)
    
    # 前向传播
    outputs = pbm(inputs, station_ids)
    
    print(f"📊 输出形状:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("✅ 优化后的PBM模块测试完成")


if __name__ == "__main__":
    test_optimized_pbm()
