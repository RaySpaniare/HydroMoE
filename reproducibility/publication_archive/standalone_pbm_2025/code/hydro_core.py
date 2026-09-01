#!/usr/bin/env python3
"""
HydroPy核心水文过程模块
包含所有核心水文计算方法，基于HydroPy原始机理公式
"""

import numpy as np
import math


class HydroPyCSVCore:
    """HydroPy CSV核心版本 - 水文过程计算"""

    def __init__(self, static_params=None):
        """
        初始化HydroPy核心模型

        参数:
            static_params: 静态参数字典，包含需要优化的静态参数
        """
        # 固定模型参数（基于HydroPy原版，不参与优化）
        self.opt = {
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

        # 静态参数（通过优化获得）
        self.params = static_params or {}
        
    def get_rain_and_snow(self, precip, tsurf):
        """降雨降雪分离 - 原始WIGMOSTA方案"""
        temp_c = tsurf  # 已为摄氏度输入
        temp_range = self.opt['snowf_upper'] - self.opt['rainf_lower']
        
        if temp_range > 0:
            snowfrct = min(1.0, max(0, (self.opt['snowf_upper'] - temp_c) / temp_range))
        else:
            snowfrct = 1.0 if temp_c <= self.opt['snowf_upper'] else 0.0
        
        snowf = precip * snowfrct
        rainf = max(0, precip - snowf)
        
        return snowf, rainf, snowfrct
    
    def get_potential_snowmelt(self, tsurf, lat, day_of_year):
        """潜在融雪计算 - 日度数方法"""
        temp_c = tsurf  # 已为摄氏度输入
        
        if temp_c <= self.opt['melt_crit']:
            return 0.0, 0.0
        
        # 计算日长（简化版本）
        decl = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
        lat_rad = math.radians(lat)
        decl_rad = math.radians(decl)
        
        try:
            hour_angle = math.acos(-math.tan(lat_rad) * math.tan(decl_rad))
            daylen = 2 * hour_angle * 12 / math.pi
        except:
            daylen = 12.0  # 备用值12小时
        
        # 融雪计算 - 时间方案
        melt_factor = daylen / 24.0 * 8.3 + 0.7
        smelt_pot = melt_factor * (temp_c - self.opt['melt_crit'])
        
        return max(0, smelt_pot), daylen
    
    def update_snow(self, swe_old, wliq_old, snowf, smelt_pot):
        """雪过程更新"""
        # 雪累积
        swe_new = swe_old + snowf
        
        # 液态水容量
        wliq_max = swe_new * self.opt['frc_liquid']
        
        # 实际融雪（不能超过现有雪量）
        smelt = min(smelt_pot, swe_new)
        swe_new = swe_new - smelt
        
        # 液态水更新
        wliq_new = wliq_old + smelt
        
        # 液态水溢出
        if wliq_new > wliq_max:
            overflow = wliq_new - wliq_max
            wliq_new = wliq_max
        else:
            overflow = 0.0
        
        # 总的液态水输入（降雨+融雪溢出）
        rainmelt = overflow
        
        return swe_new, wliq_new, smelt, rainmelt
    
    def diagnose_frozen_ground(self, temp, frozen_temp_threshold=-1.0):
        """诊断冻土状态"""
        temp_c = temp  # 已为摄氏度输入
        frozen = temp_c < frozen_temp_threshold
        
        return frozen
    
    def get_surface_runoff(self, throughfall, rootmoist, wcap, beta=None, wmin=None, wmax=None, frozen=False, static_params=None):
        """地表径流计算 - 改进ARNO方案 (与说明文档一致)"""
        if throughfall < 0:
            return 0.0

        # 使用参数或默认比值
        if beta is None:
            beta = self.opt.get('beta_e', 0.75)
        if wmin is None:
            wmin = wcap * 0.1
        if wmax is None:
            wmax = wcap

        # 冻土：所有穿透雨作为地表径流
        if frozen:
            return throughfall

        # 站点化处理：不再进行“子网格”变换，直接使用站点的根区土壤湿度
        rm_sub = rootmoist

        # 计算方程组分
        if wmax > wmin:
            c1 = ((wmax - rm_sub) / (wmax - wmin)) ** (1 + beta)
            c1 = min(c1, 1.0)
            if rm_sub + throughfall <= wmax:
                c2 = ((wmax - rm_sub - throughfall) / (wmax - wmin)) ** (1 + beta)
                c2 = max(c2, 0.0)
            else:
                c2 = 0.0
        else:
            c1 = c2 = 0.0

        # 状态与超量
        if throughfall > (wcap - rootmoist):
            excess = throughfall + (rootmoist - wcap)
        else:
            excess = 0.0

        rm_res = max(0, wmin - rootmoist)
        qs = throughfall - rm_res - ((wmax - wmin) / (1 + beta)) * (c1 - c2)
        qs = max(qs, 0.0)

        overflow = (rm_sub + throughfall) >= wmax
        too_dry = (rm_sub + throughfall) <= wmin
        no_qs = throughfall < 0
        if overflow:
            qs = excess
        elif too_dry or no_qs:
            qs = 0.0

        return max(0.0, qs)
    
    def get_drainage(self, rootmoist, wcap, dt=86400, frozen=False, static_params=None):
        """地下径流计算 - 原始MPI-HM (与说明文档一致)"""
        if wcap <= 1.0e-10:
            return 0.0

        # 冻土情况：无地下径流
        if frozen:
            return 0.0

        # 原始MPI-HM：低阈值无排水，高阈值按非线性增强
        no_qsb = (wcap <= 1.0e-10) or (rootmoist <= wcap * self.opt['qsb_low'])
        full_qsb = (wcap > 1.0e-10) and (rootmoist >= wcap * self.opt['qsb_hig'])

        if no_qsb:
            return 0.0

        # 线性部分
        qsb = self.opt['qsb_min'] * dt * (rootmoist / wcap)

        if full_qsb:
            # 增强的非线性部分
            maxqsb = (qsb + dt * (self.opt['qsb_max'] - self.opt['qsb_min']) *
                      ((rootmoist - wcap * self.opt['qsb_hig']) /
                       (wcap - wcap * self.opt['qsb_hig'])) ** self.opt['qsb_exp'])
            qsb = maxqsb

        qsb = min(qsb, rootmoist)
        return max(0.0, qsb)
    
    def get_initial_groundwater(self):
        """获取初始地下水储量"""
        return self.opt.get('initial_groundwater', 10.0)

    def get_transpiration(self, potevap, rootmoist, wcap, fveg, lai=2.0, static_params=None):
        """计算植物蒸腾 - 简化线性胁迫版本（与说明文档一致）"""
        if static_params:
            rm_crit = static_params.get('rm_crit', self.opt.get('rm_crit', 0.7))
            wilting_ratio = static_params.get('wilting_point_ratio', 0.1)
            transp_frac = static_params.get('transp_fraction', 1.0)
            et_alpha = static_params.get('et_alpha', 1.0)
            lai_eff = static_params.get('lai_efficiency', 1.0)
        else:
            rm_crit = self.opt.get('rm_crit', 0.7)
            wilting_ratio = 0.1
            transp_frac = 1.0
            et_alpha = 1.0
            lai_eff = 1.0

        wcrit = wcap * rm_crit
        wlow = wcap * wilting_ratio

        if rootmoist >= wcrit:
            transp_stress = 1.0
        elif rootmoist > wlow:
            transp_stress = (rootmoist - wlow) / (wcrit - wlow)
        else:
            transp_stress = 0.0

        effective_lai = lai * lai_eff
        transp = potevap * effective_lai * transp_stress * fveg * transp_frac * et_alpha
        transp = min(transp, max(0, rootmoist - wlow))
        return max(0, transp)

    def get_soilevap(self, potevap, rootmoist, wcap, fbare, static_params=None):
        """计算土壤蒸发 - 简化线性胁迫版本（与说明文档一致）"""
        if static_params:
            sevap_low = static_params.get('sevap_low', self.opt.get('sevap_low', 0.1))
            et_alpha = static_params.get('et_alpha', 1.0)
            sevap_alpha = static_params.get('sevap_alpha', 1.0)
        else:
            sevap_low = self.opt.get('sevap_low', 0.1)
            et_alpha = 1.0
            sevap_alpha = 1.0

        wlow = wcap * sevap_low
        if rootmoist > wlow:
            sevap_stress = (rootmoist - wlow) / (wcap - wlow)
        else:
            sevap_stress = 0.0

        sevap = potevap * sevap_stress * fbare * et_alpha * sevap_alpha
        sevap = min(sevap, max(0, rootmoist - wlow))
        return max(0, sevap)

    def get_evapotranspiration(self, potevap, rootmoist, wcap, lai=2.0, static_params=None):
        """蒸散发过程 - 简化版本"""
        # 土壤水分胁迫因子 - 使用优化参数
        if static_params:
            wilting_ratio = static_params.get('wilting_point_ratio', 0.1)
            rm_crit = static_params.get('rm_crit', self.opt.get('rm_crit', 0.7))
        else:
            wilting_ratio = 0.1
            rm_crit = self.opt.get('rm_crit', 0.7)

        wlow = wcap * wilting_ratio  # 萎蔫点
        wcrit = wcap * rm_crit       # 临界点

        # 植物蒸腾
        if rootmoist > wcrit:
            transp_stress = 1.0
        elif rootmoist > wlow:
            transp_stress = (rootmoist - wlow) / (wcrit - wlow)
        else:
            transp_stress = 0.0

        # 使用优化的蒸腾分配比例和多个系数
        if static_params:
            transp_frac = static_params.get('transp_fraction', 0.6)
            et_alpha = static_params.get('et_alpha', 1.0)
            lai_eff = static_params.get('lai_efficiency', 1.0)
        else:
            transp_frac = 0.6
            et_alpha = 1.0
            lai_eff = 1.0
        transp = potevap * (lai * lai_eff) * transp_stress * transp_frac * et_alpha

        # 土壤蒸发
        if rootmoist > wlow:
            sevap_stress = (rootmoist - wlow) / (wcap - wlow)
        else:
            sevap_stress = 0.0

        sevap = (potevap - transp) * sevap_stress

        # 限制总蒸散发不超过潜在蒸发
        total_et = transp + sevap
        if total_et > potevap:
            scale = potevap / total_et
            transp *= scale
            sevap *= scale

        return max(0, transp), max(0, sevap)

    def update_soil(self, rootmoist_old, throughfall, qs, transp, sevap, qsb, wcap):
        """土壤水分更新 - 简化质量守恒（与说明文档一致）"""
        water_input = throughfall - qs
        water_output = transp + sevap + qsb
        rootmoist_new = rootmoist_old + water_input - water_output

        if rootmoist_new > wcap:
            overflow = rootmoist_new - wcap
            rootmoist_new = wcap
            qs_additional = overflow
        else:
            qs_additional = 0.0

        rootmoist_new = max(0, rootmoist_new)
        return rootmoist_new, qs_additional

    def update_groundwater(self, groundwstor_old, qsb, retention_time=None, static_params=None):
        """地下水更新 - 原始简化线性水库（与说明文档一致）"""
        # 参数
        if retention_time is None:
            if static_params and 'groundwater_recession' in static_params:
                retention_time = static_params['groundwater_recession']
            else:
                retention_time = self.opt.get('groundwater_retention_time', 30.0)

        # 获取其他地下水参数
        if static_params:
            baseflow_thresh = static_params.get('baseflow_threshold', 0.3)
            gw_recharge = static_params.get('gw_recharge_rate', 0.2)
        else:
            baseflow_thresh = 0.3
            gw_recharge = 0.2

        # 地下水补给
        recharge = qsb * gw_recharge

        # 基流（阈值+线性水库）
        if groundwstor_old > baseflow_thresh:
            qg = groundwstor_old / retention_time if retention_time > 0 else 0.0
        else:
            qg = 0.0

        # 更新地下水储量
        groundwstor_new = groundwstor_old + recharge - qg

        # 非负限制
        groundwstor_new = max(0.0, groundwstor_new)
        qg = max(0.0, qg)

        return groundwstor_new, qg

    def get_total_runoff(self, qs, qsb, qg, qs_additional=0.0):
        """总径流计算公式：qtot = 地表径流 + 地下径流 + 基流 + 额外地表径流。
        返回非负值。
        """
        qtot = (qs or 0.0) + (qsb or 0.0) + (qg or 0.0) + (qs_additional or 0.0)
        return max(0.0, qtot)
