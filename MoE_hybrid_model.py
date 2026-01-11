"""
混合HydroMoE模型 - 集成PBM物理模块和神经网络专家
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import math
import sys
import os

from MoE_attention import HydroAttentionBlock
from MoE_gate import MoEGate, ExpertDispatcher, ExpertCombiner
from MoE_experts import MLPExpert, BaseExpert
from MoE_pbm import OptimizedPBM
from MoE_cmaes_loader import CMAESParamLoader


class PBMExpert(BaseExpert):
    """PBM专家 - 包装物理模型"""
    
    def __init__(self, input_dim: int, output_dim: int, module_type: str = 'runoff'):
        super().__init__(input_dim, output_dim)
        
        self.module_type = module_type
        
        # 强制使用CMA-ES参数，确保PBM专家正确初始化
        try:
            # 使用CMA-ES参数直接计算PBM
            self.pbm = OptimizedPBM(
                config={'use_precomputed_pbm': False},  # 直接计算，不依赖结果文件
                cmaes_loader=CMAESParamLoader()
            )
            self.use_simple_nn = False
            
            # 验证PBM是否正确初始化
            self._validate_pbm_initialization()
            print(f"✅ {module_type}模块PBM专家初始化成功，包含{len(self.pbm.cmaes_loader.params_data)}个站点参数")
            
        except Exception as e:
            print(f"  ⚠️ {module_type}模块PBM初始化失败: {e}")
            print(f"  🔧 使用修复策略重新初始化...")
            
            # 修复策略：确保CMA-ES参数正确加载
            try:
                cmaes_loader = CMAESParamLoader()
                if cmaes_loader.params_data and len(cmaes_loader.params_data) > 0:
                    self.pbm = OptimizedPBM(
                        config={'use_precomputed_pbm': False},
                        cmaes_loader=cmaes_loader
                    )
                    self.use_simple_nn = False
                    print(f"  ✅ {module_type}模块PBM专家修复成功")
                else:
                    raise Exception("CMA-ES参数为空")
            except:
                print(f"  ❌ {module_type}模块彻底失败，使用NN替代")
                # 使用简单的神经网络替代PBM
                self.pbm = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
                self.use_simple_nn = True
        
        # 输出映射层（将PBM输出映射到标准输出维度）
        self.output_projection = nn.Linear(1, output_dim)
        
    def _validate_pbm_initialization(self):
        """验证PBM是否正确初始化"""
        if not hasattr(self.pbm, 'cmaes_loader'):
            raise Exception("PBM缺少CMA-ES加载器")
        
        if not self.pbm.cmaes_loader.params_data:
            raise Exception("CMA-ES参数数据为空")
        
        # 测试一个样本站点的参数
        sample_station = list(self.pbm.cmaes_loader.params_data.keys())[0]
        sample_params = self.pbm.cmaes_loader.params_data[sample_station]
        
        # 验证站点数据结构
        if 'best_params' not in sample_params:
            raise Exception(f"站点 {sample_station} 缺少best_params字段")
        
        best_params = sample_params['best_params']
        
        # 验证核心CMA-ES参数存在（直接来自优化结果）
        required_cmaes_params = ['wmin', 'wmax', 'beta', 'baseflow_threshold']
        missing_params = [p for p in required_cmaes_params if p not in best_params]
        if missing_params:
            raise Exception(f"缺少关键CMA-ES参数: {missing_params}")
        
        print(f"    ✅ PBM参数验证通过，包含{len(self.pbm.cmaes_loader.params_data)}个站点参数")
        print(f"    📊 样本站点 {sample_station} 参数数量: {len(best_params)}")
        
        # 测试参数转换
        converted_params = self.pbm.cmaes_loader.get_station_params(sample_station)
        if not converted_params:
            raise Exception("参数转换失败")
            
        print(f"    🔄 参数转换成功，包含 {len(converted_params)} 个参数组")
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, input_dim] 输入特征
            **kwargs: 可能包含station_ids等信息
        """
        batch_size = x.shape[0]
        device = x.device
        
        if self.use_simple_nn:
            # 使用简单神经网络
            output = self.pbm(x)
            output = self.output_projection(output)
        else:
            # 使用完整PBM
            # 优先从kwargs中获取未标准化的物理驱动(raw_features_last: [batch, 3])
            raw_feats = kwargs.get('raw_features_last', None)
            if raw_feats is not None:
                # raw顺序与数据集feature_cols一致: [pet, precip, temp]
                pet = raw_feats[:, 0]
                precip = raw_feats[:, 1]
                temp = raw_feats[:, 2]
            else:
                # 回退：从模块输入中取前三维（可能是编码特征，物理意义较弱）
                pet = x[:, 0] if x.shape[1] > 0 else torch.zeros(batch_size, device=device)
                precip = x[:, 1] if x.shape[1] > 1 else torch.zeros(batch_size, device=device)
                temp = x[:, 2] if x.shape[1] > 2 else torch.zeros(batch_size, device=device)

            # 构建PBM输入（确保为设备上的一维张量）
            pbm_inputs = {
                'precip': precip,
                'temp': temp,
                'pet': pet,
                'time_step': torch.zeros(batch_size, dtype=torch.long, device=device)
            }
            
            # 🚀 修复：获取实际的站点ID字符串，而不是索引
            station_ids_str = kwargs.get('station_ids_str', None)
            station_ids_idx = kwargs.get('station_ids', torch.zeros(batch_size, dtype=torch.long, device=device))
            
            # 如果有站点ID字符串，直接使用；否则使用默认站点
            if station_ids_str is not None and isinstance(station_ids_str, (list, tuple)):
                # 直接使用字符串ID列表
                pass
            else:
                # 如果没有字符串ID，使用一个固定的测试站点
                station_ids_str = ["camels_09378630"] * batch_size
            
            # 运行PBM
            with torch.no_grad():  # PBM不需要梯度
                pbm_outputs = self.pbm(pbm_inputs, station_ids_idx, station_ids_str)
            
            # 根据模块类型选择输出
            if self.module_type == 'snow':
                output = pbm_outputs['snow_output']
            elif self.module_type == 'runoff':
                output = pbm_outputs['runoff_output']
            elif self.module_type == 'et':
                output = pbm_outputs['et_output']
            elif self.module_type == 'drainage':
                output = pbm_outputs['groundwater_output']
            else:
                output = pbm_outputs['runoff_output']  # 默认
            
            # 投影到标准输出维度
            output = self.output_projection(output.unsqueeze(-1))
        
        return output


class ModuleGate(nn.Module):
    """模块门控 - 在PBM和NN专家之间选择"""
    
    def __init__(self, input_dim: int, num_experts: int = 2, dropout: float = 0.1, 
                 pbm_min_weight: float = 0.0, top_k: int = 1, temperature: float = 0.7):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.pbm_min_weight = pbm_min_weight  # PBM专家最小权重约束（默认不启用）
        self.top_k = max(1, min(int(top_k), num_experts))
        self.temperature = max(1e-6, float(temperature))
        
        # 🚀 改进的门控网络 - 增强表达能力，允许专家差异化
        hidden_dim = max(input_dim // 2, 32)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加归一化稳定训练
            nn.GELU(),  # 使用GELU激活函数，更好的梯度特性
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # 🚀 专家偏好引导机制
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        
        # 🚀 专家质量评估机制（可选）
        self.enable_quality_guidance = True
        if self.enable_quality_guidance:
            self.quality_tracker = nn.Parameter(torch.ones(num_experts), requires_grad=False)
            self.quality_momentum = 0.95  # 动量系数
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化 - 允许专家差异化的初始化"""
        for i, module in enumerate(self.modules()):
            if isinstance(module, nn.Linear):
                # 🚀 增加初始化强度，允许更大的初始logits差异
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # 从0.1增加到0.5
                if module.bias is not None:
                    # 🚀 添加小的随机偏置，打破对称性
                    nn.init.uniform_(module.bias, -0.1, 0.1)  # 而不是全零初始化
    
    def forward(self, features: torch.Tensor, pbm_output: torch.Tensor, 
                nn_output: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            features: [batch_size, input_dim] 输入特征
            pbm_output: [batch_size, output_dim] PBM专家输出
            nn_output: [batch_size, output_dim] NN专家输出
            
        Returns:
            混合输出和门控信息
        """
        # 计算门控权重
        gate_logits = self.gate(features)  # [batch_size, num_experts]
        
        # 🚀 添加专家偏好偏置，鼓励差异化
        gate_logits = gate_logits + self.expert_bias.unsqueeze(0)
        
        # 🚀 可选：根据专家历史质量调整logits
        if self.enable_quality_guidance and hasattr(self, 'quality_tracker'):
            # 质量越高的专家获得更高的logits偏置
            quality_bias = (self.quality_tracker - self.quality_tracker.mean()) * 0.5
            gate_logits = gate_logits + quality_bias.unsqueeze(0)
        
        # 🚀 降低温度，增强选择性（从0.7降到0.3）
        effective_temperature = max(0.3, self.temperature)
        gate_weights = F.softmax(gate_logits / effective_temperature, dim=-1)  # [batch_size, num_experts]
        # 形状稳健：若门控输出维度异常，取 Top-2 并归一化
        if gate_weights.size(-1) != self.num_experts:
            top_k = min(self.num_experts, gate_weights.size(-1))
            top_w, _ = torch.topk(gate_weights, k=top_k, dim=-1)
            gate_weights = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 应用PBM最小权重约束（如启用）
        if self.pbm_min_weight > 0:
            pbm_weights = gate_weights[:, 0]
            pbm_weights_constrained = torch.clamp(pbm_weights, min=self.pbm_min_weight)
            rest = torch.clamp(1.0 - pbm_weights_constrained, min=0.0)
            # 将余量平均分配给其余专家（当前为2专家时即另一个）
            if self.num_experts > 1:
                others = gate_weights[:, 1:]
                others_sum = others.sum(dim=1, keepdim=True) + 1e-8
                others_norm = others / others_sum
                others_new = others_norm * rest
                gate_weights = torch.cat([pbm_weights_constrained.unsqueeze(1), others_new], dim=1)

        # Top-k 策略：k=1 使用硬选择（one-hot），k>1 使用软加权
        if self.top_k == 1:
            # Straight-Through Gumbel-Softmax（温度可控）
            # 训练时近似 one-hot，反向用soft梯度，避免早期冻结
            gumbel = -torch.log(-torch.log(torch.rand_like(gate_logits).clamp_(1e-9, 1 - 1e-9)))
            y_soft = F.softmax((gate_logits + gumbel) / self.temperature, dim=-1)
            idx = torch.argmax(y_soft, dim=-1)
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(1, idx.unsqueeze(1), 1.0)
            gate_weights = (y_hard - y_soft).detach() + y_soft
        
        # 专家输出堆叠
        # 确保专家输出为 [B, output_dim]
        if pbm_output.dim() == 1:
            pbm_output = pbm_output.unsqueeze(-1)
        if nn_output.dim() == 1:
            nn_output = nn_output.unsqueeze(-1)
        expert_outputs = torch.stack([pbm_output, nn_output], dim=1)  # 期望 [B, 2, output_dim]
        # 若专家维度与输出维度被误置换，自动纠正
        if expert_outputs.size(1) != self.num_experts and expert_outputs.size(-1) == self.num_experts:
            expert_outputs = expert_outputs.transpose(1, 2)
        
        # 加权组合（自适应识别专家维度位置，避免维度错置）
        if expert_outputs.dim() != 3:
            raise RuntimeError(f"expert_outputs 维度异常: {expert_outputs.shape}")
        if expert_outputs.size(1) == self.num_experts:
            outputs_expert_first = expert_outputs  # [B, K, D]
        elif expert_outputs.size(2) == self.num_experts:
            outputs_expert_first = expert_outputs.transpose(1, 2)  # [B, K, D]
        else:
            # 无法识别，强制将最后一维视作特征，第二维聚合为专家数
            if expert_outputs.size(1) != self.num_experts:
                # 尝试切到前K
                k = min(self.num_experts, expert_outputs.size(1))
                outputs_expert_first = expert_outputs[:, :k, :]
                gate_weights = gate_weights[:, :k]
            else:
                outputs_expert_first = expert_outputs
        # 对齐 gate_weights 的专家维度到 outputs_expert_first 的 K 维
        K_out = outputs_expert_first.size(1)
        if gate_weights.size(1) != K_out:
            if gate_weights.size(1) > K_out:
                # 取前K_out大的权重并归一化
                top_w, _ = torch.topk(gate_weights, k=K_out, dim=-1)
                gate_weights = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                # 填充到K_out并归一化
                B = gate_weights.size(0)
                pad = torch.zeros(B, K_out - gate_weights.size(1), device=gate_weights.device, dtype=gate_weights.dtype)
                gate_weights = torch.cat([gate_weights, pad], dim=-1)
                gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
        # einsum 做加权求和，避免显式 expand
        mixed_output = torch.einsum('bkd,bk->bd', outputs_expert_first, gate_weights)  # [B, D]
        
        # 门控信息
        gate_info = {
            'gate_weights': gate_weights,
            'pbm_weight': gate_weights[:, 0].mean().item(),
            'nn_weight': gate_weights[:, 1].mean().item(),
            'effective_gate': gate_weights
        }
        
        return mixed_output, gate_info


class HybridHydroMoEModel(nn.Module):
    """
    混合水文MoE模型 - 集成PBM物理模块和神经网络专家
    
    架构：特征编码 → 自注意力 → 四个水文模块（PBM+NN） → 最终组合
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 兼容字典和对象配置
        if isinstance(config, dict):
            self.config = config
            self.model_config = config.get('model', config)
            self.pbm_config = config.get('pbm', {})
        else:
            self.config = config
            self.model_config = getattr(config, 'model', config)
            self.pbm_config = getattr(config, 'pbm', {})
        
        # 🚀 性能优化开关
        self.use_gradient_checkpointing = os.getenv('USE_GRAD_CHECKPOINT', '1').lower() in ['1', 'true', 'yes']
        
        # 初始化模型组件
        self._initialize_model()
    
    def _get_config_value(self, key, default=None):
        """辅助函数：从配置中获取值，兼容字典和对象"""
        if isinstance(self.model_config, dict):
            return self.model_config.get(key, default)
        else:
            return getattr(self.model_config, key, default)
    
    def _initialize_model(self):
        """初始化模型组件"""
        # 模型维度参数
        self.input_dim = self._get_config_value('input_size', 20)
        self.d_model = self._get_config_value('hidden_size', 128)
        self.output_dim = 1
        
        # 获取其他配置参数
        dropout = self._get_config_value('dropout', 0.1)
        num_heads = self._get_config_value('num_heads', 8)
        num_attention_layers = self._get_config_value('num_attention_layers', 2)
        max_sequence_length = self._get_config_value('max_sequence_length', 256)
        
        # 1. 输入特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 自注意力机制层
        self.attention_blocks = nn.ModuleList([
            HydroAttentionBlock(
                d_model=self.d_model,
                n_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_sequence_length
            ) for _ in range(num_attention_layers)
        ])
        
        # 3. 四个水文模块（PBM + NN 专家）
        pbm_min_weight = self._get_config_value('pbm_min_weight', 0.0)
        module_gate_top_k = self._get_config_value('module_gate_top_k', 2)  # 🚀 修复：默认值应该是2
        
        self.snow_pbm_expert = PBMExpert(self.d_model, self.output_dim, 'snow')
        self.snow_nn_expert = MLPExpert(self.d_model, self.output_dim, 
                                       hidden_dim=self.d_model//2, num_layers=2)
        # 🚀 统一使用新的门控配置
        module_gate_temperature = self._get_config_value('module_gate_temperature', 0.3)
        
        self.snow_gate = ModuleGate(self.d_model, num_experts=2, pbm_min_weight=pbm_min_weight, top_k=module_gate_top_k, temperature=module_gate_temperature)
        
        self.runoff_pbm_expert = PBMExpert(self.d_model, self.output_dim, 'runoff')
        self.runoff_nn_expert = MLPExpert(self.d_model, self.output_dim,
                                         hidden_dim=self.d_model//2, num_layers=2)
        self.runoff_gate = ModuleGate(self.d_model, num_experts=2, pbm_min_weight=pbm_min_weight, top_k=module_gate_top_k, temperature=module_gate_temperature)
        
        self.et_pbm_expert = PBMExpert(self.d_model, self.output_dim, 'et')
        self.et_nn_expert = MLPExpert(self.d_model, self.output_dim,
                                     hidden_dim=self.d_model//2, num_layers=2)
        self.et_gate = ModuleGate(self.d_model, num_experts=2, pbm_min_weight=pbm_min_weight, top_k=module_gate_top_k, temperature=module_gate_temperature)
        
        self.drainage_pbm_expert = PBMExpert(self.d_model, self.output_dim, 'drainage')
        self.drainage_nn_expert = MLPExpert(self.d_model, self.output_dim,
                                           hidden_dim=self.d_model//2, num_layers=2)
        self.drainage_gate = ModuleGate(self.d_model, num_experts=2, pbm_min_weight=pbm_min_weight, top_k=module_gate_top_k, temperature=module_gate_temperature)
        
        # 4. 增强最终组合器 - 扩大预测动态范围
        # 保留原 MLP 定义以兼容（但不再使用作为最终组合）
        self.final_combiner = nn.Sequential(
            nn.Linear(4 * self.output_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.LayerNorm(self.d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(self.d_model // 4, self.output_dim)
        )

        # 新增：二路凸组合权重头（生成快流/基流的分配系数 α）
        alpha_hidden = max(self.d_model // 4, 32)
        # 输入为 [module_input(d_model), snow_out, et_out] → 2
        self.alpha_head = nn.Sequential(
            nn.Linear(self.d_model + 2, alpha_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(alpha_hidden, 2)
        )

        # 新增：可用水映射头（将原始物理驱动映射到预测空间的上限标量，保证非负）
        self.avail_head = nn.Sequential(
            nn.Linear(3, max(16, self.d_model // 8)),
            nn.GELU(),
            nn.Linear(max(16, self.d_model // 8), 1)
        )

        # 4.1 输出激活层（保留但在凸组合路径中不直接使用）
        self.output_activation = nn.Softplus(beta=0.1)
        
        # 序列聚合方式
        self.sequence_aggregation = self._get_config_value('sequence_aggregation', 'last')

        # 5. 径流分期 Regime-MoE 头（低/平/洪），以残差方式细化 base runoff（自由选择）
        self.use_regime_moe = True
        regime_hidden = max(self.d_model // 4, 32)
        # 使用小型 Transformer 编码器从注意力序列中提取上下文
        nheads = min(8, max(1, self.d_model // 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nheads,
            dim_feedforward=max(self.d_model * 2, 64),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.regime_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # 将编码后的序列池化并投影到原先的 regime_hidden 维度，保持下游结构不变
        self.regime_proj = nn.Linear(self.d_model, regime_hidden)
        # 门控由 Transformer 上下文产生
        self.regime_gate = nn.Linear(regime_hidden, 3)
        # 三个水期专家：NN（基于 LSTM 隐状态 + base_runoff）
        self.regime_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 + regime_hidden, regime_hidden),
                nn.ReLU(),
                nn.Linear(regime_hidden, self.output_dim)
            ) for _ in range(3)
        ])
        # 残差尺度（可训练，小幅度）
        self.regime_residual_scale = nn.Parameter(torch.tensor(0.05))
        # 门控温度（越小越尖锐）与top-k（1=硬路由）
        self.regime_temperature = float(self._get_config_value('regime_temperature', 0.8))
        self.regime_top_k = int(self._get_config_value('regime_top_k', 1))
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, batch: Dict[str, torch.Tensor], return_gate_info: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 包含'features'等键的批次数据
            return_gate_info: 是否返回门控信息
            
        Returns:
            包含'runoff'和可选'gate_info'的输出字典
        """
        features = batch['features']  # [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = features.shape
        
        # 🚀 修复：获取实际的站点ID字符串
        station_ids_str = batch.get('station_id', None)  # 字符串列表
        station_ids = batch.get('station_idx', torch.zeros(batch_size, dtype=torch.long, device=features.device))
        if isinstance(station_ids, torch.Tensor):
            station_ids = station_ids.view(-1).to(dtype=torch.long, device=features.device)
        
        # 确保输入维度正确
        if input_dim != self.input_dim:
            # 动态调整输入编码器
            self.feature_encoder[0] = nn.Linear(input_dim, self.d_model).to(features.device)
            self.input_dim = input_dim
        
        # 1. 特征编码
        encoded_features = self.feature_encoder(features)  # [batch_size, seq_len, d_model]
        
        # 2. 自注意力机制处理
        # 🚀 优化：使用gradient checkpointing减少显存占用
        attention_output = encoded_features
        if self.training and self.use_gradient_checkpointing:
            for attention_block in self.attention_blocks:
                attention_output = torch.utils.checkpoint.checkpoint(attention_block, attention_output, use_reentrant=False)
        else:
            for attention_block in self.attention_blocks:
                attention_output = attention_block(attention_output)
        
        # 3. 序列聚合（获取单个时间步的表示）
        if self.sequence_aggregation == 'last':
            module_input = attention_output[:, -1, :]  # [batch_size, d_model]
        elif self.sequence_aggregation == 'mean':
            module_input = attention_output.mean(dim=1)  # [batch_size, d_model]
        else:
            module_input = attention_output[:, -1, :]  # 默认使用最后一个时间步
        
        # 4. 四个水文模块处理
        gate_infos = {}
        
        # Snow模块
        raw_features_last = batch.get('raw_features_last', None)
        if isinstance(raw_features_last, torch.Tensor):
            if raw_features_last.dim() == 1:
                raw_features_last = raw_features_last.unsqueeze(0).repeat(batch_size, 1)
            elif raw_features_last.dim() == 2 and raw_features_last.shape[0] == 1 and batch_size > 1:
                raw_features_last = raw_features_last.repeat(batch_size, 1)
        # 🚀 传入实际的站点ID字符串给PBM专家
        pbm_kwargs = {
            'station_ids': station_ids, 
            'station_ids_str': station_ids_str,
            'raw_features_last': raw_features_last
        }
        
        snow_pbm_out = self.snow_pbm_expert(module_input, **pbm_kwargs)
        snow_nn_out = self.snow_nn_expert(module_input)
        snow_output, snow_gate_info = self.snow_gate(module_input, snow_pbm_out, snow_nn_out)
        gate_infos['snow'] = snow_gate_info
        
        # Runoff模块
        runoff_pbm_out = self.runoff_pbm_expert(module_input, **pbm_kwargs)
        runoff_nn_out = self.runoff_nn_expert(module_input)
        runoff_output, runoff_gate_info = self.runoff_gate(module_input, runoff_pbm_out, runoff_nn_out)
        gate_infos['runoff'] = runoff_gate_info
        
        # ET模块
        et_pbm_out = self.et_pbm_expert(module_input, **pbm_kwargs)
        et_nn_out = self.et_nn_expert(module_input)
        et_output, et_gate_info = self.et_gate(module_input, et_pbm_out, et_nn_out)
        gate_infos['et'] = et_gate_info
        
        # Drainage模块
        drainage_pbm_out = self.drainage_pbm_expert(module_input, **pbm_kwargs)
        drainage_nn_out = self.drainage_nn_expert(module_input)
        drainage_output, drainage_gate_info = self.drainage_gate(module_input, drainage_pbm_out, drainage_nn_out)
        gate_infos['drainage'] = drainage_gate_info
        
        # 5. 最终组合（物理化）：二路凸组合（runoff/drainage）+ 可用水上限 A
        # 5.1 生成分配系数 α（非负且和为1），融雪/蒸散作为调制因子
        alpha_in = torch.cat([module_input, snow_output, et_output], dim=-1)  # [B, d_model+2]
        alpha_logits = self.alpha_head(alpha_in)  # [B,2]
        alpha = torch.softmax(alpha_logits, dim=-1)  # [B,2]

        # 5.2 凸组合的基础径流
        # 对分量施加非负性，避免负值被凸组合放大
        q_quick = F.softplus(runoff_output)  # [B,1]
        q_base = F.softplus(drainage_output)  # [B,1]
        q_comb = alpha[:, 0:1] * q_quick + alpha[:, 1:2] * q_base  # [B,1]

        # 5.3 可用水上限 A（平滑守恒）：A ≈ ReLU(precip + snow - pet)
        # 尽量使用未标准化物理驱动
        precip_raw = None
        pet_raw = None
        if raw_features_last is not None and isinstance(raw_features_last, torch.Tensor):
            try:
                pet_raw = raw_features_last[:, 0].reshape(-1, 1)  # [B,1]
                precip_raw = raw_features_last[:, 1].reshape(-1, 1)  # [B,1]
            except Exception:
                pass
        if precip_raw is None:
            precip_raw = torch.zeros_like(q_comb)
        if pet_raw is None:
            pet_raw = torch.zeros_like(q_comb)
        snow_pos = torch.relu(snow_output)  # [B,1]
        A_raw = torch.relu(precip_raw + snow_pos - pet_raw)  # [B,1]
        # 将原始A映射到预测空间，保证非负
        A_in = torch.cat([precip_raw, snow_pos, pet_raw], dim=-1)  # [B,3]
        A_mapped = F.softplus(self.avail_head(A_in))  # [B,1]

        # 5.4 软上限：final = A_mapped - Softplus(A_mapped - q_comb)
        final_output = A_mapped - F.softplus(A_mapped - q_comb)

        # 5.1 Regime-MoE 输出细化（残差到 base runoff）
        regime_debug = None
        weights = None
        if self.use_regime_moe:
            # 使用 Transformer 编码器提取时序上下文
            enc_out = self.regime_encoder(attention_output)  # [batch, seq, d_model]
            # 全局平均池化得到上下文，再投影到原先 hidden 维度
            enc_ctx = enc_out.mean(dim=1)  # [batch, d_model]
            regime_ctx = self.regime_proj(enc_ctx)  # [batch, regime_hidden]

            base_runoff = final_output  # [batch, 1]

            # 门控权重
            logits = self.regime_gate(regime_ctx)  # [batch, 3]
            weights = torch.softmax(logits / max(self.regime_temperature, 1e-6), dim=-1)  # [batch, 3]

            # Regime top-k: 1 => 硬路由；>1 => 软加权
            if self.regime_top_k == 1:
                with torch.no_grad():
                    idx = torch.argmax(weights, dim=-1)
                one_hot = torch.zeros_like(weights)
                one_hot.scatter_(1, idx.unsqueeze(1), 1.0)
                weights = one_hot

            # 三个专家基于 [base_runoff, regime_ctx]
            expert_outs = []
            regime_input = torch.cat([base_runoff, regime_ctx], dim=-1)  # [batch, 1+hidden]
            for i in range(3):
                expert_out = self.regime_experts[i](regime_input)  # [batch, 1]
                # 对专家输出也应用激活函数
                expert_outs.append(self.output_activation(expert_out))
            experts_stack = torch.stack(expert_outs, dim=-1)  # [batch, 1, 3]

            # 加权求和得到残差
            weights_exp = weights.unsqueeze(1)  # [batch, 1, 3]
            regime_residual = (experts_stack * weights_exp).sum(dim=-1)  # [batch, 1]
            final_output = base_runoff + self.regime_residual_scale * regime_residual

            # 最终确保输出非负
            final_output = torch.clamp(final_output, min=0.0)

            # 收集调试信息
            regime_debug = {
                'weights': weights,
                'weights_mean': weights.mean(dim=0),
                'residual_scale': self.regime_residual_scale.detach().clone()
            }
        
        # 确保输出为标量（如果output_dim=1）
        if self.output_dim == 1:
            final_output = final_output.squeeze(-1)  # [batch_size]
        
        # 构建输出字典
        output = {
            'runoff': final_output,
            'regime_weights': weights,
            'alpha_weights': alpha,
            'available_water': A_mapped
        }
        
        if return_gate_info:
            # 处理门控信息
            output['gate_info'] = {
                'module_gates': gate_infos,
                'expert_usage': self._compute_expert_usage(gate_infos),
                'load_balancing_loss': self._compute_load_balancing_loss(gate_infos)
            }
            if regime_debug is not None:
                output['gate_info']['regime'] = regime_debug
        
        return output
    
    def _compute_expert_usage(self, gate_infos: Dict) -> Dict[str, float]:
        """计算专家使用统计"""
        expert_usage = {}
        for module_name, gate_info in gate_infos.items():
            pbm_usage = gate_info['pbm_weight']
            nn_usage = gate_info['nn_weight']
            expert_usage[f'{module_name}_pbm'] = pbm_usage
            expert_usage[f'{module_name}_nn'] = nn_usage
        return expert_usage
    
    def _compute_load_balancing_loss(self, gate_infos: Dict) -> torch.Tensor:
        """计算负载均衡损失"""
        total_loss = 0.0
        for module_name, gate_info in gate_infos.items():
            gate_weights = gate_info['gate_weights']  # [batch_size, num_experts]
            # 计算每个专家的平均使用率
            expert_usage = gate_weights.mean(dim=0)  # [num_experts]
            # 理想情况下每个专家的使用率应该是 0.5
            ideal_usage = 0.5
            # 计算方差作为负载均衡损失
            loss = torch.var(expert_usage) / (ideal_usage ** 2)
            total_loss += loss
        
        return total_loss / len(gate_infos)


def create_hybrid_hydro_moe_model(config) -> HybridHydroMoEModel:
    """创建混合HydroMoE模型的工厂函数"""
    return HybridHydroMoEModel(config)


if __name__ == "__main__":
    # 测试混合模型
    print("🧪 测试混合HydroMoE模型...")
    
    # 简化配置
    class TestConfig:
        def __init__(self):
            self.model = TestModelConfig()
    
    class TestModelConfig:
        def __init__(self):
            self.input_size = 3
            self.d_model = 64
            self.num_heads = 4
            self.num_attention_layers = 2
            self.dropout = 0.1
            self.max_sequence_length = 100
            self.sequence_aggregation = 'last'
    
    config = TestConfig()
    model = HybridHydroMoEModel(config)
    
    # 测试数据
    batch_size = 4
    seq_len = 10
    batch = {
        'features': torch.randn(batch_size, seq_len, 3),
        'station_idx': torch.tensor([1, 2, 3, 4])
    }
    
    # 前向传播
    output = model(batch, return_gate_info=True)
    
    print(f" 输出形状: {output['runoff'].shape}")
    print(f" 专家使用统计: {output['gate_info']['expert_usage']}")
    print(" 混合HydroMoE模型测试成功！")