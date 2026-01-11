"""
专家网络模块 - 不同类型的专家网络实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from MoE_attention import RMSNorm


class BaseExpert(nn.Module, ABC):
    """专家网络基类"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MLPExpert(BaseExpert):
    """多层感知机专家"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__(input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 🚀 优化：使用inplace激活函数，节省显存
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # 构建网络层
        layers = []
        
        # 输入层
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout)
        ])
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConvExpert(BaseExpert):
    """卷积专家 - 适用于时间序列模式识别"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_channels: int = 64,
                 kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.1):
        super().__init__(input_dim, output_dim)
        
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        
        # 多尺度卷积分支
        self.conv_branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(input_dim, hidden_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_channels, hidden_channels//2, 1),
                nn.BatchNorm1d(hidden_channels//2),
                nn.ReLU()
            )
            self.conv_branches.append(branch)
        
        # 特征融合
        total_channels = len(kernel_sizes) * (hidden_channels // 2)
        self.feature_fusion = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(total_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, input_dim] 或 [batch_size, seq_len, input_dim]
            
        Returns:
            output: [batch_size, output_dim]
        """
        # 处理输入维度
        if len(x.shape) == 2:
            # 如果是2D，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # 转置为卷积格式 [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)  # [batch_size, hidden_channels//2, seq_len]
            branch_outputs.append(branch_out)
        
        # 拼接所有分支
        concat_features = torch.cat(branch_outputs, dim=1)  # [batch_size, total_channels, seq_len]
        
        # 特征融合和输出
        output = self.feature_fusion(concat_features)
        
        return output


class AttentionExpert(BaseExpert):
    """注意力专家 - 专注于时间序列的关键特征"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__(input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 归一化（Pre-RMSNorm）
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 输出层（不再额外做LayerNorm）
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, input_dim] 或 [batch_size, seq_len, input_dim]
            
        Returns:
            output: [batch_size, output_dim]
        """
        # 处理输入维度
        if len(x.shape) == 2:
            # 如果是2D，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        batch_size, seq_len, _ = x.shape
        
        # 自注意力（Pre-Norm）
        x_proj = self.input_projection(x)  # [batch, seq, hidden]
        attn_input = self.norm1(x_proj)
        attn_output, _ = self.self_attention(attn_input, attn_input, attn_input)
        x = x_proj + self.dropout(attn_output)

        # 前馈网络（Pre-Norm）
        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.dropout(ffn_output)
        
        # 全局平均池化（如果有多个时间步）
        if seq_len > 1:
            x = x.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # 输出
        output = self.output_layer(x)
        return output


class HydrologySpecificExpert(BaseExpert):
    """水文学专用专家 - 集成水文学先验知识"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 expert_type: str = 'runoff',
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__(input_dim, output_dim)
        
        self.expert_type = expert_type
        self.hidden_dim = hidden_dim
        
        # 根据专家类型设计不同的网络结构
        if expert_type == 'runoff':
            # 径流专家：关注降水、土壤湿度、地形特征
            self.feature_extractor = self._build_runoff_extractor()
        elif expert_type == 'evapotranspiration':
            # 蒸散发专家：关注温度、湿度、太阳辐射
            self.feature_extractor = self._build_et_extractor()
        elif expert_type == 'snowmelt':
            # 融雪专家：关注温度梯度、雪深、能量平衡
            self.feature_extractor = self._build_snow_extractor()
        elif expert_type == 'baseflow':
            # 基流专家：关注地下水、长期趋势
            self.feature_extractor = self._build_baseflow_extractor()
        else:
            # 通用专家
            self.feature_extractor = self._build_general_extractor()
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._init_weights()
    
    def _build_runoff_extractor(self):
        """构建径流特征提取器"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 非线性变换模拟降水-径流关系
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),  # 限制输出范围
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_et_extractor(self):
        """构建蒸散发特征提取器"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 模拟彭曼方程的非线性关系
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),  # 蒸散发总是正值
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_snow_extractor(self):
        """构建融雪特征提取器"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 度日因子模型的非线性扩展
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),  # 融雪速率非负
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_baseflow_extractor(self):
        """构建基流特征提取器"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 指数衰减特征模拟地下水释放
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),  # 平滑的指数特征
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_general_extractor(self):
        """构建通用特征提取器"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取
        features = self.feature_extractor(x)
        
        # 输出预测
        output = self.output_layer(features)
        
        return output


def create_expert(expert_type: str,
                 input_dim: int,
                 output_dim: int,
                 config: Dict[str, Any]) -> BaseExpert:
    """
    专家工厂函数

    Args:
        expert_type: 专家类型 ('mlp', 'conv', 'attention', 'hydrology', 'flow_regime', 'seasonal')
        input_dim: 输入维度
        output_dim: 输出维度
        config: 配置参数

    Returns:
        创建的专家网络
    """
    if expert_type == 'mlp':
        return MLPExpert(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'relu')
        )

    elif expert_type == 'conv':
        return ConvExpert(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_channels=config.get('hidden_channels', 64),
            kernel_sizes=config.get('kernel_sizes', [3, 5, 7]),
            dropout=config.get('dropout', 0.1)
        )

    elif expert_type == 'attention':
        return AttentionExpert(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.get('hidden_dim', 128),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1)
        )

    elif expert_type == 'hydrology':
        return HydrologySpecificExpert(
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type=config.get('hydrology_type', 'runoff'),
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.1)
        )

    elif expert_type == 'flow_regime':
        return FlowRegimeExpert(
            input_dim=input_dim,
            output_dim=output_dim,
            regime_type=config.get('regime_type', 'medium'),
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.1)
        )

    elif expert_type == 'seasonal':
        return SeasonalExpert(
            input_dim=input_dim,
            output_dim=output_dim,
            season_type=config.get('season_type', 'spring'),
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.1)
        )

    else:
        raise ValueError(f"Unknown expert type: {expert_type}")


class FlowRegimeExpert(BaseExpert):
    """流量分级专家 - 专门处理不同流量级别"""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 regime_type: str = 'low',  # 'low', 'medium', 'high', 'extreme'
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__(input_dim, output_dim)

        self.regime_type = regime_type
        self.hidden_dim = hidden_dim

        # 根据流量级别设计不同的网络结构
        if regime_type == 'low':
            # 低流量专家：关注基流、蒸散发
            self.network = self._build_low_flow_network()
        elif regime_type == 'medium':
            # 中等流量专家：关注常规降水-径流关系
            self.network = self._build_medium_flow_network()
        elif regime_type == 'high':
            # 高流量专家：关注洪峰、快速响应
            self.network = self._build_high_flow_network()
        elif regime_type == 'extreme':
            # 极端流量专家：关注极端事件
            self.network = self._build_extreme_flow_network()
        else:
            raise ValueError(f"Unknown regime type: {regime_type}")

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _build_low_flow_network(self):
        """构建低流量网络 - 平滑、稳定的响应"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),  # 平滑激活
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ELU(),   # 平滑的指数特征，适合基流
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            nn.Softplus()  # 确保正值输出
        )

    def _build_medium_flow_network(self):
        """构建中等流量网络 - 标准的非线性响应"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )

    def _build_high_flow_network(self):
        """构建高流量网络 - 快速响应、非线性增强"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),  # 更强的非线性
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),  # 更大容量
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def _build_extreme_flow_network(self):
        """构建极端流量网络 - 处理异常值和极端响应"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),  # 允许负值传播
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.Swish(),  # 平滑但有界的激活
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU()  # 确保非负输出
        )

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.regime_type == 'low':
                    # 低流量：小权重，稳定初始化
                    nn.init.xavier_uniform_(module.weight, gain=0.3)
                elif self.regime_type == 'extreme':
                    # 极端流量：较大权重，增强表达能力
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                else:
                    # 中等和高流量：标准初始化
                    nn.init.xavier_uniform_(module.weight, gain=0.5)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SeasonalExpert(BaseExpert):
    """季节性专家 - 处理季节性水文过程"""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 season_type: str = 'spring',  # 'spring', 'summer', 'autumn', 'winter'
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__(input_dim, output_dim)

        self.season_type = season_type
        self.hidden_dim = hidden_dim

        # 季节特定的网络结构
        if season_type == 'spring':
            # 春季：融雪、降水增加
            self.network = self._build_spring_network()
        elif season_type == 'summer':
            # 夏季：高蒸散发、雷暴
            self.network = self._build_summer_network()
        elif season_type == 'autumn':
            # 秋季：稳定降水、蒸散发减少
            self.network = self._build_autumn_network()
        elif season_type == 'winter':
            # 冬季：雪积累、低蒸散发
            self.network = self._build_winter_network()
        else:
            raise ValueError(f"Unknown season type: {season_type}")

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _build_spring_network(self):
        """春季网络 - 处理融雪和降水"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 温度敏感层（融雪）
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),  # 温度阈值效应
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def _build_summer_network(self):
        """夏季网络 - 处理高蒸散发和雷暴"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),  # 非线性蒸散发关系
            nn.Dropout(0.15),
            # 蒸散发抑制层
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),  # 有界激活，模拟蒸散发上限
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def _build_autumn_network(self):
        """秋季网络 - 稳定的水文过程"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )

    def _build_winter_network(self):
        """冬季网络 - 处理雪积累和低活动"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),  # 低活动期的小梯度
            nn.Dropout(0.05),   # 较少dropout，保持稳定
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),  # 平滑激活
            nn.Dropout(0.05),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softplus()  # 确保非负
        )

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.season_type in ['winter', 'autumn']:
                    # 冬秋季：较小权重，稳定响应
                    nn.init.xavier_uniform_(module.weight, gain=0.3)
                else:
                    # 春夏季：标准权重
                    nn.init.xavier_uniform_(module.weight, gain=0.5)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)