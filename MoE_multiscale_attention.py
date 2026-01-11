"""
多尺度时序注意力模块 - 捕获不同时间尺度的水文过程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class MultiScaleTemporalAttention(nn.Module):
    """多尺度时序注意力机制"""
    
    def __init__(self, 
                 d_model: int = 128,
                 num_heads: int = 8,
                 scales: List[int] = [1, 3, 7, 14],  # 日、3日、周、双周
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.scales = scales
        self.num_scales = len(scales)
        
        # 每个尺度的注意力头
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads // self.num_scales,
                dropout=dropout,
                batch_first=True
            ) for _ in scales
        ])
        
        # 尺度融合网络
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * self.num_scales, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 可选的注意力掩码
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        scale_outputs = []
        
        for i, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            # 对于不同尺度，使用不同的下采样策略
            if scale == 1:
                # 原始尺度
                scale_input = x
            else:
                # 下采样到对应尺度
                scale_input = self._downsample(x, scale)
            
            # 应用注意力
            attn_output, _ = attention(scale_input, scale_input, scale_input, attn_mask=mask)
            
            # 上采样回原始尺度
            if scale != 1:
                attn_output = self._upsample(attn_output, seq_len)
            
            scale_outputs.append(attn_output)
        
        # 融合不同尺度的输出
        fused_output = torch.cat(scale_outputs, dim=-1)  # [batch, seq, d_model * num_scales]
        fused_output = self.scale_fusion(fused_output)   # [batch, seq, d_model]
        
        # 残差连接和层归一化
        output = self.layer_norm(x + fused_output)
        
        return output
    
    def _downsample(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """下采样到指定尺度"""
        batch_size, seq_len, d_model = x.shape
        
        # 使用平均池化进行下采样
        if seq_len % scale != 0:
            # 填充到scale的倍数
            pad_len = scale - (seq_len % scale)
            x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            seq_len = x.shape[1]
        
        # 重塑并平均
        x_reshaped = x.view(batch_size, seq_len // scale, scale, d_model)
        downsampled = x_reshaped.mean(dim=2)  # [batch, seq_len//scale, d_model]
        
        return downsampled
    
    def _upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """上采样到目标长度"""
        batch_size, current_len, d_model = x.shape
        
        if current_len == target_len:
            return x
        
        # 使用线性插值上采样
        x_permuted = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        upsampled = F.interpolate(x_permuted, size=target_len, mode='linear', align_corners=False)
        upsampled = upsampled.permute(0, 2, 1)  # [batch, target_len, d_model]
        
        return upsampled


class HierarchicalTemporalEncoder(nn.Module):
    """分层时序编码器"""
    
    def __init__(self,
                 d_model: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 多个多尺度注意力层
        self.layers = nn.ModuleList([
            MultiScaleTemporalAttention(
                d_model=d_model,
                num_heads=num_heads,
                scales=[1, 3, 7, 14] if i == 0 else [1, 2, 4, 8],  # 不同层使用不同尺度
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        # 前馈网络
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers * 2)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        for i, (layer, ffn) in enumerate(zip(self.layers, self.ffns)):
            # 多尺度注意力
            x = layer(x)
            
            # 前馈网络
            ffn_input = self.layer_norms[i * 2 + 1](x)
            ffn_output = ffn(ffn_input)
            x = self.layer_norms[i * 2 + 1](x + ffn_output)
        
        return x


class AdaptiveTemporalPooling(nn.Module):
    """自适应时序池化"""
    
    def __init__(self, d_model: int = 128, pool_sizes: List[int] = [1, 3, 7, 14]):
        super().__init__()
        
        self.d_model = d_model
        self.pool_sizes = pool_sizes
        
        # 每个池化尺度的权重网络
        self.pool_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            ) for _ in pool_sizes
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * len(pool_sizes), d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        pooled_features = []
        
        for pool_size, weight_net in zip(self.pool_sizes, self.pool_weights):
            if pool_size == 1:
                # 全局平均池化
                pooled = x.mean(dim=1)  # [batch, d_model]
            else:
                # 自适应池化
                x_permuted = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
                pooled = F.adaptive_avg_pool1d(x_permuted, pool_size)  # [batch, d_model, pool_size]
                pooled = pooled.mean(dim=-1)  # [batch, d_model]
            
            # 计算权重
            weight = weight_net(pooled.unsqueeze(-1)).squeeze(-1)  # [batch, 1]
            
            # 加权特征
            weighted_feature = pooled * weight
            pooled_features.append(weighted_feature)
        
        # 融合所有池化特征
        fused_features = torch.cat(pooled_features, dim=-1)  # [batch, d_model * num_pools]
        output = self.feature_fusion(fused_features)  # [batch, d_model]
        
        return output


class SeasonalAwareAttention(nn.Module):
    """季节感知注意力机制"""
    
    def __init__(self, d_model: int = 128, num_seasons: int = 4):
        super().__init__()
        
        self.d_model = d_model
        self.num_seasons = num_seasons
        
        # 季节嵌入
        self.season_embedding = nn.Embedding(num_seasons, d_model)
        
        # 季节特定的注意力权重
        self.season_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        
        # 季节调制网络
        self.season_modulation = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, season_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            season_ids: [batch_size, seq_len] 季节ID (0-3)
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 获取季节嵌入
        season_emb = self.season_embedding(season_ids)  # [batch, seq_len, d_model]
        
        # 季节感知注意力
        combined_input = x + season_emb
        attn_output, _ = self.season_attention(combined_input, combined_input, combined_input)
        
        # 季节调制
        modulation_input = torch.cat([x, season_emb], dim=-1)  # [batch, seq_len, d_model*2]
        modulation_weights = self.season_modulation(modulation_input)  # [batch, seq_len, d_model]
        
        # 应用调制
        output = attn_output * modulation_weights + x * (1 - modulation_weights)
        
        return output


if __name__ == "__main__":
    # 测试多尺度时序注意力
    print("🧪 测试多尺度时序注意力...")
    
    batch_size = 4
    seq_len = 96
    d_model = 128
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    season_ids = torch.randint(0, 4, (batch_size, seq_len))
    
    # 测试多尺度注意力
    multiscale_attn = MultiScaleTemporalAttention(d_model=d_model)
    output1 = multiscale_attn(x)
    print(f"多尺度注意力输出形状: {output1.shape}")
    
    # 测试分层编码器
    hierarchical_encoder = HierarchicalTemporalEncoder(d_model=d_model)
    output2 = hierarchical_encoder(x)
    print(f"分层编码器输出形状: {output2.shape}")
    
    # 测试自适应池化
    adaptive_pooling = AdaptiveTemporalPooling(d_model=d_model)
    output3 = adaptive_pooling(x)
    print(f"自适应池化输出形状: {output3.shape}")
    
    # 测试季节感知注意力
    seasonal_attn = SeasonalAwareAttention(d_model=d_model)
    output4 = seasonal_attn(x, season_ids)
    print(f"季节感知注意力输出形状: {output4.shape}")
    
    print("✅ 多尺度时序注意力测试完成！")
