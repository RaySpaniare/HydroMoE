"""
注意力机制模块 - 水文时间序列的自注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """RMSNorm实现，比LayerNorm更稳定且计算更高效"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 均方根归一化：x / rms(x)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)


def _make_norm(norm_type: str, dim: int):
    if norm_type.lower() == 'rms':
        return RMSNorm(dim)
    else:
        return nn.LayerNorm(dim)


class MultiHeadAttention(nn.Module):
    """多头自注意力机制 - 专为时间序列设计"""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 pre_norm: bool = True,
                 norm_type: str = 'rms'):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.norm = _make_norm(norm_type, d_model)
        
        # 数值稳定性初始化
        self._init_weights()
    
    def _init_weights(self):
        """保守的权重初始化"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 可选的注意力掩码
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 残余连接输入
        residual = x

        # Pre-Norm：先归一化再进入注意力
        x_in = self.norm(x) if self.pre_norm else x
        
        # 1. 线性变换得到Q, K, V
        Q = self.w_q(x_in)  # [batch, seq_len, d_model]
        K = self.w_k(x_in)
        V = self.w_v(x_in)
        
        # 2. 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch, n_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算注意力
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 5. 输出线性变换
        output = self.w_o(attention_output)
        output = self.dropout(output)
        
        # 残差
        output = residual + output
        
        # Post-Norm（如选择后归一化）
        if not self.pre_norm:
            output = self.norm(output)
        
        return output
    
    def _scaled_dot_product_attention(self, 
                                    Q: torch.Tensor, 
                                    K: torch.Tensor, 
                                    V: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """缩放点积注意力 - 优化版本"""
        
        # 🚀 优化：使用PyTorch 2.0+的scaled_dot_product_attention（如果可用）
        if hasattr(F, 'scaled_dot_product_attention') and mask is None:
            # 使用原生实现，更高效且节省显存
            context = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            return context
        
        # 1. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        # 2. 应用掩码（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        
        # 🚀 优化：简化clamp操作
        attention_weights = self.dropout(attention_weights)
        
        # 4. 加权求和
        context = torch.matmul(attention_weights, V)
        
        return context


class PositionalEncoding(nn.Module):
    """位置编码 - 为时间序列添加位置信息"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class HydroAttentionBlock(nn.Module):
    """水文注意力块 - 集成位置编码和多头注意力"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 pre_norm: bool = True,
                 norm_type: str = 'rms'):
        super().__init__()
        
        self.d_model = d_model
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 多头注意力
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, pre_norm=pre_norm, norm_type=norm_type)
        
        # 前馈网络（可选，用于更强的表达能力）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        self.pre_norm = pre_norm
        self.norm2 = _make_norm(norm_type, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：输入特征 → 位置编码 → 自注意力 → 前馈网络
        
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: 可选的注意力掩码
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 1. 位置编码
        x = self.pos_encoding(x)
        
        # 2. 自注意力 + 残差连接（归一化在注意力内部处理）
        attn_output = self.attention(x, mask)
        
        # 3. 前馈网络 + 残差连接 + 层归一化
        residual = attn_output
        x_ffn_in = self.norm2(attn_output) if self.pre_norm else attn_output
        ffn_output = self.ffn(x_ffn_in)
        output = residual + ffn_output
        if not self.pre_norm:
            output = self.norm2(output)
        
        return output


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """创建因果掩码（下三角矩阵）用于时间序列预测"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """创建填充掩码"""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask