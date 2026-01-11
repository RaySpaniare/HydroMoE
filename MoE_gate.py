"""
MoE门控路由器 - 专家网络选择和负载均衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class MoEGate(nn.Module):
    """MoE门控网络 - 路由输入到不同的专家网络"""
    
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 top_k: int = 2,
                 capacity_factor: float = 1.25,
                 dropout: float = 0.1,
                 noisy_gating: bool = True,
                 noise_epsilon: float = 1e-2):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # 确保top_k不超过专家数量
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        
        # 噪声网络（用于探索）
        if noisy_gating:
            self.noise_gate = nn.Linear(input_dim, num_experts, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 负载均衡参数
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
        self._init_weights()
    
    def _init_weights(self):
        """保守的权重初始化"""
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        if self.noisy_gating:
            nn.init.zeros_(self.noise_gate.weight)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 计算专家选择概率和路由信息
        
        Args:
            x: [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
            training: 是否在训练模式
            
        Returns:
            Dict包含:
                - gate_weights: [batch_size, num_experts] 门控权重
                - top_k_indices: [batch_size, top_k] 选中的专家索引
                - top_k_weights: [batch_size, top_k] 对应的权重
                - load_balancing_loss: 负载均衡损失
                - capacity_info: 容量信息
        """
        # 处理输入维度
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, seq_len, input_dim = x.shape
            x = x.view(-1, input_dim)  # [batch_size * seq_len, input_dim]
        else:
            batch_size, seq_len = x.shape[0], 1
        
        # 1. 计算门控分数
        gate_logits = self.gate(x)  # [tokens, num_experts]
        
        # 2. 添加噪声（如果启用）
        if self.noisy_gating and training:
            noise_logits = self.noise_gate(x)
            noise = torch.randn_like(noise_logits) * F.softplus(noise_logits) * self.noise_epsilon
            gate_logits += noise
        
        # 3. 计算门控权重
        gate_weights = F.softmax(gate_logits, dim=-1)  # [tokens, num_experts]
        
        # 4. 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        
        # 5. 重新归一化top-k权重
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 6. 计算负载均衡损失
        load_balancing_loss = self._compute_load_balancing_loss(gate_weights)
        
        # 7. 更新专家使用统计
        if training:
            self._update_expert_usage(gate_weights)
        
        # 8. 计算容量信息
        capacity_info = self._compute_capacity_info(top_k_indices, gate_weights.shape[0])
        
        # 恢复形状信息
        if len(original_shape) == 3:
            gate_weights = gate_weights.view(batch_size, seq_len, self.num_experts)
            top_k_indices = top_k_indices.view(batch_size, seq_len, self.top_k)
            top_k_weights = top_k_weights.view(batch_size, seq_len, self.top_k)
        
        return {
            'gate_weights': gate_weights,
            'top_k_indices': top_k_indices,
            'top_k_weights': top_k_weights,
            'load_balancing_loss': load_balancing_loss,
            'capacity_info': capacity_info,
            'expert_usage': self.expert_usage.clone()
        }
    
    def _compute_load_balancing_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失 - 优化版本"""
        # 🚀 优化：使用更简单的计算，减少操作
        expert_usage = gate_weights.mean(dim=0)  # [num_experts]
        
        # 🚀 优化：直接计算与理想分布的偏差
        ideal_usage = 1.0 / self.num_experts
        load_balancing_loss = ((expert_usage - ideal_usage) ** 2).mean()
        
        return load_balancing_loss
    
    def _update_expert_usage(self, gate_weights: torch.Tensor):
        """更新专家使用统计"""
        with torch.no_grad():
            current_usage = gate_weights.sum(dim=0)  # [num_experts]
            self.expert_usage = 0.99 * self.expert_usage + 0.01 * current_usage
            self.total_tokens += gate_weights.shape[0]
    
    def _compute_capacity_info(self, top_k_indices: torch.Tensor, num_tokens: int) -> Dict[str, float]:
        """计算容量信息用于监控"""
        with torch.no_grad():
            # 计算每个专家被选择的次数
            expert_counts = torch.zeros(self.num_experts, device=top_k_indices.device)
            for i in range(self.num_experts):
                expert_counts[i] = (top_k_indices == i).sum().float()
            
            # 计算容量利用率
            capacity_per_expert = num_tokens * self.capacity_factor / self.num_experts
            capacity_utilization = expert_counts / capacity_per_expert
            
            return {
                'capacity_utilization_mean': capacity_utilization.mean().item(),
                'capacity_utilization_std': capacity_utilization.std().item(),
                'expert_usage_entropy': self._compute_entropy(expert_counts / expert_counts.sum()),
                'overflow_rate': (capacity_utilization > 1.0).float().mean().item()
            }
    
    def _compute_entropy(self, probs: torch.Tensor) -> float:
        """计算熵"""
        probs = probs + 1e-8  # 避免log(0)
        entropy = -(probs * torch.log(probs)).sum()
        return entropy.item()


class ExpertDispatcher(nn.Module):
    """专家分发器 - 将输入分发给选中的专家"""
    
    def __init__(self, num_experts: int, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
    
    def forward(self, 
                inputs: torch.Tensor,
                gate_info: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        分发输入到专家
        
        Args:
            inputs: [batch_size, seq_len, input_dim]
            gate_info: 门控信息
            
        Returns:
            分发结果字典
        """
        batch_size, seq_len, input_dim = inputs.shape
        top_k_indices = gate_info['top_k_indices']  # [batch_size, seq_len, top_k]
        top_k_weights = gate_info['top_k_weights']  # [batch_size, seq_len, top_k]
        
        # 展平输入用于分发
        flat_inputs = inputs.view(-1, input_dim)  # [batch_size * seq_len, input_dim]
        flat_indices = top_k_indices.view(-1, top_k_indices.shape[-1])  # [batch_size * seq_len, top_k]
        flat_weights = top_k_weights.view(-1, top_k_weights.shape[-1])  # [batch_size * seq_len, top_k]
        
        # 为每个专家准备输入
        expert_inputs = {}
        expert_weights = {}
        
        for expert_id in range(self.num_experts):
            # 找到分配给这个专家的所有token
            mask = (flat_indices == expert_id)  # [batch_size * seq_len, top_k]
            
            if mask.any():
                # 获取token索引和对应的权重
                token_indices, k_indices = torch.where(mask)
                selected_inputs = flat_inputs[token_indices]  # [num_selected, input_dim]
                selected_weights = flat_weights[token_indices, k_indices]  # [num_selected]
                
                expert_inputs[expert_id] = {
                    'inputs': selected_inputs,
                    'token_indices': token_indices,
                    'weights': selected_weights,
                    'original_shape': (batch_size, seq_len, input_dim)
                }
        
        return expert_inputs


class ExpertCombiner(nn.Module):
    """专家输出组合器"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self,
                expert_outputs: Dict[int, torch.Tensor],
                expert_inputs: Dict[int, Dict],
                original_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        组合专家输出
        
        Args:
            expert_outputs: {expert_id: output_tensor}
            expert_inputs: 专家输入信息（包含权重和索引）
            original_shape: 原始输入形状
            
        Returns:
            combined_output: [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = original_shape
        output_dim = None
        
        # 初始化输出张量
        for expert_id, output in expert_outputs.items():
            if output_dim is None:
                output_dim = output.shape[-1]
                break
        
        if output_dim is None:
            raise ValueError("No expert outputs provided")
        
        # 创建输出张量
        flat_outputs = torch.zeros(batch_size * seq_len, output_dim, 
                                 device=next(iter(expert_outputs.values())).device,
                                 dtype=next(iter(expert_outputs.values())).dtype)
        
        # 组合专家输出
        for expert_id, output in expert_outputs.items():
            if expert_id in expert_inputs:
                info = expert_inputs[expert_id]
                token_indices = info['token_indices']
                weights = info['weights'].unsqueeze(-1)  # [num_tokens, 1]
                
                # 加权累加
                weighted_output = output * weights
                flat_outputs.index_add_(0, token_indices, weighted_output)
        
        # 恢复原始形状
        combined_output = flat_outputs.view(batch_size, seq_len, output_dim)

        return combined_output


class ContextAwareMoEGate(nn.Module):
    """上下文感知MoE门控 - 考虑历史信息和流量状态"""

    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 top_k: int = 2,
                 context_window: int = 7,  # 上下文窗口大小
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.context_window = context_window

        # 上下文编码器
        self.context_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim // 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout if context_window > 1 else 0
        )

        # 流量状态检测器
        self.flow_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 4),  # 4个流量级别
            nn.Softmax(dim=-1)
        )

        # 主门控网络
        self.main_gate = nn.Sequential(
            nn.Linear(input_dim + input_dim // 2 + 4, input_dim),  # 输入+上下文+流量状态
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_experts)
        )

        # 专家特化权重
        self.expert_specialization = nn.Parameter(torch.randn(num_experts, 4))  # 专家对流量级别的偏好

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 专家特化权重初始化
        nn.init.normal_(self.expert_specialization, mean=0, std=0.1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: [batch_size, input_dim] 当前输入
            context: [batch_size, context_window, input_dim] 历史上下文（可选）

        Returns:
            门控结果字典
        """
        batch_size = x.shape[0]

        # 1. 上下文编码
        if context is not None and context.shape[1] > 0:
            context_encoded, _ = self.context_encoder(context)
            context_feature = context_encoded[:, -1, :]  # 取最后一个时间步
        else:
            context_feature = torch.zeros(batch_size, self.input_dim // 2, device=x.device)

        # 2. 流量状态检测
        flow_state = self.flow_detector(x)  # [batch_size, 4]

        # 3. 特征融合
        combined_features = torch.cat([x, context_feature, flow_state], dim=-1)

        # 4. 主门控计算
        gate_logits = self.main_gate(combined_features)  # [batch_size, num_experts]

        # 5. 专家特化调制
        specialization_scores = torch.matmul(flow_state, self.expert_specialization.T)  # [batch_size, num_experts]
        adjusted_logits = gate_logits + 0.1 * specialization_scores

        # 6. Top-k选择
        gate_probs = F.softmax(adjusted_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)

        # 7. 重新归一化
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 8. 负载均衡损失
        expert_usage = gate_probs.mean(dim=0)
        uniform_distribution = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = F.kl_div(
            expert_usage.log(), uniform_distribution, reduction='batchmean'
        )

        return {
            'expert_weights': top_k_probs,
            'expert_indices': top_k_indices,
            'load_balancing_loss': load_balancing_loss,
            'flow_state': flow_state,
            'expert_usage': expert_usage,
            'gate_logits': adjusted_logits
        }