"""
增强训练器模块 - 包含所有训练相关功能
1) 门控熵正则（防塌缩）
2) 基于站点R²的样本加权训练（可选）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd

from MoE_losses import StationR2Loss, create_loss_function
from MoE_metrics import compute_all_metrics
from MoE_advanced_normalization import apply_gradient_clipping

logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """增强训练器类"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        # 正则与加权配置（环境变量可覆盖）
        # 🚀 完全禁用熵正则，允许专家完全专业化
        self.gate_entropy_w = float(os.getenv('GATE_ENTROPY_W', '0.0'))  # 完全禁用
        self.regime_entropy_w = float(os.getenv('REGIME_ENTROPY_W', '0.0'))  # 完全禁用
        self.enable_station_weighting = os.getenv('STATION_WEIGHTING', '1').lower() in ['1','true','yes']
        self.station_weight_lambda = float(os.getenv('STATION_WEIGHT_LAMBDA', '0.5'))
        self.station_weights_csv = os.getenv('STATION_WEIGHTS_CSV', 'outputs/enhanced_real_runoff_predictions/station_performance_real_runoff.csv')
        self._station_weight_map = self._load_station_weights(self.station_weights_csv) if self.enable_station_weighting else {}

    def _load_station_weights(self, csv_path: str) -> Dict[str, float]:
        """从历史评估CSV加载站点权重: w = 1 + λ * clamp(0.5 - R2, 0, 1)"""
        weight_map: Dict[str, float] = {}
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'station_id' in df.columns and 'R2' in df.columns:
                    for _, row in df.iterrows():
                        sid = str(row['station_id'])
                        r2 = row['R2']
                        if pd.notna(r2):
                            delta = max(0.0, min(1.0, 0.5 - float(r2)))
                            w = 1.0 + self.station_weight_lambda * delta
                        else:
                            w = 1.0
                        weight_map[sid] = float(w)
                logging.info(f"📦 已加载站点权重: {len(weight_map)}")
        except Exception as e:
            logging.warning(f"⚠️ 加载站点权重失败: {e}")
        return weight_map

    def _compute_gate_entropy_loss(self, gate_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算门控熵正则: 模块(PBM/NN) + Regime(低/平/洪)"""
        module_entropy = torch.tensor(0.0, device=self.device)
        regime_entropy = torch.tensor(0.0, device=self.device)
        count_m = 0
        if gate_info is not None and isinstance(gate_info, dict):
            modules = gate_info.get('module_gates', {})
            for mname, minfo in modules.items():
                if isinstance(minfo, dict) and 'effective_gate' in minfo:
                    p = minfo['effective_gate']  # [B,2]
                    if isinstance(p, torch.Tensor):
                        ent = (p * (p + 1e-8).log()).sum(dim=-1).mean()  # sum p log p (<=0)
                        module_entropy = module_entropy + ent
                        count_m += 1
            if count_m > 0:
                module_entropy = module_entropy / count_m
            # Regime 权重
            rw = gate_info.get('regime', {}).get('weights', None)
            if rw is None:
                rw = gate_info.get('regime_weights', None)
            if isinstance(rw, torch.Tensor):
                regime_entropy = (rw * (rw + 1e-8).log()).sum(dim=-1).mean()
        return module_entropy, regime_entropy
    
    def train_step(self, batch, criterion, optimizer):
        """单步训练 - 优化版本"""
        self.model.train()
        
        # 前向传播
        outputs = self.model(batch, return_gate_info=True)
        predictions = outputs['runoff']
        
        # 🚀 优化：直接计算加权MSE，避免创建中间张量
        targets = batch['targets']
        
        # 站点加权（如可用）
        if self.enable_station_weighting and self._station_weight_map:
            if 'station_id' in batch:
                sids = batch['station_id']
                if isinstance(sids, (list, tuple)):
                    w_list = [self._station_weight_map.get(str(s), 1.0) for s in sids]
                    w = torch.tensor(w_list, device=predictions.device, dtype=predictions.dtype)
                else:
                    w = None
            else:
                w = None
        else:
            w = None
        
        # 🚀 优化：根据是否有权重，使用不同的损失计算
        if w is not None:
            base_vec = F.mse_loss(predictions, targets, reduction='none')
            loss_main = (base_vec * w).mean()
        else:
            loss_main = F.mse_loss(predictions, targets)

        # 门控熵正则（防塌缩，最大化熵 => 最小化 sum p log p）
        gate_info = outputs.get('gate_info', {}) if isinstance(outputs, dict) else {}
        mod_ent, reg_ent = self._compute_gate_entropy_loss(gate_info)
        loss_reg = self.gate_entropy_w * mod_ent + self.regime_entropy_w * reg_ent

        # 负载均衡损失（来自模型 gate_info），默认关闭，可通过环境变量开启
        # 彻底关闭负载均衡损失，避免均衡化倾向
        lb_w = float(os.getenv('LOAD_BALANCE_W', '0.0'))
        lb_loss = torch.tensor(0.0, device=self.device)
        try:
            if isinstance(gate_info, dict) and 'load_balancing_loss' in gate_info and lb_w > 0.0:
                raw_lb = gate_info['load_balancing_loss']
                if isinstance(raw_lb, torch.Tensor):
                    lb_loss = raw_lb
        except Exception:
            pass

        loss = loss_main + loss_reg + lb_w * lb_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = apply_gradient_clipping(self.model, max_norm=1.0)
        
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss_main': float(loss_main.detach().cpu().item()) if torch.is_tensor(loss_main) else float(loss_main),
            'loss_reg': float(loss_reg.detach().cpu().item()) if torch.is_tensor(loss_reg) else float(loss_reg),
            'loss_lb': float(lb_loss.detach().cpu().item()) if torch.is_tensor(lb_loss) else float(lb_loss),
            'grad_norm': grad_norm,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'accumulated': False
        }
    
    def validate(self, val_loader, criterion):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(batch)
                predictions = outputs['runoff']
                
                if hasattr(criterion, '__call__') and 'station_idx' in str(criterion.__class__):
                    loss = criterion(predictions, batch['targets'], batch.get('station_idx'))
                else:
                    loss = criterion(predictions, batch['targets'])
                
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(batch['targets'].cpu().numpy())
        
        # 计算指标
        metrics = compute_all_metrics(all_targets, all_preds)
        
        return total_loss / len(val_loader), metrics


def create_enhanced_trainer(model, strategy="conservative"):
    """创建增强训练器"""
    return EnhancedTrainer(model)


def validate_model(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(batch)
            predictions = outputs['runoff']
            
            if isinstance(criterion, StationR2Loss):
                loss = criterion(predictions, batch['targets'], batch.get('station_idx'))
            else:
                loss = criterion(predictions, batch['targets'])
            
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def quick_validation_metrics(model, val_loader, device, dataset):
    """快速验证指标计算（将列表安全拼接为NumPy数组后再计算）"""
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs = model(batch)
            predictions = outputs['runoff']  # Tensor [B] or [B,1]
            targets = batch['targets']       # Tensor [B] or [B,1]

            # 始终展平为一维并转为numpy
            preds_np = predictions.detach().cpu().numpy().reshape(-1)
            targets_np = targets.detach().cpu().numpy().reshape(-1)

            preds_list.append(preds_np)
            targets_list.append(targets_np)

    if len(preds_list) == 0:
        return {'R2': 0.0, 'KGE': 0.0, 'RMSE': float('inf')}

    # 安全拼接
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)

    return compute_all_metrics(y_true, y_pred)


def enhanced_training_loop(model, train_loader, val_loader, device, epochs=50, patience: int = 5):
    """增强训练循环 - 优化版本"""

    print("\n🚀 开始增强训练...")
    
    # 🚀 显存监控
    if device.type == 'cuda':
        print(f"💾 初始显存: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"💾 保留显存: {torch.cuda.memory_reserved()/1024**2:.1f} MB")

    # 1. 使用保守配置创建训练器
    enhanced_trainer = create_enhanced_trainer(model, strategy="conservative")

    # 2. 使用标准MSE损失函数
    criterion = nn.MSELoss()
    print("✅ 使用MSE损失函数（标准做法）")
    print("  📊 训练配置: 保守模式 - LR=1e-4, GradClip=1.0")

    # 3. 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # 4. 训练状态
    best_val_loss = float('inf')
    best_val_r2 = -1e9
    patience_counter = 0
    patience = int(patience)
    
    # 5. 训练循环
    for epoch in range(epochs):
        should_log = True
        # 训练阶段
        model.train()
        train_losses = []
        train_grad_norms = []
        if should_log:
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("-" * 50)

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            stats = enhanced_trainer.train_step(batch, criterion, optimizer)

            if not stats.get('accumulated', False):
                train_losses.append(stats['loss'])
                train_grad_norms.append(stats['grad_norm'])

        # 🚀 优化：训练阶段结束后再统一清理显存，减少额外同步开销
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 验证阶段
        if True:
            val_loss = validate_model(model, val_loader, criterion, device)
            # 详细指标
            val_metrics = quick_validation_metrics(model, val_loader, device, val_loader.dataset)
            val_r2 = float(val_metrics.get('R2', 0.0))

            # 可选：统计Regime门控
            if should_log:
                try:
                    with torch.no_grad():
                        reg_weights = []
                        for vbatch in val_loader:
                            vbatch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in vbatch.items()}
                            voutputs = model(vbatch)
                            if 'regime_weights' in voutputs:
                                reg_weights.append(voutputs['regime_weights'].cpu().numpy())
                        
                        if reg_weights:
                            reg_weights = np.concatenate(reg_weights, axis=0)
                            regime_means = np.mean(reg_weights, axis=0)
                            regime_labels = ['低', '平', '洪']
                            regime_str = ', '.join([f"{label}={mean:.3f}" for label, mean in zip(regime_labels, regime_means)])
                            print(f"  🧭 Regime门控均值: {regime_str}")
                except Exception as e:
                    pass  # 忽略门控统计错误

            # 输出训练信息
            if should_log:
                print(f"  📊 训练损失: {np.mean(train_losses):.4f}")
                try:
                    print(f"    ├─ 主损失: {stats.get('loss_main', float('nan')):.4f}")
                    print(f"    ├─ 正则(门控+Regime): {stats.get('loss_reg', float('nan')):.4f}")
                    print(f"    └─ 负载均衡: {stats.get('loss_lb', float('nan')):.6f}")
                except Exception:
                    pass
                print(f"  📈 验证损失: {val_loss:.4f}")
                print(f"  🎯 平均梯度范数: {np.mean(train_grad_norms):.4f}")
                print(f"  📋 验证指标 (mm/day):")
                print(f"    🎯 R²: {val_r2:.4f} (主要目标)")
                print(f"    🎯 KGE: {val_metrics.get('KGE', 0.0):.4f} (主要目标)")
                print(f"    📊 RMSE: {val_metrics.get('RMSE', 0.0):.4f}")
                print(f"    📊 Bias: {val_metrics.get('bias', 0.0):.4f}")

            # 早停逻辑
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型（确保目录存在）
                os.makedirs("outputs", exist_ok=True)
                model_path = "outputs/enhanced_hydromoe_best.pth"
                torch.save(model.state_dict(), model_path)
                print(f"  💾 模型已保存: {model_path}")
                if should_log:
                    print(f"  ✅ 新的最佳模型 (R²={val_r2:.4f})")
            else:
                patience_counter += 1
                if should_log:
                    print(f"  ⏳ 未改进计数: {patience_counter}/{patience}")

            # 特殊处理：如果R²明显回落，给予更多耐心
            if should_log and epoch > 25 and val_r2 < best_val_r2 - 0.05 and patience_counter < patience:
                print(f"  ⚠️ 验证R²较明显回落 (当前={val_r2:.4f} vs 最佳={best_val_r2:.4f})，继续观察，不早停")

            if patience_counter >= patience:
                print(f"  🛑 早停触发 (patience={patience})")
                break
        
        # 更新学习率
        scheduler.step()
    
    print("\n✅ 增强训练完成")
    return enhanced_trainer


def evaluate_enhanced_model(model, test_loader, device):
    """评估增强模型"""

    print("\n📊 评估增强模型...")

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs = model(batch)
            predictions = outputs['runoff']

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch['targets'].cpu().numpy())

    if len(all_predictions) > 0:
        metrics = compute_all_metrics(all_targets, all_predictions)

        print("🎯 最终测试结果:")
        print(f"  R²: {metrics.get('R2', 0):.4f}")
        print(f"  KGE: {metrics.get('KGE', 0):.4f}")
        print(f"  RMSE: {metrics.get('RMSE', 0):.4f}")
        print(f"  Bias: {metrics.get('bias', 0):.4f}")

        return metrics
    else:
        print("⚠️ 没有可用的测试数据")
        return {'R2': 0.0, 'KGE': 0.0, 'RMSE': float('inf')}
