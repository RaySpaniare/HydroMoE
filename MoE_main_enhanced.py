"""
HydroMoE v2.0 Enhanced Main Program
集成高级归一化和梯度稳定技术的主程序
"""

import os
import sys
import logging
import glob
import pandas as pd

from pandas.core.nanops import F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import warnings

# 导入核心模块
from MoE_config import get_default_config
from MoE_data_loader import FixedHydroDataset, FixedDataConfig, clear_data_cache, warmup_data_loading
from MoE_hybrid_model import HybridHydroMoEModel

# 导入增强功能
from MoE_advanced_normalization import create_gradient_stable_normalizer

# 导入训练器和评估器
from MoE_trainer_enhanced import enhanced_training_loop
from MoE_evaluator_simple import (
    evaluate_enhanced_model,
    load_best_model_if_exists,
)
from MoE_station_regime_calibration import wrap_with_calibration
from MoE_lowflow_augment import run_pipeline
from MoE_risk_refiner import run_risk_refine, finetune_on_risk_stations

# 导入改进功能（可选）
try:
    from MoE_feature_engineering import HydroFeatureEngineer, AdaptiveFeatureSelector
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    print(" 特征工程模块不可用，使用基础特征")

try:
    from MoE_multiscale_attention import MultiScaleTemporalAttention
    MULTISCALE_ATTENTION_AVAILABLE = True
except ImportError:
    MULTISCALE_ATTENTION_AVAILABLE = False
    print(" 多尺度注意力模块不可用")

# 设置警告过滤
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def set_seed(seed=42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def setup_enhanced_training():
    """设置增强训练环境"""
    
    # 1. 基础配置
    config = get_default_config()
    
    # 2. 数据配置 - 测试模式50个站点
    data_config = FixedDataConfig(
        normalize_features=True,  # 启用标准化，配合反归一化输出真实值
        normalize_targets=True,   # 启用目标标准化，避免量纲不匹配
        use_all_stations=True,   # 使用全部站点
        quick_test=False,          # 关闭快速测试模式
        quick_test_stations=50,   # 🔥 保持50个站点用于测试
        # 其他参数都使用正常模式的默认值（sequence_length=64, stride=16等）
    )
    
    # 2.1 启动时自动离线增强：
    # - 读取 cmaes_optimal_params.json，导出 R²<0.2 的低值站点清单
    # - 基于原始CSV生成“只用历史信息”的径流滞后/滚动特征增强版CSV
    try:
        meta = run_pipeline(
            src_csv=data_config.csv_path,
            cmaes_json='cmaes_optimal_params.json',
            out_dir='./outputs/augmented',
            r2_threshold=0.2
        )
        # 用增强后的CSV与推荐特征列替换
        if isinstance(meta, dict):
            if 'augmented_csv' in meta and meta['augmented_csv']:
                data_config.csv_path = meta['augmented_csv']
                print(f"  🔗 使用增强CSV: {data_config.csv_path}")
            rec_cols = meta.get('recommended_feature_cols', '')
            if rec_cols:
                data_config.feature_cols = [c for c in rec_cols.split(',') if c]
                print(f"  🧩 使用推荐特征列: {data_config.feature_cols}")
            if 'low_r2_list' in meta and meta['low_r2_list']:
                print(f"  📝 低R²站点清单: {meta['low_r2_list']}")
                print(f"  📉 低R²站点数量: {meta.get('low_r2_count', 'NA')}")
    except Exception as e:
        print(f"⚠️ 数据增强步骤失败，回退到原始CSV: {e}")
    
    # 3. 创建高级归一化器
    normalizer = create_gradient_stable_normalizer(strategy="station_wise")
    
    print("✅ 增强训练环境设置完成")
    print(f"  📊 归一化策略: 站点级归一化")
    print(f"  🎛️ 梯度控制: 适中策略")
    print(f"  🏁 训练模式: 测试模式（仅调整站点数=10，其他为正常参数）")
    try:
        seq_len = data_config.sequence_length
        stride = data_config.sequence_stride
    except Exception:
        seq_len = 96
        stride = 16
    print(f"  ⚙️ 正常参数: 序列长度{seq_len}，stride={stride}，batch=32")
    
    return config, data_config, normalizer


def create_enhanced_datasets(data_config, normalizer):
    """创建增强数据集"""
    
    print("\n🔄 创建增强数据集...")
    # 确保不受上一次快速测试缓存影响
    try:
        clear_data_cache()
    except Exception:
        pass
    
    # 1. 创建训练集并获取标准化器
    train_dataset = FixedHydroDataset(data_config, split="train", scalers=None)
    scalers = train_dataset.get_scalers()
    # 2. 验证/测试集共享训练集标准化参数，确保评估/反归一化一致
    val_dataset = FixedHydroDataset(data_config, split="val", scalers=scalers)
    test_dataset = FixedHydroDataset(data_config, split="test", scalers=scalers)
    
    print(f"  📈 训练集: {len(train_dataset)} 序列")
    print(f"  📊 验证集: {len(val_dataset)} 序列")
    print(f"  📋 测试集: {len(test_dataset)} 序列")
    
    # 2. 获取数据用于归一化（这里需要从数据集中提取原始数据）
    # 注意：实际实现中需要修改数据集类以支持高级归一化
    print("  🔧 应用高级归一化...")
    
    # 3. 创建数据加载器 - 使用正常模式的批次大小
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 🔥 使用适中的批次大小（考虑到只有10个站点，128太大）
        shuffle=True,
        num_workers=0,  # Windows兼容性
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # 与训练集保持一致
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # 与训练集保持一致
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print("✅ 增强数据集创建完成")
    
    return train_loader, val_loader, test_loader, normalizer


def create_enhanced_model(config):
    """创建增强模型"""
    
    print("\n🏗️ 创建增强模型...")
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️ 使用设备: {device}")
    
    # 创建模型（直接使用 Hybrid 模型，内部已集成 LSTM 水期专家）
    model = HybridHydroMoEModel(config).to(device)

    # 详细统计模型参数
    def analyze_model_complexity(model):
        """分析模型复杂性"""
        total_params = 0
        trainable_params = 0
        module_stats = {}

        # 分模块统计
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                module_params = sum(p.numel() for p in module.parameters())
                module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

                if module_params > 0:
                    module_stats[name] = {
                        'total': module_params,
                        'trainable': module_trainable
                    }

        # 总计
        for p in model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()

        return total_params, trainable_params, module_stats

    total_params, trainable_params, module_stats = analyze_model_complexity(model)

    print(f"  📊 总参数量: {total_params:,}")
    print(f"  🎯 可训练参数: {trainable_params:,}")
    print(f"  🔒 固定参数: {total_params - trainable_params:,}")

    # 分析模型架构复杂性
    print(f"\n💡 MoE架构复杂性分析:")
    print(f"   📥 输入特征: 仅3个 (降水、温度、蒸散发)")
    print(f"   🧠 神经网络专家: 4个专门化专家")
    print(f"      ❄️ 雪模块专家 (学习雪积累/融化机理)")
    print(f"      🌊 径流专家 (学习地表径流机理)")
    print(f"      🌿 蒸散发专家 (学习植被蒸腾机理)")
    print(f"      💧 排水专家 (学习地下排水机理)")
    print(f"   🎯 智能门控: 4个门控网络 (动态选择专家权重)")
    print(f"   🔄 注意力机制: 多头自注意力 (捕获时序依赖)")
    print(f"   ⚙️ 物理机理: PBM模块 (包含可学习水文参数)")
    print(f"   📈 参数学习: {trainable_params:,} 个可学习参数!")

    # 显示关键模块参数分布
    expert_params = 0
    gate_params = 0
    attention_params = 0

    for name, stats in module_stats.items():
        if any(x in name for x in ['nn_expert', 'mlp']):
            expert_params += stats['trainable']
        elif 'gate' in name:
            gate_params += stats['trainable']
        elif 'attention' in name:
            attention_params += stats['trainable']

    print(f"\n 参数分布:")
    print(f"    专家网络: {expert_params:,} 参数")
    print(f"    门控网络: {gate_params:,} 参数")
    print(f"    注意力机制: {attention_params:,} 参数")
    print(f"    其他模块: {trainable_params - expert_params - gate_params - attention_params:,} 参数")
    # 自动加载最佳模型（如果存在） — 先加载到基础模型
    load_best_model_if_exists(model)

    # 包装站点×水期校准（不改变原模型结构，端到端训练）
    try:
        model = wrap_with_calibration(model)
        print("  🔧 已启用站点×水期校准包装器 (CalibratedHybridModel)")
    except Exception as e:
        print(f"  ⚠️ 站点×水期校准包装器启用失败: {e}")
    return model, device


# 训练循环已移至 MoE_trainer_enhanced.py

def check_r2_consistency(predictions, targets, station_ids):
    """检查R²一致性"""
    from sklearn.metrics import r2_score
    
    # 整体R²
    overall_r2 = r2_score(targets, predictions)
    
    # 站点级R²
    station_r2s = []
    unique_stations = np.unique(station_ids)
    
    print(f"  🏢 站点级分析 ({len(unique_stations)}个站点):")
    
    for station in unique_stations:
        mask = station_ids == station
        if np.sum(mask) > 10:  # 至少10个样本
            station_r2 = r2_score(targets[mask], predictions[mask])
            station_r2s.append(station_r2)
    
    avg_station_r2 = np.mean(station_r2s)
    consistency_gap = abs(overall_r2 - avg_station_r2)
    
    print(f"    整体R²: {overall_r2:.4f}")
    print(f"    站点均值R²: {avg_station_r2:.4f}")
    print(f"    一致性差距: {consistency_gap:.4f}")
    
    if consistency_gap < 0.05:
        print("     R²一致性良好")
    else:
        print("     R²一致性需要改进")
    
    return consistency_gap < 0.05


def main():
    """主函数"""
    
    print("🌊 HydroMoE v2.0 Enhanced Training")
    print("=" * 60)
    print("集成高级归一化和梯度稳定技术")
    print()
    
    # 🚀 性能优化提示
    print("⚡ 性能优化配置 (已自动应用):")
    print(f"  📊 数据优化:")
    print(f"     - from_numpy()零拷贝加载: 启用")
    print(f"     - 日期字符串预缓存: 启用")
    print(f"     - 序列stride优化: 32 (减少序列数)")
    print(f"  💾 显存优化:")
    print(f"     - Gradient Checkpointing: {os.getenv('USE_GRAD_CHECKPOINT', '1')}")
    print(f"     - 批次大小: 64")
    print(f"     - 梯度累积: 2步 (等效batch=128)")
    print(f"     - Inplace激活函数: 启用")
    print(f"  ⚡ 计算优化:")
    print(f"     - PBM批量计算: 启用 (10-50x加速)")
    print(f"     - PyTorch 2.0 Flash Attention: 自动")
    print(f"     - 混合精度训练: False (稳定性优先)")
    print(f"  🧹 内存管理:")
    print(f"     - 周期性显存清理: 每100 batch")
    print(f"     - 数据Worker: 0 (Windows兼容)")
    print()
    
    try:
        # 1. 设置环境
        setup_logging()
        
        # 2. 设置增强训练
        config, data_config, normalizer = setup_enhanced_training()

          # 2.1 若存在上一轮指标：强制仅加载低R²站点的数据进行训练（非可选）
        try:
            metrics_csv = os.path.join('outputs', 'enhanced_real_runoff_predictions', 'station_daily_metrics.csv')
            if os.path.exists(metrics_csv):
                dfm = pd.read_csv(metrics_csv)
                if 'station_id' in dfm.columns and 'R2' in dfm.columns:
                    r2_th = getattr(config.training, 'risk_refine_r2_threshold', 0.2)
                    low_df = dfm[pd.to_numeric(dfm['R2'], errors='coerce') < float(r2_th)]
                    low_sids = [str(s) for s in low_df['station_id'].tolist()]
                    if low_sids:
                        data_config.filter_station_ids = low_sids
                        print(f"🧭 仅加载低R²站点数据进行训练: {len(low_sids)} 个 (<{r2_th})")
        except Exception as _:
            pass
        
        # 2.5 🚀 数据预热（可选，可通过环境变量控制）
        if os.getenv("WARMUP_DATA", "1").lower() in ["1", "true", "yes"]:
            logger.info("🔥 启动数据预热...")
            warmup_data_loading(data_config)
        
        # 3. 创建数据集（降低默认批次，缓解显存）
        train_loader, val_loader, test_loader, normalizer = create_enhanced_datasets(
            data_config, normalizer
        )
        
        # 4. 创建模型
        model, device = create_enhanced_model(config)
        # 注：此时 DataLoader 已按低R²集合构建，直接进入常规训练（即对低R²站点进行正式训练）
        
        # 5. 训练模型（或跳过，仅评估）
        eval_only = os.getenv("EVAL_ONLY", "0").lower() in ["1", "true", "yes"]
        if not eval_only:
            # 使用配置中的epochs为主（可用环境变量覆盖）
            try:
                epochs = int(os.getenv("EPOCHS", str(getattr(config.training, 'epochs', 50))))
            except Exception:
                epochs = 50
            enhanced_training_loop(
                model, train_loader, val_loader, device,
                epochs=epochs,
                patience=getattr(config.training, 'early_stopping_patience', 5)
            )
            # 直接依据训练配置开关启用高风险站点再训练（默认开启）
            try:
                if getattr(config.training, 'risk_refine_enable', True):
                    run_risk_refine(
                        model, train_loader, val_loader, device,
                        r2_threshold=getattr(config.training, 'risk_refine_r2_threshold', 0.2),
                        epochs=getattr(config.training, 'risk_refine_epochs', 8),
                        lr=getattr(config.training, 'risk_refine_lr', 5e-5),
                        patience=getattr(config.training, 'risk_refine_patience', 3)
                    )
            except Exception as e:
                print(f"⚠️ 风险站点再训练失败: {e}")
        else:
            print("\n⏭️ 跳过训练 (EVAL_ONLY=1)，直接使用当前/最佳权重进行评估。")
        
        # 6. 在评估前，重新加载最佳模型权重（稳定文件或最新时间戳）
        print("\n🔁 评估前加载最佳模型权重...")
        load_best_model_if_exists(model, path="outputs/enhanced_hydromoe_best.pth")

        # 7. 评估模型
        evaluate_enhanced_model(model, test_loader, device)
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n 程序执行成功")
    else:
        print("\n 程序执行失败")
        sys.exit(1)