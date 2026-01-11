"""
HydroMoE v2.0 配置管理
简化且类型安全的配置系统
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径 - 使用合并后的长表文件
    data_root: str = r'D:\Science Research\中科院地理所\PBM+ML\数据\美国已处理'
    csv_file: str = '特征合并长表.csv'  # 主要数据文件
    
    # 特征列定义
    feature_cols: List[str] = None  # 输入特征列：降水、温度、蒸散发
    target_col: str = "runoff"  # 目标列：径流
    
    # 序列配置 - 优化GPU利用率
    sequence_length: int = 64  # 🚀 优化：64平衡速度与显存
    stride: int = 32  # 🚀 优化：增大stride减少序列数，提升速度
    
    # 时间划分
    train_start: str = '1980-01-01'
    train_end: str = '1999-12-31'
    val_start: str = '2000-01-01'
    val_end: str = '2007-12-31'
    test_start: str = '2008-01-01'
    test_end: str = '2014-09-30'
    
    # 全站点训练配置
    use_all_stations: bool = True  # 启用全部站点训练
    quick_test: bool = False
    quick_test_stations: int = 10
    
    def __post_init__(self):
        """初始化后处理"""
        if self.feature_cols is None:
            # 根据您的说明：输入是蒸散发、降水、温度，输出是径流
            self.feature_cols = ["pet", "precip", "temp"]  # 蒸散发、降水、温度作为输入特征


@dataclass
class ModelConfig:
    """模型架构配置"""
    input_size: int = 10  # 输入特征数量
    hidden_size: int = 128  # 隐藏层大小
    num_layers: int = 2  # 网络层数
    dropout: float = 0.1  # Dropout概率
    
    # 序列相关
    sequence_length: int = 96  # 序列长度
    max_sequence_length: int = 512  # 最大序列长度（用于位置编码）
    
    # MoE架构参数
    d_model: int = 256  # 模型维度
    num_heads: int = 8  # 注意力头数
    num_attention_layers: int = 2  # 注意力层数
    num_experts: int = 4  # 专家网络数量
    top_k: int = 2  # 每次选择的专家数量
    capacity_factor: float = 1.25  # 容量因子
    noisy_gating: bool = True  # 是否使用噪声门控
    noise_epsilon: float = 1e-2  # 噪声幅度
    
    # 专家配置
    expert_configs: List[Dict] = None  # 专家网络配置列表
    
    # 输出配置
    sequence_aggregation: str = 'last'  # 序列聚合方式: 'last', 'mean', 'attention'
    use_final_layer: bool = False  # 是否使用最终处理层
    
    # 混合模型模块门控配置（PBM vs NN）
    module_gate_top_k: int = 2  # 1=只选一个专家(硬选择); 2=两者加权
    module_gate_temperature: float = 0.3  # 🚀 降低温度，增强选择性（从0.7降到0.3）
    pbm_min_weight: float = 0.0  # 🚀 移除最小权重约束，允许完全选择
    # 径流Regime头门控
    regime_top_k: int = 2 # 1=硬选择一个Regime专家; >1=软加权
    regime_temperature: float = 1.8
    
    def __post_init__(self):
        """初始化后处理"""
        if self.expert_configs is None:
            # 改进的专家配置：更专业化的专家组合
            self.expert_configs = [
                # 流量分级专家
                {'type': 'flow_regime', 'regime_type': 'low', 'hidden_dim': 128, 'dropout': 0.1},
                {'type': 'flow_regime', 'regime_type': 'high', 'hidden_dim': 128, 'dropout': 0.15},
                # 季节性专家
                {'type': 'seasonal', 'season_type': 'summer', 'hidden_dim': 128, 'dropout': 0.1},
                {'type': 'seasonal', 'season_type': 'winter', 'hidden_dim': 128, 'dropout': 0.05},
                # 传统水文专家
                {'type': 'hydrology', 'hydrology_type': 'runoff', 'hidden_dim': 128, 'dropout': 0.1},
                {'type': 'attention', 'hidden_dim': 128, 'num_heads': 4, 'dropout': 0.1}
            ]
    
    # 物理约束
    use_pbm: bool = True
    pbm_weight: float = 0.3
    

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数 - GPU优化
    epochs: int = 50
    batch_size: int = 64  # 🚀 优化：降回64避免显存溢出，配合gradient_checkpointing
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 优化器
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    
    # 梯度控制
    gradient_clip: float = 1.0
    accumulation_steps: int = 2  # 🚀 优化：使用梯度累积模拟更大batch
    
    # 验证和保存
    eval_every: int = 5  # 更频繁的验证
    save_every: int = 20
    early_stopping_patience: int = 5
    
    # 数值稳定性和GPU优化
    use_amp: bool = False  # 🚀 优化：暂时关闭AMP，提升稳定性
    check_grad_norm: bool = True
    max_grad_norm: float = 10.0

    # 低R²站点风险细化（默认启用，无需环境变量）
    risk_refine_enable: bool = True
    risk_refine_r2_threshold: float = 0.2
    risk_refine_epochs: int = 8
    risk_refine_lr: float = 5e-5
    risk_refine_patience: int = 3


@dataclass
class EvalConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: ['mse', 'rmse', 'mae', 'r2', 'nse', 'kge'])
    save_predictions: bool = True
    plot_results: bool = True


@dataclass
class SystemConfig:
    """系统配置"""
    # 设备 - GPU优化
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    num_workers: int = 4  # 🚀 优化：降低worker数量，减少内存开销
    pin_memory: bool = True
    prefetch_factor: int = 2  # 🚀 优化：降低预取因子，减少内存占用
    persistent_workers: bool = False  # 🚀 优化：关闭持久worker，减少内存
    
    # 输出
    output_dir: str = './outputs'
    experiment_name: Optional[str] = None
    log_level: str = 'INFO'
    
    # 可复现性
    seed: int = 42
    deterministic: bool = False  # 🚀 优化：关闭确定性模式提升速度
    
    # GPU内存优化
    empty_cache_every: int = 100  # 🚀 优化：降低清理频率，减少开销
    monitor_gpu: bool = False  # 🚀 优化：关闭监控减少开销


@dataclass
class HydroMoEConfig:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """配置验证和调整"""
        # Windows系统优化
        import platform
        if platform.system() == 'Windows':
            self.system.num_workers = 0  # Windows兼容性
            self.system.pin_memory = False
        
        # 快速测试模式调整
        if self.data.quick_test:
            self.training.epochs = min(self.training.epochs, 10)
            self.training.batch_size = min(self.training.batch_size, 16)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HydroMoEConfig':
        """从字典创建配置"""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        eval_config = EvalConfig(**config_dict.get('eval', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            eval=eval_config,
            system=system_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)


def get_default_config() -> HydroMoEConfig:
    """获取默认配置"""
    return HydroMoEConfig()


def get_debug_config() -> HydroMoEConfig:
    """获取调试配置"""
    config = HydroMoEConfig()
    
    # 调试模式调整
    config.data.quick_test = True
    config.data.quick_test_stations = 3
    config.data.sequence_length = 32
    config.data.stride = 16
    
    config.model.hidden_dim = 32
    config.model.expert_dim = 16
    
    config.training.epochs = 5
    config.training.batch_size = 4
    config.training.learning_rate = 1e-5  # 保守学习率
    config.training.eval_every = 2
    config.training.use_amp = False
    
    return config


# CMA-ES相关配置
CMAES_CONFIG = {
    'params_file': 'cmaes_optimal_params.json',
    'cache_size': 1000,
    'default_params_available': True,
    'param_mapping': {
        'runoff_params': {
            'c_max': 'wcap',          # 容量参数
            'beta_e': 'beta',         # 蒸发系数
            'wmin_ratio': 'wmin',     # 最小含水量比例  
            'wmax_ratio': 'wmax',     # 最大含水量比例
            'b': 'wava',              # 土壤参数
            'k': 'beta',              # 渗透系数
            'alpha': 'fveg'           # 植被覆盖度
        },
        'et_params': {
            'transp_fraction': 'transp_fraction',  # 蒸腾比例
            'et_alpha': 'et_alpha',               # ET系数
            'rm_crit': 'fbare',                   # 临界土壤湿度
            'et_beta': 'lai_annual'               # ET beta参数
        },
        'snow_params': {
            'melt_factor': 'wava',      # 融雪因子
            'melt_temp': 'wmin'         # 融雪临界温度
        },
        'groundwater_params': {
            'baseflow_threshold': 'baseflow_threshold',  # 基流阈值（如果存在）
            'k_drainage': 'beta',                        # 排水系数
            'drainage_exp': 'wmax',                      # 排水指数
            'baseflow_factor': 'pet_correction',         # 基流因子
            'groundwater_decay': 'fveg'                  # 地下水衰减
        }
    }
}

# PBM配置 - 直接使用CMA-ES优化参数
PBM_CONFIG = {
    'params_file': 'cmaes_optimal_params.json',  # CMA-ES参数文件
    'use_precomputed_results': False,  # 不使用预计算结果，直接计算
    'cache_size': 1000,
    'time_col': 'time_step',
    'station_id_col': 'station_id',
    'modules': ['snow', 'runoff', 'et', 'drainage']  # 四个水文模块
}

# 保持向后兼容
PBM_RESULTS_CONFIG = PBM_CONFIG