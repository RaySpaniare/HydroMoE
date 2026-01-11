"""
水文模型评估指标计算
包含R², KGE, RMSE, MSE, bias等完整指标
"""

import torch
import numpy as np
from typing import Dict, Tuple, Union, List, Any


def _to_numpy_1d(x: Union[torch.Tensor, np.ndarray, List[Any]]) -> np.ndarray:
    """将输入统一转换为 1D numpy 数组。
    支持 torch.Tensor / np.ndarray / list-like。
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    try:
        x = x.astype(np.float64, copy=False)
    except Exception:
        x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1)


def compute_r2(y_true: Union[torch.Tensor, np.ndarray, List[Any]], 
               y_pred: Union[torch.Tensor, np.ndarray, List[Any]]) -> float:
    """
    计算决定系数 R²
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        R²值
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    # 计算R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)


def compute_kge(y_true: Union[torch.Tensor, np.ndarray, List[Any]], 
                y_pred: Union[torch.Tensor, np.ndarray, List[Any]]) -> Tuple[float, Dict[str, float]]:
    """
    计算Kling-Gupta效率系数 (KGE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        KGE值和分解组件
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan, {}
    
    # 计算KGE组件（数值稳定）
    eps = 1e-12
    n_valid = len(y_true)
    # 1) 相关系数 (r) - 避免零方差/长度不足
    std_true = float(np.std(y_true)) if n_valid > 0 else 0.0
    std_pred = float(np.std(y_pred)) if n_valid > 0 else 0.0
    if n_valid < 2 or std_true < eps or std_pred < eps:
        correlation = 0.0
    else:
        try:
            correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
        except Exception:
            correlation = 0.0
    if np.isnan(correlation):
        correlation = 0.0
    
    # 2. 偏差比率 (β) - 均值比率
    mean_true = float(np.mean(y_true)) if n_valid > 0 else 0.0
    mean_pred = float(np.mean(y_pred)) if n_valid > 0 else 0.0
    bias_ratio = (mean_pred + eps) / (mean_true + eps)
    
    # 3. 变异系数比率 (γ) - 标准差比率
    # 3) 变异系数比 (γ) - 稳定计算
    cv_true = (std_true) / (mean_true + eps)
    cv_pred = (std_pred) / (mean_pred + eps)
    variability_ratio = cv_pred / (cv_true + eps)
    
    # 计算KGE
    kge = 1 - float(np.sqrt((correlation - 1)**2 + (bias_ratio - 1)**2 + (variability_ratio - 1)**2))
    
    components = {
        'correlation': float(correlation),
        'bias_ratio': float(bias_ratio),
        'variability_ratio': float(variability_ratio)
    }
    
    return float(kge), components


def compute_rmse(y_true: Union[torch.Tensor, np.ndarray, List[Any]], 
                 y_pred: Union[torch.Tensor, np.ndarray, List[Any]]) -> float:
    """
    计算均方根误差 (RMSE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        RMSE值
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)


def compute_mse(y_true: Union[torch.Tensor, np.ndarray, List[Any]], 
                y_pred: Union[torch.Tensor, np.ndarray, List[Any]]) -> float:
    """
    计算均方误差 (MSE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MSE值
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse)


def compute_bias(y_true: Union[torch.Tensor, np.ndarray, List[Any]], 
                 y_pred: Union[torch.Tensor, np.ndarray, List[Any]]) -> float:
    """
    计算偏差 (bias)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        bias值 (预测均值 - 观测均值)
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    bias = np.mean(y_pred) - np.mean(y_true)
    return float(bias)


def compute_mae(y_true: Union[torch.Tensor, np.ndarray, List[Any]], 
                y_pred: Union[torch.Tensor, np.ndarray, List[Any]]) -> float:
    """
    计算平均绝对误差 (MAE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MAE值
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae)


def compute_nse(y_true: Union[torch.Tensor, np.ndarray, List[Any]], 
                y_pred: Union[torch.Tensor, np.ndarray, List[Any]]) -> float:
    """
    计算Nash-Sutcliffe效率系数 (NSE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        NSE值
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    # 计算NSE
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    nse = 1 - (numerator / denominator)
    return float(nse)


def compute_all_metrics(y_true: Union[torch.Tensor, np.ndarray], 
                       y_pred: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    计算所有评估指标 - 优化版本
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含所有指标的字典
    """
    # 🚀 优化：统一预处理，避免重复转换
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    
    # 移除NaN值（一次性处理）
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'R2': np.nan, 'KGE': np.nan, 'RMSE': np.nan, 'MSE': np.nan, 
                'bias': np.nan, 'MAE': np.nan, 'NSE': np.nan}
    
    # 🚀 优化：批量计算所有指标，共享中间计算
    errors = y_true - y_pred
    sq_errors = errors ** 2
    abs_errors = np.abs(errors)
    
    # 基础统计
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    
    # MSE, RMSE, MAE
    mse = np.mean(sq_errors)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)
    bias = mean_pred - mean_true
    
    # R² 和 NSE（公式相同）
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((y_true - mean_true) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    nse = r2  # NSE和R²计算相同
    
    # KGE
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 and std_true > 0 and std_pred > 0 else 0.0
    bias_ratio = (mean_pred + 1e-12) / (mean_true + 1e-12)
    cv_true = std_true / (mean_true + 1e-12)
    cv_pred = std_pred / (mean_pred + 1e-12)
    variability_ratio = cv_pred / (cv_true + 1e-12)
    kge = 1 - float(np.sqrt((correlation - 1)**2 + (bias_ratio - 1)**2 + (variability_ratio - 1)**2))
    
    # 汇总所有指标
    metrics = {
        'R2': float(r2),
        'KGE': float(kge),
        'RMSE': float(rmse),
        'MSE': float(mse),
        'bias': float(bias),
        'MAE': float(mae),
        'NSE': float(nse),
        'KGE_correlation': float(correlation),
        'KGE_bias_ratio': float(bias_ratio),
        'KGE_variability_ratio': float(variability_ratio)
    }
    
    return metrics


def format_metrics_string(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    格式化指标为字符串
    
    Args:
        metrics: 指标字典
        precision: 小数位精度
        
    Returns:
        格式化的字符串
    """
    formatted_parts = []
    
    # 主要指标
    main_metrics = ['R2', 'KGE', 'RMSE', 'MSE', 'bias']
    for metric in main_metrics:
        if metric in metrics:
            value = metrics[metric]
            if np.isnan(value):
                formatted_parts.append(f"{metric}: NaN")
            else:
                formatted_parts.append(f"{metric}: {value:.{precision}f}")
    
    return ", ".join(formatted_parts)


if __name__ == "__main__":
    # 测试评估指标
    print("🧪 测试评估指标计算...")
    
    # 生成测试数据
    np.random.seed(42)
    y_true = np.random.randn(1000) * 2 + 5
    y_pred = y_true + np.random.randn(1000) * 0.5  # 添加一些噪声
    
    # 计算所有指标
    metrics = compute_all_metrics(y_true, y_pred)
    
    print("📊 评估指标结果:")
    for metric, value in metrics.items():
        if np.isnan(value):
            print(f"  {metric}: NaN")
        else:
            print(f"  {metric}: {value:.6f}")
    
    print(f"\n📝 格式化输出: {format_metrics_string(metrics)}")
    print(" 评估指标测试完成！")


def compute_stratified_metrics(y_true: Union[torch.Tensor, np.ndarray],
                              y_pred: Union[torch.Tensor, np.ndarray],
                              quantiles: List[float] = [0.33, 0.67]) -> Dict[str, Dict[str, float]]:
    """
    计算分层评估指标（低、中、高径流分别评估）

    Args:
        y_true: 真实值
        y_pred: 预测值
        quantiles: 分层的分位数阈值

    Returns:
        分层指标字典
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    # 计算分位数阈值
    thresholds = np.quantile(y_true, quantiles)

    # 分层
    low_mask = y_true <= thresholds[0]
    high_mask = y_true > thresholds[-1]
    mid_mask = ~(low_mask | high_mask)

    stratified_metrics = {}

    # 低径流
    if np.sum(low_mask) > 0:
        stratified_metrics['low_flow'] = compute_all_metrics(y_true[low_mask], y_pred[low_mask])
        stratified_metrics['low_flow']['sample_count'] = int(np.sum(low_mask))
        stratified_metrics['low_flow']['flow_range'] = f"{y_true[low_mask].min():.3f}-{y_true[low_mask].max():.3f}"

    # 中径流
    if np.sum(mid_mask) > 0:
        stratified_metrics['mid_flow'] = compute_all_metrics(y_true[mid_mask], y_pred[mid_mask])
        stratified_metrics['mid_flow']['sample_count'] = int(np.sum(mid_mask))
        stratified_metrics['mid_flow']['flow_range'] = f"{y_true[mid_mask].min():.3f}-{y_true[mid_mask].max():.3f}"

    # 高径流
    if np.sum(high_mask) > 0:
        stratified_metrics['high_flow'] = compute_all_metrics(y_true[high_mask], y_pred[high_mask])
        stratified_metrics['high_flow']['sample_count'] = int(np.sum(high_mask))
        stratified_metrics['high_flow']['flow_range'] = f"{y_true[high_mask].min():.3f}-{y_true[high_mask].max():.3f}"

    return stratified_metrics


def compute_peak_flow_metrics(y_true: Union[torch.Tensor, np.ndarray],
                             y_pred: Union[torch.Tensor, np.ndarray],
                             peak_threshold: float = 0.9) -> Dict[str, float]:
    """
    计算峰值径流预测准确性指标

    Args:
        y_true: 真实值
        y_pred: 预测值
        peak_threshold: 峰值阈值（分位数）

    Returns:
        峰值预测指标
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    # 确定峰值阈值
    threshold_value = np.quantile(y_true, peak_threshold)
    peak_mask = y_true >= threshold_value

    if np.sum(peak_mask) == 0:
        return {'peak_count': 0}

    # 峰值预测指标
    peak_true = y_true[peak_mask]
    peak_pred = y_pred[peak_mask]

    # 峰值检测准确性
    pred_peaks = y_pred >= threshold_value
    true_positives = np.sum(peak_mask & pred_peaks)
    false_positives = np.sum(~peak_mask & pred_peaks)
    false_negatives = np.sum(peak_mask & ~pred_peaks)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 峰值量级预测准确性
    peak_r2 = compute_r2(peak_true, peak_pred)
    peak_rmse = compute_rmse(peak_true, peak_pred)
    peak_bias = compute_bias(peak_true, peak_pred)

    return {
        'peak_count': int(np.sum(peak_mask)),
        'peak_threshold': float(threshold_value),
        'peak_precision': float(precision),
        'peak_recall': float(recall),
        'peak_f1_score': float(f1_score),
        'peak_r2': float(peak_r2),
        'peak_rmse': float(peak_rmse),
        'peak_bias': float(peak_bias),
        'peak_mean_true': float(np.mean(peak_true)),
        'peak_mean_pred': float(np.mean(peak_pred))
    }


def compute_comprehensive_metrics(y_true: Union[torch.Tensor, np.ndarray],
                                 y_pred: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
    """
    计算综合评估指标，包括整体、分层和峰值指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        综合指标字典
    """
    from typing import Any

    # 整体指标
    overall_metrics = compute_all_metrics(y_true, y_pred)

    # 分层指标
    stratified_metrics = compute_stratified_metrics(y_true, y_pred)

    # 峰值指标
    peak_metrics = compute_peak_flow_metrics(y_true, y_pred)

    return {
        'overall': overall_metrics,
        'stratified': stratified_metrics,
        'peak_flow': peak_metrics
    }