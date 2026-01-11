"""
简化评估模块 - 包含模型评估相关功能
"""

import torch
import numpy as np
import logging
from datetime import datetime
import os
import glob
import pandas as pd

from MoE_metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def evaluate_enhanced_model(model, test_loader, device, output_prefix: str = 'enhanced_real_runoff_predictions'):
    """评估增强模型：
    - 收集预测与目标
    - 反归一化到真实单位（若可用）
    - 保存时间序列 CSV 与站点评估 CSV
    - 打印总体指标
    """
    print("\n📊 评估增强模型...")

    model.eval()
    preds_list, targets_list = [], []
    station_names, lons, lats, dates = [], [], [], []
    # 为逐日重建准备：收集每条样本的站点与日期范围
    records = []  # 每条记录包含: station_id, start_date, end_date, pred, target, lon, lat

    # 站点门控使用率聚合器
    gate_usage = {}

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch, return_gate_info=True)
            predictions = outputs['runoff']

            # 🚀 优化：减少reshape操作，直接flatten
            preds_np = predictions.cpu().numpy().flatten()
            targs_np = batch['targets'].cpu().numpy().flatten()
            preds_list.append(preds_np)
            targets_list.append(targs_np)

            if 'station_id' in batch:
                station_names.extend(batch['station_id'])
            if 'lon' in batch:
                try:
                    lons.extend(batch['lon'].cpu().numpy().reshape(-1).tolist())
                except Exception:
                    pass
            if 'lat' in batch:
                try:
                    lats.extend(batch['lat'].cpu().numpy().reshape(-1).tolist())
                except Exception:
                    pass
            # 记录窗口覆盖信息（逐日重建用）+ 收集日期信息到dates列表
            try:
                sid_list = batch.get('station_id', [])
                sd_list = batch.get('start_date', [])
                ed_list = batch.get('end_date', [])
                for i in range(len(preds_np)):
                    sid = sid_list[i] if i < len(sid_list) else None
                    sd = sd_list[i] if i < len(sd_list) else None
                    ed = ed_list[i] if i < len(ed_list) else None
                    
                    # 🔥 修复：将end_date添加到dates列表中（预测对应的日期）
                    if ed is not None:
                        dates.append(ed)
                    elif sd is not None:
                        dates.append(sd)
                    else:
                        dates.append(None)
                    
                    lon_i = None
                    lat_i = None
                    try:
                        if 'lon' in batch:
                            lon_i = float(batch['lon'][i].item()) if hasattr(batch['lon'][i], 'item') else float(batch['lon'][i])
                        if 'lat' in batch:
                            lat_i = float(batch['lat'][i].item()) if hasattr(batch['lat'][i], 'item') else float(batch['lat'][i])
                    except Exception:
                        pass
                    records.append({
                        'station_id': sid,
                        'start_date': sd,
                        'end_date': ed,
                        'pred': float(preds_np[i]),
                        'target': float(targs_np[i]),
                        'lon': lon_i,
                        'lat': lat_i,
                    })
            except Exception:
                pass

            # 收集门控使用率（PBM vs NN）与Regime权重
            try:
                sid_list = batch['station_id'] if 'station_id' in batch else None
                if sid_list is not None and 'gate_info' in outputs:
                    module_gates = outputs['gate_info'].get('module_gates', {})
                    regime_w = outputs.get('regime_weights', None)
                    bsz = predictions.shape[0]
                    for i in range(bsz):
                        sid = sid_list[i]
                        rec = gate_usage.setdefault(sid, {
                            'count': 0,
                            'snow_pbm': 0.0, 'snow_nn': 0.0,
                            'runoff_pbm': 0.0, 'runoff_nn': 0.0,
                            'et_pbm': 0.0, 'et_nn': 0.0,
                            'drainage_pbm': 0.0, 'drainage_nn': 0.0,
                            'regime_low': 0.0, 'regime_mid': 0.0, 'regime_high': 0.0,
                        })
                        rec['count'] += 1
                        for mname in ['snow', 'runoff', 'et', 'drainage']:
                            if mname in module_gates and 'effective_gate' in module_gates[mname]:
                                gw = module_gates[mname]['effective_gate']  # [B,2]
                                pbm_w = float(gw[i, 0].detach().cpu().item())
                                nn_w = float(gw[i, 1].detach().cpu().item())
                                rec[f'{mname}_pbm'] += pbm_w
                                rec[f'{mname}_nn'] += nn_w
                        if regime_w is not None:
                            rw = regime_w  # [B,3]
                            rec['regime_low'] += float(rw[i, 0].detach().cpu().item())
                            rec['regime_mid'] += float(rw[i, 1].detach().cpu().item())
                            rec['regime_high'] += float(rw[i, 2].detach().cpu().item())
            except Exception:
                pass

    if not preds_list:
        print("⚠️ 没有可用的测试数据")
        return {'R2': 0.0, 'KGE': 0.0, 'RMSE': float('inf')}

    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # 反归一化（若可用）
    ds = getattr(test_loader, 'dataset', None)
    target_scaler = getattr(ds, 'scalers', {}).get('target_scaler') if getattr(ds, 'scalers', None) else None
    if target_scaler is not None:
        try:
            predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
            predictions = np.clip(predictions, a_min=0.0, a_max=None)
        except Exception:
            pass

    # 指标
    metrics = compute_all_metrics(targets, predictions)

    print("🎯 最终测试结果:")
    print(f"  R²: {metrics.get('R2', 0):.4f}")
    print(f"  KGE: {metrics.get('KGE', 0):.4f}")
    print(f"  RMSE: {metrics.get('RMSE', 0):.4f}")
    print(f"  Bias: {metrics.get('bias', 0):.4f}")

    # 保存CSV（原始拼接）
    out_dir = os.path.join('outputs', output_prefix)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({
        'predicted_runoff': predictions,
        'actual_runoff': targets,
    })
    if station_names:
        df['station_id'] = station_names[:len(df)]
    if lons:
        df['lon'] = (lons[:len(df)] if len(lons) >= len(df) else lons + [None]*(len(df)-len(lons)))
    if lats:
        df['lat'] = (lats[:len(df)] if len(lats) >= len(df) else lats + [None]*(len(df)-len(lats)))
    if dates:
        try:
            df['date'] = pd.to_datetime(dates[:len(df)])
        except Exception:
            df['date'] = dates[:len(df)]

    # 🔥 现在测试集使用stride=1，已经包含逐日预测，只需补充经纬度信息
    if 'station_id' in df.columns:
        try:
            # 映射lon/lat（每站固定值）
            if 'lon' in df.columns:
                lon_map = df[['station_id', 'lon']].dropna().drop_duplicates(subset=['station_id']).set_index('station_id')['lon'].to_dict()
                df['lon'] = df['station_id'].map(lon_map).fillna(df['lon'])
            if 'lat' in df.columns:
                lat_map = df[['station_id', 'lat']].dropna().drop_duplicates(subset=['station_id']).set_index('station_id')['lat'].to_dict()
                df['lat'] = df['station_id'].map(lat_map).fillna(df['lat'])
        except Exception:
            pass
    
    # 🔥 修复：只保留真正的测试期预测（从test_start开始）
    ds_cfg = getattr(getattr(test_loader, 'dataset', None), 'config', None)
    if ds_cfg is not None and 'date' in df.columns:
        try:
            test_start = pd.to_datetime(ds_cfg.test_start)
            test_end = pd.to_datetime(ds_cfg.test_end)
            
            # 筛选真正的测试期数据
            test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)
            df_before_filter = df.copy()
            df = df[test_mask].copy()
            
            print(f"📊 筛选测试期预测完成:")
            print(f"   - 扩展数据点: {len(df_before_filter)}")
            print(f"   - 测试期数据点: {len(df)} (从 {test_start.date()} 到 {test_end.date()})")
            print(f"   - 有效预测点: {df['predicted_runoff'].notna().sum()}")
            print(f"   - 覆盖率: {df['predicted_runoff'].notna().sum()/len(df)*100:.1f}%")
        except Exception as e:
            print(f"   ⚠️ 测试期筛选失败: {e}")
            print(f"📊 生成逐日预测完成:")
            print(f"   - 总数据点: {len(df)}")
            print(f"   - 有效预测点: {df['predicted_runoff'].notna().sum()}")
            print(f"   - 覆盖率: {df['predicted_runoff'].notna().sum()/len(df)*100:.1f}%")
    else:
        print(f"📊 生成逐日预测完成:")
        print(f"   - 总数据点: {len(df)}")
        print(f"   - 有效预测点: {df['predicted_runoff'].notna().sum()}")
        print(f"   - 覆盖率: {df['predicted_runoff'].notna().sum()/len(df)*100:.1f}%")

    # 误差列在补齐后再计算（保留NaN位置）
    df['error'] = df['predicted_runoff'] - df['actual_runoff']

    # 相对误差（避免除 0）
    df['relative_error_percent'] = ((df['error']) / (df['actual_runoff'] + 1e-8)) * 100

    csv_path = os.path.join(out_dir, 'real_runoff_predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"  💾 时间序列CSV已保存: {csv_path}")

    # 额外导出：逐站点逐日期对齐的去重版本（每站一天一行，预测与真实均有效）
    try:
        if 'station_id' in df.columns and 'date' in df.columns:
            df_aligned = df[['station_id','date','lon','lat','actual_runoff','predicted_runoff']].copy()
            # 丢弃无效对
            df_aligned = df_aligned[np.isfinite(df_aligned['actual_runoff']) & np.isfinite(df_aligned['predicted_runoff'])]
            # 可能存在同一站点-日期多条记录（窗口重叠），取均值并保留首个经纬度
            agg = {
                'actual_runoff': 'mean',
                'predicted_runoff': 'mean',
                'lon': 'first',
                'lat': 'first',
            }
            df_aligned = df_aligned.groupby(['station_id','date'], as_index=False).agg(agg)
            aligned_path = os.path.join(out_dir, 'real_runoff_predictions_aligned.csv')
            df_aligned.to_csv(aligned_path, index=False)
            print(f"  💾 对齐CSV已保存: {aligned_path}")
    except Exception as e:
        print(f"⚠️ 生成对齐CSV失败: {e}")


    # 站点评估
    if 'station_id' in df.columns:
        station_stats = []
        for sid, g in df.groupby('station_id'):
            y_true = g['actual_runoff'].to_numpy()
            y_pred = g['predicted_runoff'].to_numpy()
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            valid_count = int(mask.sum())
            m = compute_all_metrics(y_true[mask], y_pred[mask]) if valid_count > 0 else {'R2': np.nan, 'KGE': np.nan, 'RMSE': np.nan, 'bias': np.nan}
            lon_val = float(g['lon'].iloc[0]) if 'lon' in g.columns and len(g['lon'])>0 and pd.notna(g['lon'].iloc[0]) else np.nan
            lat_val = float(g['lat'].iloc[0]) if 'lat' in g.columns and len(g['lat'])>0 and pd.notna(g['lat'].iloc[0]) else np.nan
            station_stats.append({
                'station_id': sid,
                'lon': lon_val,
                'lat': lat_val,
                'days_total': int(len(g)),
                'sample_count': valid_count,
                'mean_actual_runoff': float(np.nanmean(g['actual_runoff'])),
                'mean_predicted_runoff': float(np.nanmean(g['predicted_runoff'])),
                'rmse': float(np.sqrt(np.nanmean((g['error'])**2))),
                'R2': m.get('R2', np.nan),
                'KGE': m.get('KGE', np.nan),
            })
        stats_df = pd.DataFrame(station_stats)
        stats_path = os.path.join(out_dir, 'station_performance_real_runoff.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"  💾 站点评估CSV已保存: {stats_path}")

        # 简要概览
        if len(stats_df) > 0 and 'R2' in stats_df.columns and stats_df['R2'].notna().any():
            best_row = stats_df.loc[stats_df['R2'].idxmax()]
            worst_row = stats_df.loc[stats_df['R2'].idxmin()]
            print(f"  🏆 最佳站点: {best_row['station_id']} (R²={best_row['R2']:.3f})")
            print(f"  🎯 待提升站: {worst_row['station_id']} (R²={worst_row['R2']:.3f})")

        # 站点层面汇总统计（均值/中位数）
        try:
            r2_vals = stats_df['R2'].dropna()
            kge_vals = stats_df['KGE'].dropna()
            if len(r2_vals) > 0:
                r2_mean = float(r2_vals.mean())
                r2_median = float(r2_vals.median())
                print(f"  📦 站点R²: 均值={r2_mean:.4f}, 中位数={r2_median:.4f}")
            if len(kge_vals) > 0:
                kge_mean = float(kge_vals.mean())
                kge_median = float(kge_vals.median())
                print(f"  📦 站点KGE: 均值={kge_mean:.4f}, 中位数={kge_median:.4f}")

            # 保存汇总JSON
            import json
            summary = {
                'station_count': int(len(stats_df)),
                'R2': {
                    'mean': float(r2_vals.mean()) if len(r2_vals)>0 else None,
                    'median': float(r2_vals.median()) if len(r2_vals)>0 else None,
                    'p25': float(r2_vals.quantile(0.25)) if len(r2_vals)>0 else None,
                    'p10': float(r2_vals.quantile(0.10)) if len(r2_vals)>0 else None,
                    'min': float(r2_vals.min()) if len(r2_vals)>0 else None,
                    'max': float(r2_vals.max()) if len(r2_vals)>0 else None,
                },
                'KGE': {
                    'mean': float(kge_vals.mean()) if len(kge_vals)>0 else None,
                    'median': float(kge_vals.median()) if len(kge_vals)>0 else None,
                    'p25': float(kge_vals.quantile(0.25)) if len(kge_vals)>0 else None,
                    'p10': float(kge_vals.quantile(0.10)) if len(kge_vals)>0 else None,
                    'min': float(kge_vals.min()) if len(kge_vals)>0 else None,
                    'max': float(kge_vals.max()) if len(kge_vals)>0 else None,
                },
            }
            summary_path = os.path.join(out_dir, 'station_metrics_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"  💾 站点指标汇总已保存: {summary_path}")

            # 导出低R²站点列表（Top-N或底部10%）
            if len(stats_df) > 0 and 'R2' in stats_df.columns:
                n = max(5, min(10, len(stats_df)))
                low_df = stats_df.sort_values('R2', ascending=True).head(n)
                low_cols = [c for c in ['station_id','lon','lat','R2','KGE','rmse','mean_actual_runoff','mean_predicted_runoff','sample_count'] if c in low_df.columns]
                low_path = os.path.join(out_dir, 'low_r2_stations.csv')
                low_df[low_cols].to_csv(low_path, index=False)
                names = ", ".join([f"{row.station_id}(R²={row.R2:.3f})" for _, row in low_df.iterrows()])
                print(f"  📉 低R²站点Top{len(low_df)}: {names}")
                print(f"  💾 已导出低R²站点列表: {low_path}")
        except Exception as e:
            print(f"⚠️ 站点层面统计失败: {e}")

        # 导出门控使用率（如已收集）
        try:
            if gate_usage:
                usage_rows = []
                for sid, rec in gate_usage.items():
                    c = rec.pop('count', 1)
                    row = {'station_id': sid}
                    for k, v in rec.items():
                        row[k] = v / max(1, c)
                    usage_rows.append(row)
                usage_df = pd.DataFrame(usage_rows)
                # 尽量补充经纬度
                if 'station_id' in stats_df.columns:
                    usage_df = usage_df.merge(stats_df[['station_id','lon','lat']], on='station_id', how='left')
                # 固定列顺序
                ordered_cols = [
                    'station_id','lon','lat',
                    'snow_pbm','snow_nn',
                    'runoff_pbm','runoff_nn',
                    'et_pbm','et_nn',
                    'drainage_pbm','drainage_nn',
                    'regime_low','regime_mid','regime_high'
                ]
                final_cols = [c for c in ordered_cols if c in usage_df.columns]
                usage_df = usage_df[final_cols]

                # 仅保留一个文件：station_expert_weights.csv
                expert_path = os.path.join(out_dir, 'station_expert_weights.csv')
                usage_df.to_csv(expert_path, index=False)
                print(f"  💾 专家/门控权重CSV已保存: {expert_path}")
        except Exception as e:
            print(f"⚠️ 导出专家/门控权重失败: {e}")

    return metrics


def save_best_model_with_timestamp(model, base_dir="outputs", base_name="enhanced_hydromoe_best"):
    """保存带时间戳的最佳模型"""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(base_dir, f"{base_name}_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    
    # 同时保存一个不带时间戳的版本（用于加载）
    simple_path = os.path.join(base_dir, f"{base_name}.pth")
    torch.save(model.state_dict(), simple_path)
    
    return model_path


def _load_state_dict_partial(model, state):
    """名称+形状匹配的部分加载，并对线性层输入维度做自适应合并。"""
    ms = model.state_dict()
    compatible = {}
    adapted, mismatched = [], []
    for k, v in state.items():
        if k in ms:
            try:
                if ms[k].shape == v.shape:
                    compatible[k] = v
                else:
                    if len(ms[k].shape) == 2 and len(v.shape) == 2 and ms[k].shape[0] == v.shape[0]:
                        out_dim, in_model = ms[k].shape
                        _, in_ckpt = v.shape
                        merged = ms[k].clone()
                        copy_in = min(in_ckpt, in_model)
                        merged[:, :copy_in] = v[:, :copy_in]
                        compatible[k] = merged
                        adapted.append((k, (out_dim, in_ckpt), (out_dim, in_model), copy_in))
                    else:
                        mismatched.append((k, tuple(v.shape), tuple(ms[k].shape)))
            except Exception:
                mismatched.append((k, "?", "?"))
    if compatible:
        ms.update(compatible)
        model.load_state_dict(ms)
    print("🔄 简化加载器 - 部分/自适应加载摘要：")
    print(f"   ✅ 加载参数: {len(compatible)}")
    if adapted:
        for name, s_ckpt, s_model, copied in adapted[:5]:
            print(f"      - {name}: ckpt{s_ckpt} -> model{s_model}, 合并前 {copied} 列")
    if mismatched:
        print(f"   🧩 仍有形状不匹配(已忽略): {len(mismatched)}")
    return len(compatible) > 0


def load_best_model_if_exists(model, path="outputs/enhanced_hydromoe_best.pth"):
    """如果存在最佳模型则加载（部分加载+自适应，支持时间戳回退）"""
    def try_load(p):
        try:
            state = torch.load(p, map_location='cpu')
            print(f"🔄 尝试加载: {p}")
            if _load_state_dict_partial(model, state):
                print(f"✅ 加载最佳模型: {p}")
                return True
            return False
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")
            return False

    # 🚀 添加详细的文件检查
    print(f"🔍 检查模型文件: {path}")
    print(f"   - 文件存在: {os.path.exists(path)}")
    if os.path.exists(path):
        file_size = os.path.getsize(path)
        print(f"   - 文件大小: {file_size / 1024 / 1024:.2f} MB")
        
        if try_load(path):
            return True
        else:
            print(f"   ⚠️ 文件存在但加载失败")

    # 尝试寻找带时间戳的备份文件
    base_dir = os.path.dirname(path) or "."
    base_name = os.path.basename(path).replace('.pth', '')
    candidates = sorted(glob.glob(os.path.join(base_dir, f"{base_name}_*.pth")))
    
    if candidates:
        print(f"🔍 找到 {len(candidates)} 个候选模型文件:")
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for i, p in enumerate(candidates[:3]):  # 只显示最新的3个
            file_size = os.path.getsize(p) / 1024 / 1024
            print(f"   {i+1}. {p} ({file_size:.2f} MB)")
            
        for p in candidates:
            if try_load(p):
                return True

    print(f"❌ 未找到可用的预训练模型: {path}")
    return False


def analyze_model_complexity(model):
    """分析模型复杂度"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🔍 模型复杂度分析:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 分模块统计
    module_params = {}
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                module_params[name] = params
    
    # 显示主要模块
    print(f"  主要模块参数分布:")
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    for name, params in sorted_modules[:10]:  # 显示前10个最大的模块
        if params > 1000:  # 只显示参数数量大于1000的模块
            print(f"    {name}: {params:,}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024
    }


def quick_validation_metrics(model, val_loader, device, dataset=None):
    """快速验证指标计算"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(batch)
            predictions = outputs['runoff']
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch['targets'].cpu().numpy())
    
    if len(all_preds) > 0:
        return compute_all_metrics(all_targets, all_preds)
    else:
        return {'R2': 0.0, 'KGE': 0.0, 'RMSE': float('inf')}


def validate_model_simple(model, val_loader, criterion, device):
    """简单验证模型"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(batch)
            predictions = outputs['runoff']
            
            # 简单MSE损失
            loss = torch.nn.functional.mse_loss(predictions, batch['targets'])
            val_losses.append(loss.item())
    
    return np.mean(val_losses) if val_losses else float('inf')


def print_training_summary(epoch, epochs, train_loss, val_loss, val_metrics, patience_counter, patience):
    """打印训练摘要"""
    print(f"\n📅 Epoch {epoch+1}/{epochs}")
    print("-" * 50)
    print(f"  📊 训练损失: {train_loss:.4f}")
    print(f"  📈 验证损失: {val_loss:.4f}")
    print(f"  📋 验证指标:")
    print(f"    🎯 R²: {val_metrics.get('R2', 0):.4f}")
    print(f"    🎯 KGE: {val_metrics.get('KGE', 0):.4f}")
    print(f"    📊 RMSE: {val_metrics.get('RMSE', 0):.4f}")
    print(f"  ⏳ 未改进计数: {patience_counter}/{patience}")
