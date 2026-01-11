"""
风险站点再训练模块（不泄露测试集）

功能：
- 验证集按站点计算R²
- 与 CMA-ES best_r2 低值站点求交集，识别高风险站点
- 仅对高风险站点进行小学习率短期再训练，早停依据为这些站点在验证集上的均值R²

用法（从主程序调用）：
    from MoE_risk_refiner import run_risk_refine
    run_risk_refine(model, train_loader, val_loader, device,
                    r2_threshold=0.2, epochs=8, lr=5e-5, patience=3)
"""

from typing import Dict, List, Tuple

import torch
import numpy as np

from MoE_metrics import compute_all_metrics
from MoE_cmaes_loader import CMAESParamLoader


def _collect_station_predictions(model, loader, device) -> Dict[str, Dict[str, List[float]]]:
    """收集 loader 内每个站点的预测与真实。

    返回：{sid: {'y_true': [..], 'y_pred': [..]}}
    """
    model.eval()
    station_to_pairs: Dict[str, Dict[str, List[float]]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
            out = model(batch)
            y_pred = out['runoff']
            y_true = batch['targets']
            yp = y_pred.detach().cpu().numpy().reshape(-1)
            yt = y_true.detach().cpu().numpy().reshape(-1)
            sids = batch.get('station_id', [None] * len(yp))
            for i in range(len(yp)):
                sid = sids[i]
                if sid is None:
                    continue
                rec = station_to_pairs.setdefault(sid, {'y_true': [], 'y_pred': []})
                rec['y_true'].append(float(yt[i]))
                rec['y_pred'].append(float(yp[i]))
    return station_to_pairs


def get_validation_station_r2(model, val_loader, device) -> Dict[str, float]:
    """计算验证集每个站点R²。"""
    pairs = _collect_station_predictions(model, val_loader, device)
    sid_to_r2: Dict[str, float] = {}
    for sid, d in pairs.items():
        m = compute_all_metrics(d['y_true'], d['y_pred'])
        sid_to_r2[sid] = float(m.get('R2', 0.0))
    return sid_to_r2


def identify_high_risk_stations(val_r2_map: Dict[str, float],
                                cmaes_loader: CMAESParamLoader,
                                r2_threshold: float = 0.2) -> Tuple[List[str], List[str], List[str]]:
    """基于 验证R² 与 CMA-ES best_r2 的交集识别高风险站点。"""
    val_low = {sid for sid, r2 in val_r2_map.items() if r2 < r2_threshold}
    cmaes_low = set()
    try:
        for sid, rec in (cmaes_loader.params_data or {}).items():
            try:
                br2 = float(rec.get('best_r2', rec.get('r2', rec.get('R2', 0.0))))
            except Exception:
                br2 = 0.0
            if br2 < r2_threshold:
                cmaes_low.add(str(sid))
    except Exception:
        pass
    risk = sorted(list(val_low.intersection(cmaes_low)))
    return risk, sorted(list(val_low)), sorted(list(cmaes_low))


def _filter_batch_by_stations(batch: Dict, risk_set: set, device) -> Dict:
    """过滤出 batch 中属于风险站点的子批次；若无则返回 None。"""
    sids = batch.get('station_id', [])
    idxs = [i for i, sid in enumerate(sids) if sid in risk_set]
    if not idxs:
        return None
    sub = {}
    for k, v in batch.items():
        if hasattr(v, 'index_select') and getattr(v, 'dim', lambda: 0)() >= 1:
            sub[k] = v.index_select(0, torch.tensor(idxs, dtype=torch.long, device=v.device))
        elif isinstance(v, list):
            sub[k] = [v[i] for i in idxs]
        else:
            sub[k] = v
    return sub


def finetune_on_risk_stations(model,
                              train_loader,
                              val_loader,
                              device,
                              risk_stations: List[str],
                              epochs: int = 8,
                              lr: float = 5e-5,
                              patience: int = 3) -> float:
    """对高风险站点做小学习率再训练，早停依据为验证集上这些站点的均值R²。"""
    risk_set = set(risk_stations)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_state = None
    best_score = -1e9
    bad = 0
    print(f"\n🔁 高风险站点再训练: {len(risk_set)} 个站点, lr={lr}, epochs={epochs}")
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            sub = _filter_batch_by_stations(batch, risk_set, device)
            if sub is None:
                continue
            sub = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in sub.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(sub, return_gate_info=False)
            loss = torch.nn.functional.mse_loss(out['runoff'], sub['targets'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        avg_loss = total_loss / max(1, steps)
        # 验证（仅风险站点）
        val_map = get_validation_station_r2(model, val_loader, device)
        vals = [r2 for sid, r2 in val_map.items() if sid in risk_set]
        mean_r2 = float(np.mean(vals)) if vals else -1e9
        print(f"  Epoch {ep+1}/{epochs}: train_loss={avg_loss:.4f}, risk_val_R2={mean_r2:.4f}")
        if mean_r2 > best_score + 1e-4:
            best_score = mean_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("  ⏹️ 早停触发")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score


def run_risk_refine(model,
                    train_loader,
                    val_loader,
                    device,
                    r2_threshold: float = 0.2,
                    epochs: int = 8,
                    lr: float = 5e-5,
                    patience: int = 3) -> Dict[str, object]:
    """一键执行：识别高风险站点并再训练。返回元信息。"""
    print("\n🔎 基于验证R²与CMA-ES先验识别高风险站点...")
    val_r2_map = get_validation_station_r2(model, val_loader, device)
    cmaes_loader = CMAESParamLoader()
    risk, val_low, cmaes_low = identify_high_risk_stations(val_r2_map, cmaes_loader, r2_threshold=r2_threshold)
    print(f"  验证低R²站点: {len(val_low)}，CMA-ES低R²站点: {len(cmaes_low)}，交集(高风险): {len(risk)}")
    best_r = None
    if len(risk) > 0:
        best_r = finetune_on_risk_stations(model, train_loader, val_loader, device, risk,
                                           epochs=epochs, lr=lr, patience=patience)
        print(f"  ✅ 再训练完成，高风险站点验证均值R²={best_r:.4f}")
    return {
        'risk': risk,
        'val_low': val_low,
        'cmaes_low': cmaes_low,
        'best_risk_val_r2': best_r
    }


