## HydroMoE v2.0 Architecture Guide

A physics–statistics hybrid Mixture-of-Experts framework for runoff forecasting that integrates CMA-ES calibrated station-level PBM, neural experts, attention-based sequence encoding, and targeted risk refinement tools.

---

### 1. Key Highlights
- **Physics + Deep Learning**: Four hydrological modules (snow, runoff, ET, groundwater) combine station-specific PBM (`MoE_hybrid_model.PBMExpert`) and MLP experts via dynamic gating.
- **Temporal Context Encoding**: Positional encoding + multi-head attention (see `MoE_attention.py`) with optional multiscale attention modules.
- **Regime & Station Calibration**: Internal regime-based residual MoE and optional station-level affine calibration (`MoE_station_regime_calibration.py`).
- **Robust data pipeline**: station-wise normalizers, feature augmentation, caching, and low-R² targeted retraining.
- **Engineering tools**: Enhanced trainer (station weighting, gate entropy monitoring), diagnostics, and augmentation scripts.

---

### 2. Code Layout
- `MoE_config.py`: config dataclasses (HydroMoEConfig, DataConfig, ModelConfig)
- `MoE_data_*`: data configuration, caching, and loaders
- `MoE_feature_engineering.py`: feature creation
- `MoE_advanced_normalization.py`: robust normalizers and `HydroLogNormalizer`
- `MoE_hybrid_model.py`: core hybrid model with PBM experts, gates, alpha_head, avail_head, regime MoE
- `MoE_pbm.py` / `MoE_cmaes_loader.py`: PBM and CMA-ES parameter loading
- `MoE_attention.py` / `MoE_multiscale_attention.py`: attention blocks and experimental variants
- `MoE_gate.py` / `MoE_experts.py`: gate implementations and expert types
- `MoE_trainer_enhanced.py` / `MoE_losses.py`: enhanced trainer and loss functions
- `MoE_evaluator_simple.py` / `MoE_metrics.py`: evaluation and metrics
- `MoE_lowflow_augment.py`: lag and rolling augmentation
- `MoE_main_enhanced.py`: main training entry and enhanced workflow

---

### 3. Data Pipeline & Preprocessing
1. Raw long-table CSV with station_id, date, drivers, target.
2. Optional lag augmentation via `MoE_lowflow_augment.run_pipeline`.
3. `FixedHydroDataset` builds sequences with caching and robust scalers.
4. Normalization: site-level robust scaling (feature) + `HydroLogNormalizer` for target.
5. Sliding windows: `sequence_length`, `sequence_stride`, and stride=1 on test set.

---

### 4. Model Overview
1. Encoder: linear projection → attention blocks (RMSNorm + Multi-Head Attention + FFN)
2. Four module experts: each module is PBM + MLP expert fused by `ModuleGate`
3. Water balance: alpha head convexly combines quick and baseflow; avail_head enforces a soft upper bound
4. Regime residual MoE: internal Transformer-based regime encoder and 3 small regime experts
5. Station calibration: optional station embedding and affine calibration (scale/shift)

---

### 5. Training Flow
1. `MoE_main_enhanced.py` prepares data, builds model, and optionally loads previous best weights.
2. `EnhancedTrainer` performs train/validate loops, site weighting, gradient clipping, and checkpointing.
3. `MoE_risk_refiner.py` filters low-performing stations for targeted fine-tuning.
4. Ablation and low-R² retrain scripts are available for experiments.

Environment variables (Windows PowerShell):
```powershell
set EPOCHS=100
set EVAL_ONLY=1
set GATE_ENTROPY_W=0.02
set STATION_WEIGHTING=0
set USE_GRAD_CHECKPOINT=0
```

---

### 6. Outputs & Evaluation
- `MoE_evaluator_simple.evaluate_enhanced_model` reverses normalization and saves daily predictions, station metrics, gate weights, and summary JSON.
- Best model saved to `outputs/enhanced_hydromoe_best.pth` with timestamped backups.
- Key output directories: `outputs/enhanced_real_runoff_predictions/`, `outputs/lowR2/`, `outputs/augmented/`.

---

### 7. Quick Start
1. Point `FixedDataConfig.csv_path` to your data.
2. Install dependencies: Python 3.9+, PyTorch 2.0+, NumPy, Pandas; install `pyarrow` to enable Parquet.
3. Train:
```bash
python MoE_main_enhanced.py
```
4. Evaluate only:
```bash
set EVAL_ONLY=1
python MoE_main_enhanced.py
```
5. Run ablation:
```bash
python MoE_hybrid_ablation.py --variants runoff et --checkpoint outputs/enhanced_hydromoe_best.pth
```
6. Low R² retraining:
```bash
python MoE_main_low_r2.py
```

---

### 8. Extensions & Notes
- Add new features via `HydroFeatureEngineer` or modify `FixedDataConfig.feature_cols`.
- Extend normalization strategies by adding new Normalizer classes.
- Register new expert types in `MoE_experts.create_expert`.
- Modify gating temperature/top-k or switch to other routing strategies.
- Keep in mind Windows defaults (`num_workers=0`, `pin_memory=False`) for DataLoader compatibility.

Known caveats: default hard-coded hyperparameters may require tuning; if `cmaes_optimal_params.json` is missing PBM falls back to NN; turning off gradient checkpointing or enabling multiscale attention increases memory usage significantly.

---

### 9. Utilities
- `run_risk_refine` identifies and fine-tunes on risk stations.
- `merge_lowR2_results.py` aggregates multiple evaluation runs.
- Logs are stored in `hydromoe_v2_main_*.log` with run configs and environment info.

---

This README summarizes the HydroMoE v2.0 codebase and workflow. Please update the documentation alongside code changes to keep experiments reproducible and maintainable.
