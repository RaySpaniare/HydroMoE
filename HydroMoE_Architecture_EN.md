# HydroMoE v1.0 Architecture Detailed Description

## 1. Abstract

`HydroMoE v1.0` is a Mixture-of-Experts (MoE) deep learning model for basin runoff forecasting. Its core design innovatively fuses physics-based models (PBM) with specialized neural network (NN) experts to leverage both the interpretability and conservation properties of physical models and the flexible nonlinear fitting ability of data-driven models. The architecture is modular and decomposes the hydrological cycle into four core modules: Snow (accumulation and melt), Runoff generation, Evapotranspiration (ET), and Drainage & baseflow.

Within each module, the PBM expert and NN expert operate in parallel. A module-level gate network dynamically learns how to combine them. Sequence context is extracted by a multi-head self-attention based encoder. Module outputs are integrated using a two-way convex combination and a soft "available water" upper bound mechanism to produce final runoff predictions. Additionally, a regime-specific residual calibration module provides refined corrections based on flow regimes (low/mid/high).

Training incorporates several advanced strategies such as station-level sample weighting based on historical performance and a risk-refinement retraining procedure for low-performing stations. The design aims to deliver high accuracy, strong generalization, and interpretability while remaining flexible for future research and improvements.

---

## 2. Architectural Philosophy

`HydroMoE` is rooted in hybrid modeling: no single model can perfectly capture all aspects of hydrological processes. PBMs are strong at enforcing conservation laws and macro-scale behavior, while NNs excel at learning complex local nonlinearities. `HydroMoE` is built to combine the strengths of both.

### 2.1 Hybrid Mixture-of-Experts

Unlike classical MoE which tends to route inputs to a single "best" expert, `HydroMoE` uses a cooperative expert paradigm inside each hydrological module:

- PBM expert: provides a physically-grounded baseline prediction.
- NN expert: learns complex, data-driven corrections and patterns that PBM cannot capture.

A gate network dynamically decides how much to trust the PBM vs the NN given the current input.

### 2.2 Modular Decoupling of Physical Processes

The hydrological cycle is decomposed into four modules which provides:

- Better interpretability: examine module-specific behavior (e.g., snow melt, baseflow).
- Easier targeted improvements: replace or improve a single module without reworking the whole model.
- Reduced learning complexity: each NN expert focuses on a specific sub-process.

### 2.3 End-to-End Training with Physical Priors

Although the PBM expert's parameters are optimized offline, the full hybrid model (NN experts and gating networks) is trained end-to-end. Gradients flow to NN experts and gates, not into PBM parameters. The learning objective is to learn when to rely on the physical model rather than to modify the physical laws themselves.

### 2.4 Hierarchical Gating

Two hierarchical gating levels provide coarse-to-fine decisions:

1. Module-level gating: decides the PBM/NN mixture per module.
2. Regime calibration gating: a final residual correction based on the current flow regime (low/mid/high).

---

## 3. Data Processing & Feature Engineering

High-quality data and features are essential. `HydroMoE` uses an efficient and robust pipeline that converts long-table station time series (CSV/Parquet) into model-ready normalized sequence samples.

Default driver features: `precip`, `temp`, `pet`. Target: `runoff` (daily).

### 3.1 Smart Data Loading & Caching (`MoE_data_loader.py`, `MoE_data_utils.py`)

To handle large-scale time series, the loader implements:

- Two-level caching: file-level (Parquet/Feather caches) and in-memory `_GLOBAL_DATA_CACHE` for session reuse.
- Automatic format detection (`read_table_auto`) which reads Parquet/Feather preferentially; for CSV it infers dtypes and uses float32 and categorical encodings to reduce memory.
- Sequence caching: once sequences are built (given sequence_length, stride, feature_cols), they are cached for reuse when configuration parameters match.

### 3.2 Advanced Normalization (`MoE_advanced_normalization.py`)

Runoff often follows a highly skewed distribution. `HydroLogNormalizer` applies a log1p transform plus robust scaling using median and MAD (median absolute deviation) to reduce sensitivity to extreme events.

Normalization (forward):

- y' = log1p(y + c)
- y_norm = (y' - median(y')) / MAD(y')

Inverse:

- y = expm1(y' * MAD + median) - c
- final output clamped to min 0.0.

### 3.3 Feature Augmentation (`MoE_lowflow_augment.py` & `MoE_feature_engineering.py`)

- Low-flow augmentation: `augment_csv_with_runoff_lags` creates lagged runoff features and rolling stats to capture autocorrelation and improve low-R² station performance. All lagged features use strict history (shift(1)) to avoid leakage.

- HydroFeatureEngineer (optional): creates multi-window statistics, seasonal encodings, extreme-event flags, interaction features (e.g., precip - pet), and trend features.

### 3.4 Dataset Construction (`FixedHydroDataset`)

Datasets are constructed by time-based splits (recommended) or proportion-based splits and then sliced into sliding windows of `sequence_length` with stride `sequence_stride`.

Key sample items returned by the DataLoader:

- `features`: [sequence_length, num_features]
- `targets`: scalar (last timestep runoff)
- `targets_seq`: [sequence_length]
- `time_features`: [sequence_length, num_time_features]
- `raw_features_last`: unnormalized features at last timestep (PET, precip, temp) for PBM
- `station_id`, `station_idx`

Test set uses stride=1 to produce daily predictions.

---

## 4. Core Model Architecture (`MoE_hybrid_model.py`)

The model follows an "encode → process → combine → calibrate" pipeline.

### 4.1 Input Encoding & Temporal Context

- `feature_encoder`: linear projection from input features d_input to d_model.
- `HydroAttentionBlock`: stacked multi-head self-attention with positional encoding and RMSNorm.
- Pre-norm RMSNorm improves stability; FFN uses Linear(d_model, 4*d_model) → GELU → Dropout → Linear(4*d_model, d_model).

The last time step of the attention output is used as `module_input` for all modules.

### 4.2 Modular Hydrological Processes

Each module (snow, runoff, et, drainage) follows the PBM + NN expert pattern with a `ModuleGate` computing a softmax-weighted mixture:

Output_module = w_pbm * Output_pbm + w_nn * Output_nn

#### NN Expert:
- `MLPExpert` by default uses hidden_dim = d_model // 2 and num_layers = 2, producing a scalar output per module.

#### PBM Expert:
- Implemented as a batch-vectorized, stateless physical calculator using station-specific CMA-ES parameters. PBM runs under `torch.no_grad()` and does not receive gradients.

#### ModuleGate:
- Small MLP producing logits which are temperature-scaled and softmaxed to yield [w_pbm, w_nn]. A learnable bias `b_expert` and temperature `tau_module` control the soft/hardness.

### 4.3 Final Runoff Combination & Physical Constraints

- Two-way convex combination: an `alpha_head` computes allocation between quick flow (from runoff module) and baseflow (from drainage module), followed by `softplus` to ensure non-negativity and convex combination.

- Available Water Soft Upper Bound: available water A ≈ ReLU(precip + snow_output - pet). An `avail_head` maps components to an A_mapped upper bound. The final smoothing min uses:

Q_final = A_mapped - softplus(A_mapped - Q_comb)

This yields a differentiable soft-min preventing the final runoff from exceeding available water.

### 4.4 Two-Stage Calibration

1) Internal regime residual calibration: a `regime_encoder` pools attention outputs and a `regime_gate` outputs weights for low/mid/high experts which predict residuals ΔQ_i. The weighted sum ΔQ is scaled by a small trainable factor s and added to Q_final with clamp to non-negative.

2) External station affine calibration: `CalibratedHybridModel` wraps the base model and learns a station embedding and a small affine correction (scale, shift) based on station_vec, regime weights, lon/lat. Scale and shift are constrained via tanh so calibration is a subtle adjustment.

---

## 5. Training Strategy & Losses

### 5.1 Enhanced Trainer (`MoE_trainer_enhanced.py`)

- Station sample weighting: stations with low historical R² receive higher loss weight to focus training on weak stations.

w_station = 1 + λ * clamp(0.5 - R2_station, 0, 1)

### 5.2 Compound Losses (`MoE_losses.py`)

- Core loss: MSE.
- KGE loss: targets Kling-Gupta Efficiency directly.
- StationR2Loss: per-station 1 - R² averaged across station groups.
- WeightedHydroLoss: gives more weight to high-flow events.
- ExpertSpecializationLoss: encourages gate specialization by penalizing very uniform gate distributions.

Gradient clipping is applied after backprop to prevent exploding gradients.

---

## 6. Evaluation & Risk Optimization

### 6.1 Metrics (`MoE_metrics.py`)

R², KGE, RMSE, Bias, and other hydrology-focused metrics are computed per station and globally.

### 6.2 Risk Refinement (`MoE_risk_refiner.py`)

Low-performing stations (R² < threshold) are identified and the model is fine-tuned on those stations with targeted data filtering, smaller LR, and specialized early stopping based on average R² for risk stations.

---

## 7. Configuration Overview

- DataConfig / FixedDataConfig: data paths, `feature_cols`, `target_col`, `sequence_length`, time split definitions.
- ModelConfig: `input_size`, `d_model`, `num_heads`, `num_attention_layers`, `module_gate_temperature`, `pbm_min_weight`, etc.
- TrainingConfig: `batch_size`, `learning_rate`, `scheduler`, `gradient_clip`, `eval_every`, `early_stopping_patience`.
- SystemConfig: `device` ("auto"), `num_workers`, `pin_memory` (Windows defaults are set for compatibility).

Configs can be loaded via `HydroMoEConfig.from_dict()`.

---

## 8. CMA-ES Integration & PBM Robustness

`CMAESParamLoader` loads station-specific PBM parameters (default file `cmaes_optimal_params.json`) and caches them. PBM uses station parameters when available and falls back to defaults or a simple NN when not.

PBM computations are vectorized and constrained (clamping, softplus) to keep results non-negative and within reasonable ranges.

---

## 9. Global Caching & Warmup

- `_GLOBAL_DATA_CACHE` caches raw_data, filtered views, sequences, and scalers.
- `preload_all_datasets` warms the cache to reduce startup latency.
- `clear_data_cache` frees memory when configuration changes.

---

## 10. Attention Encoder Stack

- Default 2 stacked `HydroAttentionBlock` layers. Gradient checkpointing optional to trade compute for memory.
- Pre-norm RMSNorm and FFN with dropout and GELU.
- Multiscale attention modules available in `MoE_multiscale_attention.py` but not enabled by default.

---

## 11. Training Loop & Validation

- `EnhancedTrainer` drives epochs, training steps, validation, and early stopping.
- Saving policy: best model (by validation metric) saved as `outputs/enhanced_hydromoe_best.pth` with backups.

---

## 12. Risk Refinement Workflow

- Identify risk stations (default r2_threshold = 0.2), build filtered datasets, fine-tune the model only on those stations with smaller LR and shorter cycles.

---

## 13. Gating Mechanics

- `ModuleGate` supports soft/hard routing (Top-K or Gumbel-softmax), temperature control, and optional quality-guidance via a performance tracker.
- `RegimeGate` pools sequence context and produces regime weights used internally and also by station calibration.

---

## 14. Inference & Diagnostics

- `MoE_export_gate_timeseries.py` exports per-time-step predictions, gate weights, available water, and calibration params for diagnostic analysis.
- `MoE_evaluator_simple.py` computes global and per-station metrics and exports `station_expert_weights.csv` and other artifacts.

---

## 15. End-to-End Pipeline

1. Raw long-table CSV → optional lag augmentation → dataset construction → normalization → sliding-window sequences → DataLoader
2. Model: feature encoder → attention → 4 modules (PBM + NN + gate) → convex combination & soft upper bound → internal regime calibration → optional station affine calibration
3. Loss computation, gradient clipping, optimizer updates
4. Evaluation and export of diagnostics

---

## 16. Monitoring & Debugging

- Gradient health check tools, gate usage statistics, cache info utilities, and recommendations for addressing expert collapse or abnormal gate behaviors.

---

## 17. Conclusion

`HydroMoE v1.0` provides a modular, physically-informed, and flexible framework that combines physics-based priors and data-driven correction to produce accurate and interpretable runoff forecasts. The design balances computational efficiency, interpretability, and extensibility for future research directions such as multiscale attention or online adaptation of PBM parameters.
