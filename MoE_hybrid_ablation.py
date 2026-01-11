from __future__ import annotations

import argparse
import json
import logging
import os
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from MoE_config import get_default_config
from MoE_data_config import FixedDataConfig
from MoE_data_loader import clear_data_cache, create_fixed_data_loaders
from MoE_evaluator_simple import evaluate_enhanced_model, load_best_model_if_exists
from MoE_hybrid_model import create_hybrid_hydro_moe_model
from MoE_station_regime_calibration import wrap_with_calibration
from MoE_trainer_enhanced import EnhancedTrainer, enhanced_training_loop

logger = logging.getLogger(__name__)


Variant = Tuple[str, str, str]  # (module_name, output_prefix, description)


@dataclass
class FineTuneOptions:
    epochs: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    patience: int = 3
    unfreeze_gate: bool = True
    batch_size_override: Optional[int] = None
    checkpoint_every: int = 0  # 0 => disable interim checkpoints


ABLATION_VARIANTS: Sequence[Variant] = (
    ("snow", "ablation_snowNN", "Snow module -> NN, others -> PBM"),
    ("runoff", "ablation_runoffNN", "Runoff module -> NN, others -> PBM"),
    ("et", "ablation_etNN", "Evapotranspiration module -> NN, others -> PBM"),
    ("drainage", "ablation_drainageNN", "Drainage module -> NN, others -> PBM"),
)

MODULE_NAMES: Sequence[str] = ("snow", "runoff", "et", "drainage")

DEFAULT_CHECKPOINT = Path("outputs/enhanced_hydromoe_best.pth")
OUTPUT_ROOT = Path("outputs")


# ---------------------------------------------------------------------------
# Argument parsing and utility helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HydroMoE ablation runner with optional fine-tuning")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to pretrained HydroMoE checkpoint (default: outputs/enhanced_hydromoe_best.pth)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Subset of variants to run (choose from snow/runoff/et/drainage)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override for evaluation (and fine-tuning unless overridden)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Enable light fine-tuning for the NN branch of the ablated module",
    )
    parser.add_argument(
        "--ft-epochs",
        type=int,
        default=3,
        help="Fine-tuning epochs when --finetune is set (default: 3)",
    )
    parser.add_argument(
        "--ft-lr",
        type=float,
        default=5e-5,
        help="Fine-tuning learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--ft-weight-decay",
        type=float,
        default=0.0,
        help="Fine-tuning weight decay (default: 0.0)",
    )
    parser.add_argument(
        "--ft-patience",
        type=int,
        default=3,
        help="Fine-tuning early stopping patience (default: 3)",
    )
    parser.add_argument(
        "--ft-freeze-gate",
        action="store_true",
        help="Freeze module gate during fine-tuning (default: unfreeze)",
    )
    parser.add_argument(
        "--ft-checkpoint-every",
        type=int,
        default=0,
        help="Save interim checkpoints every N epochs during fine-tuning (0 disables)",
    )
    parser.add_argument(
        "--ft-filter-low-r2",
        action="store_true",
        help="Limit fine-tuning data to stations with R² < 0 from the merged results",
    )
    parser.add_argument(
        "--low-r2-csv",
        type=Path,
        default=Path("outputs/combined_lowR2/low_r2_stations.csv"),
        help="CSV listing low R² station IDs (default: outputs/combined_lowR2/low_r2_stations.csv)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Optional override for fine-tuning batch size (default: eval batch size)",
    )
    parser.add_argument(
        "--train-workers",
        type=int,
        default=None,
        help="Optional override for DataLoader num_workers during fine-tuning",
    )
    parser.add_argument(
        "--train-prefetch",
        type=int,
        default=None,
        help="Optional override for DataLoader prefetch_factor during fine-tuning",
    )
    parser.add_argument(
        "--train-pin-memory",
        action="store_true",
        help="Enable pin_memory on fine-tuning DataLoaders",
    )
    parser.add_argument(
        "--no-train-pin-memory",
        dest="train_pin_memory",
        action="store_false",
        help="Disable pin_memory on fine-tuning DataLoaders",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the full-training stage (only evaluate existing checkpoints)",
    )
    parser.set_defaults(train_pin_memory=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device and DataLoader helpers
# ---------------------------------------------------------------------------


def determine_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def prepare_dataloaders(config, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    clear_data_cache()
    train_loader, val_loader, test_loader = create_fixed_data_loaders(
        FixedDataConfig(),
        batch_size=batch_size or config.training.batch_size,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        prefetch_factor=config.system.prefetch_factor,
    )
    return train_loader, val_loader, test_loader


def prepare_training_loaders(
    config,
    finetune_opts: FineTuneOptions,
    low_r2_ids: Optional[List[str]] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    clear_data_cache()

    data_cfg = FixedDataConfig()
    if low_r2_ids:
        data_cfg.filter_station_ids = low_r2_ids

    train_loader, val_loader, _ = create_fixed_data_loaders(
        data_cfg,
        batch_size=finetune_opts.batch_size_override or config.training.batch_size,
        num_workers=num_workers if num_workers is not None else config.system.num_workers,
        pin_memory=pin_memory if pin_memory is not None else config.system.pin_memory,
        prefetch_factor=prefetch_factor if prefetch_factor is not None else config.system.prefetch_factor,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Gate override utilities for ablation
# ---------------------------------------------------------------------------


def _override_gate(gate, prefer_nn: bool) -> None:
    def forward_override(self, features, pbm_output, nn_output):  # type: ignore[override]
        if pbm_output.dim() == 1:
            pbm_out = pbm_output.unsqueeze(-1)
        else:
            pbm_out = pbm_output
        if nn_output.dim() == 1:
            nn_out = nn_output.unsqueeze(-1)
        else:
            nn_out = nn_output

        batch_size = features.size(0)
        device = features.device
        dtype = pbm_out.dtype

        gate_weights = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        if prefer_nn:
            mixed = nn_out
            gate_weights[:, 1] = 1.0
        else:
            mixed = pbm_out
        gate_weights[:, 0] = 0.0 if prefer_nn else 1.0

        gate_info = {
            "gate_weights": gate_weights,
            "pbm_weight": gate_weights[:, 0].mean().item(),
            "nn_weight": gate_weights[:, 1].mean().item(),
            "effective_gate": gate_weights,
        }
        return mixed, gate_info

    gate.forward = MethodType(forward_override, gate)


def configure_ablation(model, nn_module: str) -> None:
    if nn_module not in MODULE_NAMES:
        raise ValueError(f"Invalid ablation module: {nn_module}")

    for module_name in MODULE_NAMES:
        gate = getattr(model, f"{module_name}_gate", None)
        if gate is None:
            raise AttributeError(f"Model missing gate '{module_name}_gate'")
        _override_gate(gate, prefer_nn=(module_name == nn_module))


# ---------------------------------------------------------------------------
# Checkpoint loading and parameter freezing
# ---------------------------------------------------------------------------


def _load_checkpoint_flexible(model: nn.Module, path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        logger.info("⚠️ Checkpoint %s does not exist", path)
        return False

    try:
        state = torch.load(path, map_location="cpu")
    except Exception as exc:
        logger.warning("⚠️ Unable to read checkpoint %s: %s", path, exc)
        return False

    if isinstance(state, dict) and any(k in state for k in ("model_state_dict", "state_dict")):
        state = state.get("model_state_dict") or state.get("state_dict")

    if not isinstance(state, dict):
        logger.warning("⚠️ Checkpoint %s does not contain a valid state_dict", path)
        return False

    model_state = model.state_dict()
    adapted: Dict[str, torch.Tensor] = {}
    matched = 0
    prefixes = ("base.", "module.")

    for raw_key, value in state.items():
        candidates = [raw_key]
        for prefix in prefixes:
            if raw_key.startswith(prefix):
                candidates.append(raw_key[len(prefix):])
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                adapted[candidate] = value
                matched += 1
                break

    if not adapted:
        logger.warning("⚠️ Checkpoint %s has no matching parameters for the current model", path)
        return False

    model_state.update(adapted)
    model.load_state_dict(model_state)
    logger.info("✅ Loaded %d parameters from %s", matched, path)
    return True


def load_checkpoint(model, checkpoint: Path) -> bool:
    if _load_checkpoint_flexible(model, checkpoint):
        return True
    loaded = load_best_model_if_exists(model, path=str(checkpoint))
    return bool(loaded)


def freeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_module_branch(model: nn.Module, module_name: str, unfreeze_gate: bool) -> List[nn.Parameter]:
    nn_attr = f"{module_name}_nn_expert"
    gate_attr = f"{module_name}_gate"

    module_params: List[nn.Parameter] = []

    nn_module = getattr(model, nn_attr, None)
    if nn_module is None:
        raise AttributeError(f"Model missing attr '{nn_attr}'")
    for param in nn_module.parameters():
        param.requires_grad = True
        module_params.append(param)

    if unfreeze_gate:
        gate_module = getattr(model, gate_attr, None)
        if gate_module is None:
            raise AttributeError(f"Model missing attr '{gate_attr}'")
        for param in gate_module.parameters():
            param.requires_grad = True
            module_params.append(param)
    return module_params


# ---------------------------------------------------------------------------
# Fine-tuning routine
# ---------------------------------------------------------------------------


def run_finetuning(
    model: nn.Module,
    nn_module: str,
    device: torch.device,
    config,
    finetune_opts: FineTuneOptions,
    low_r2_ids: Optional[List[str]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
) -> Dict[str, float]:
    logger.info("🔧 Starting fine-tuning phase")

    freeze_all_parameters(model)
    trainable_params = unfreeze_module_branch(model, nn_module, finetune_opts.unfreeze_gate)

    if not trainable_params:
        logger.warning("No parameters were unfrozen for fine-tuning; skipping")
        return {}

    optimizer = AdamW(
        trainable_params,
        lr=finetune_opts.learning_rate,
        weight_decay=finetune_opts.weight_decay,
    )

    trainer = EnhancedTrainer(model, device=device.type)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history: Dict[str, float] = {}

    criterion = nn.MSELoss()

    for epoch in range(1, finetune_opts.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            stats = trainer.train_step(batch, criterion, optimizer)
            epoch_losses.append(stats["loss"])

        avg_train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch)
                predictions = outputs["runoff"]
                loss = criterion(predictions, batch["targets"])
                val_losses.append(float(loss.item()))

        avg_val_loss = float(sum(val_losses) / max(len(val_losses), 1)) if val_losses else float("inf")
        history[f"epoch_{epoch}_train_loss"] = avg_train_loss
        history[f"epoch_{epoch}_val_loss"] = avg_val_loss

        logger.info(
            "  Epoch %d/%d - train_loss=%.6f val_loss=%.6f",
            epoch,
            finetune_opts.epochs,
            avg_train_loss,
            avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if finetune_opts.checkpoint_every and epoch % finetune_opts.checkpoint_every == 0:
                ckpt_path = output_dir / f"finetune_epoch{epoch:02d}.pth"
                torch.save(best_state, ckpt_path)
                logger.info("    💾 Saved interim checkpoint: %s", ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= finetune_opts.patience:
                logger.info("    🛑 Early stopping triggered (patience=%d)", finetune_opts.patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("✅ Fine-tuning finished; best val loss %.6f", best_val_loss)

    # persist history for reference
    if history:
        history_path = output_dir / "finetune_history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info("  💾 Fine-tuning history saved to %s", history_path)

    return {"train_loss": avg_train_loss, "val_loss": best_val_loss}


# ---------------------------------------------------------------------------
# Variant execution
# ---------------------------------------------------------------------------


def run_variant(
    nn_module: str,
    output_prefix: str,
    description: str,
    config,
    checkpoint: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    finetune_opts: Optional[FineTuneOptions],
    low_r2_ids: Optional[List[str]],
    skip_train: bool,
) -> Dict[str, float]:
    print(f"\n=== Running ablation: {nn_module} ({description}) ===")
    model = create_hybrid_hydro_moe_model(config)

    checkpoint_loaded = False
    if checkpoint.exists():
        checkpoint_loaded = load_checkpoint(model, checkpoint)

    configure_ablation(model, nn_module)
    try:
        model = wrap_with_calibration(model)
    except Exception as exc:
        logger.warning("Calibration wrapper unavailable: %s", exc)

    model.to(device)

    variant_dir = OUTPUT_ROOT / output_prefix
    variant_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = variant_dir / "best_model.pth"

    original_env = os.environ.get("HYDROMOE_BEST_MODEL")
    os.environ["HYDROMOE_BEST_MODEL"] = str(best_model_path)

    if not skip_train:
        enhanced_training_loop(
            model,
            train_loader,
            val_loader,
            device,
            epochs=config.training.epochs,
            patience=config.training.early_stopping_patience,
        )
    elif not checkpoint_loaded:
        logger.warning("Skip training requested but no checkpoint was loaded; results may be random.")

    if best_model_path.exists():
        try:
            state = torch.load(best_model_path, map_location=device)
            model.load_state_dict(state)
        except Exception as exc:
            logger.warning("Failed to load best model from %s: %s", best_model_path, exc)

    if original_env is not None:
        os.environ["HYDROMOE_BEST_MODEL"] = original_env
    else:
        os.environ.pop("HYDROMOE_BEST_MODEL", None)

    finetune_summary: Dict[str, float] = {}
    if finetune_opts and finetune_opts.epochs > 0:
        finetune_summary = run_finetuning(
            model,
            nn_module,
            device,
            config,
            finetune_opts,
            low_r2_ids,
            train_loader,
            val_loader,
            variant_dir,
        )

    model.eval()
    metrics = evaluate_enhanced_model(
        model,
        test_loader,
        device,
        output_prefix=output_prefix,
    )

    out_dir = OUTPUT_ROOT / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  💾 metrics saved to {metrics_path}")

    if finetune_summary:
        summary_path = out_dir / "finetune_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(finetune_summary, f, ensure_ascii=False, indent=2)
        print(f"  💾 Fine-tuning summary saved to {summary_path}")

    return metrics


# ---------------------------------------------------------------------------
# Variant selection and low-R² utilities
# ---------------------------------------------------------------------------


def select_variants(requested: Iterable[str] | None) -> List[Variant]:
    if not requested:
        return list(ABLATION_VARIANTS)
    requested_set = {name.lower() for name in requested}
    selected = [v for v in ABLATION_VARIANTS if v[0] in requested_set]
    missing = requested_set - {v[0] for v in selected}
    if missing:
        raise ValueError(f"Unknown variant keys: {sorted(missing)}")
    return selected


def load_low_r2_ids(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        logger.warning("Low R² CSV %s not found; proceeding without filtering", csv_path)
        return []
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "station_id" not in df.columns:
        logger.warning("Low R² CSV missing 'station_id' column; ignoring")
        return []
    ids = df["station_id"].astype(str).tolist()
    logger.info("Loaded %d low-R² station IDs from %s", len(ids), csv_path)
    return ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    device = determine_device(args.device)
    config = get_default_config()
    train_loader, val_loader, test_loader = prepare_dataloaders(
        config, batch_size=args.batch_size
    )

    finetune_opts: Optional[FineTuneOptions] = None
    low_r2_ids: Optional[List[str]] = None

    if args.finetune:
        finetune_opts = FineTuneOptions(
            epochs=max(0, args.ft_epochs),
            learning_rate=args.ft_lr,
            weight_decay=args.ft_weight_decay,
            patience=max(1, args.ft_patience),
            unfreeze_gate=not args.ft_freeze_gate,
            batch_size_override=args.train_batch_size or args.batch_size,
            checkpoint_every=max(0, args.ft_checkpoint_every),
        )

        if args.ft_filter_low_r2:
            low_r2_ids = load_low_r2_ids(args.low_r2_csv)
            if not low_r2_ids:
                logger.warning("No low-R² stations found; fine-tuning will use all data")

        train_loader, val_loader = prepare_training_loaders(
            config,
            finetune_opts,
            low_r2_ids=low_r2_ids,
            num_workers=args.train_workers if args.train_workers is not None else None,
            pin_memory=args.train_pin_memory,
            prefetch_factor=args.train_prefetch,
        )

    variants = select_variants(args.variants)
    summary: Dict[str, Dict[str, float]] = {}

    for nn_module, prefix, desc in variants:
        metrics = run_variant(
            nn_module,
            prefix,
            desc,
            config,
            args.checkpoint,
            train_loader,
            val_loader,
            test_loader,
            device,
            finetune_opts,
            low_r2_ids,
            args.skip_train,
        )
        summary[nn_module] = {
            "R2": metrics.get("R2"),
            "KGE": metrics.get("KGE"),
            "RMSE": metrics.get("RMSE"),
        }

    print("\n=== Ablation Summary ===")
    for module_name, metrics in summary.items():
        r2 = metrics.get("R2")
        kge = metrics.get("KGE")
        rmse = metrics.get("RMSE")
        print(
            f"{module_name:>10}: "
            f"R2={(f'{r2:.4f}' if r2 is not None else 'NA')}  "
            f"KGE={(f'{kge:.4f}' if kge is not None else 'NA')}  "
            f"RMSE={(f'{rmse:.4f}' if rmse is not None else 'NA')}"
        )


if __name__ == "__main__":
    main()
