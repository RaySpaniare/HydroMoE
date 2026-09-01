# HydroMoE reproducibility and provenance materials

This directory separates recovered historical publication artifacts from corrected/re-audited workflows.

## Directory roles

- `publication_archive/standalone_pbm_2025/`: recovered historical standalone HydroPy/CMA-ES workflow and associated September 2025 artifacts. These files are preserved for provenance and are **not** recommended as a leakage-free evaluation protocol.
- `corrected/pbm_train_only/`: reserved for the clean, train-only PBM recalibration/evaluation workflow after independent reproduction from an archived code snapshot.
- `audit/`: provenance manifests, reconstruction scripts, and quantitative sensitivity/audit outputs.

## Important distinction

The historical standalone PBM workflow, the publication-time HydroMoE PBM integration pathway, and the later realtime PBM wrapper in the original public repository are distinct code paths. They must not be treated as interchangeable.

The current public-release code is intentionally not overwritten by this archive. Historical and corrected materials are kept side-by-side so that changes remain auditable.
