# Historical standalone PBM archive (2025)

**Historical publication artifact — preserved for provenance, not recommended for new evaluation.**

These files reconstruct the standalone process-based HydroPy workflow used during the 2025 analysis. The code is preserved substantially as recovered so that the provenance of the published PBM benchmark can be audited.

## What this archive shows

The recovered workflow is a state-evolving HydroPy implementation. Daily simulation carries snow-water equivalent, liquid snow water, root-zone soil moisture, and groundwater storage. Basin-specific parameters are supplied from CMA-ES result files.

The two September parameter files have different roles in the recovered provenance:

- `artifacts/cmaes_optimal_params_2025-09-12.json` — original local filename `cmaes_optimal_params.json`; 550-basin, 17-parameter run associated with the historical standalone PBM evaluation chain that produced the reported basin-level PBM metrics.
- `artifacts/cmaes_optimal_params_2025-09-13.json` — original local filename `cmaes_optimal_params备份.json`; 550-basin, 17-parameter run numerically linked to the archived full-period precomputed PBM runoff used in later integration work.

The archive also includes the recovered 550-basin PBM metric table and the legacy plotting scripts that consume it.

## Retrospective audit findings

This directory is intentionally **not** presented as the corrected evaluation protocol. Retrospective audit identified methodological issues in the historical workflow, including:

1. A historical entry point explicitly optimized PBM parameters using observations from the nominal 2008+ test period.
2. Historical optimization/evaluation code contains observed-discharge lag selection/scanning (up to ±730 days in recovered versions).
3. Historical result metadata can report `includes_runoff_correction=True`, while the recovered optimizer's 17-element `essential_params` list does not actually include `runoff_correction`.
4. The recovered stateful HydroPy implementation has a groundwater-recharge accounting inconsistency under the total-runoff definition used in the archive; this is documented separately in `../../audit/`.

These issues are preserved here rather than silently edited. A leakage-free train-only recalibration belongs under `../../corrected/pbm_train_only/`.

## Reproducibility note

Some recovered source files contain historical machine-specific absolute paths. Those paths are part of the archived provenance and should not be interpreted as portable configuration. A corrected runnable workflow should accept explicit data paths instead of modifying the historical archive.

## File naming

Several files were renamed on upload only to make dates and roles explicit. Their original local names are documented above. File contents are otherwise preserved as recovered.
