# Audit materials

This directory contains recovered provenance and reconstruction materials. These files are provided to make the historical investigation inspectable; they are not a replacement for the publication archive or a corrected reproduction package.

Included in this upload package:

- `HYDROMOE_CRITICAL_FILE_MANIFEST.csv`: inventory of critical checkpoints/predictions and their recorded hashes/roles.
- `station_sensitivity.csv`: basin-level sensitivity results from historical/zero/repaired PBM comparisons where available.
- `recovered_four_module_summary.json`: summary of recovered four-module stateful HydroPy outputs and water-balance diagnostics.
- `recover_four_module_physical_outputs.py`: reconstruction script used for the archived stateful PBM audit.

Large historical files (for example full daily prediction/runoff CSVs) should be distributed as release assets or in a durable data repository rather than silently rewritten or committed over the original public code.
