# Corrected train-only PBM workflow

This directory is reserved for the leakage-free PBM recalibration/evaluation workflow.

Target protocol:

- calibration: 1980-01-01 to 1999-12-31
- validation: 2000-01-01 to 2007-12-31
- untouched test evaluation: 2008-01-01 to 2014-09-30
- no optimization against test observations
- no test-period lag selection
- no observation-side runoff correction

A preliminary audit run produced a lower PBM test NSE than the historically reported benchmark, but the final corrected files should be added only after an independent clean-room rerun from an archived source snapshot. This placeholder is intentionally not a claim that a finalized corrected result is already committed here.
