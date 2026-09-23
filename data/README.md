# Data handling

This repository does **not** redistribute the third-party raw molecular benchmark files used in the study. The exact expected filenames, source citations, retrieval locations, and SHA-256 hashes of the copies used for the frozen study are documented in `sources.md` and `checksums.csv`.

After obtaining the source files under their original terms, place them under `data/raw/` with these filenames:

- `delaney-processed.csv`
- `BBBP.csv`
- `Lipophilicity.csv`
- `FreeSolv_database.txt`

The B3DB transfer corpus is loaded through `qc-B3DB==1.1.1` by the frozen pipeline. `src/data_pipeline.py` performs deterministic molecular cleanup, largest-fragment selection, uncharging where possible, canonicalization, duplicate handling, feature generation, and rule-score generation.

`manifests/dataset_audit.csv` records the frozen dataset audit counts. Raw and per-molecule processed files are intentionally excluded from this public repository pending their independent source terms; this repository instead preserves the methods, aggregate frozen outputs, and source-file checksums used for manuscript traceability.
