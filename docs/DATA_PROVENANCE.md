# Data Provenance

## Public third-party sources

The study analyses ESOL/Delaney solubility, BBBP, Lipophilicity, FreeSolv, and the external B3DB BBB-permeability corpus. These are third-party public scientific datasets and remain governed by their original source terms.

The repository records source locations, frozen-file checksums where available, and the code used for standardization and evaluation. It deliberately does not repackage the raw molecular tables.

## Frozen audit counts

The frozen dataset audit reports:

- ESOL: 1,128 raw rows and 1,117 standardized final rows.
- BBBP: 2,050 raw rows, 11 invalid records, 68 duplicate groups, 11 label-conflict groups, and 1,955 standardized final rows.
- Lipophilicity: 4,200 standardized final rows.
- FreeSolv: 642 standardized final rows.
- B3DB: 7,807 raw rows, 2 invalid records, 3 duplicate groups, and 7,802 standardized rows before external overlap filtering.
- External B3DB evaluation: 6,049 molecules after exact standardized-SMILES overlap removal against BBBP.

See `../data/manifests/dataset_audit.csv` for the machine-readable audit summary.

## Scientific-rule provenance caveat

The ESOL equation is historically tied to the Delaney/ESOL data source, so its use in the study is not characterized as a fully independent external scientific prior. The benchmark is retained to study conditional rule reliance and rule failure, with that historical dependence stated explicitly.

## External transfer caveat

B3DB filtering removes exact standardized-SMILES overlap with BBBP. This addresses direct identity overlap but does not eliminate broader scaffold, analogue, or chemical-space similarity.
