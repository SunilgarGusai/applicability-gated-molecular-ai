# Reproducibility

## Scope of the public repository

This repository exposes the research pipeline, frozen experiment configuration, provenance records, aggregate machine-readable outputs, and manuscript table/figure support scripts. Raw and per-molecule third-party molecular data are not redistributed.

The authoritative frozen submission archive contains additional author-side audit records, including per-molecule predictions and processed molecular tables, that were used for manuscript production and independent numerical checks. Those records are intentionally omitted from the public repository to avoid republishing third-party molecular records without a source-by-source redistribution determination.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate agrr-molecular-ai
```

or install the listed Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## 2. Retrieve the public data

Follow [`../data/sources.md`](../data/sources.md). Place the ESOL, BBBP, Lipophilicity and FreeSolv source files under `data/raw/` with the filenames stated in [`../data/README.md`](../data/README.md). B3DB is loaded through the pinned `qc-B3DB` package.

When the exact historical source snapshot remains accessible, compare SHA-256 values to [`../data/checksums.csv`](../data/checksums.csv).

## 3. Inspect the frozen protocol

Read [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) and [`../config/experiment.yaml`](../config/experiment.yaml). The source modules under `../src/` implement molecular standardization and features, random/scaffold/Butina splits, scientific-rule handling, classical and graph models, AGRR, calibration/uncertainty, external transfer, attribution stability and independent rule-failure diagnosis.

## 4. Validate the publicly exposed frozen outputs

```bash
python scripts/validate_frozen_package.py
```

This validator uses only public repository files. It checks dataset/split/seed coverage and verifies several manuscript-critical frozen aggregate values, including the ESOL scaffold AGRR result and the de-overlapped B3DB comparison.

## 5. Rerun the research modules

After retrieving the source data, use a disposable clone because research runs write generated files into the working tree. With `PYTHONPATH=src` where required, the authoritative source package contains these real entry points:

```bash
python src/run_experiments.py
python src/external_comparators.py
python src/explanation_analysis.py
python src/rule_failure_diagnostics.py
```

The first command performs preprocessing and the primary benchmark. The subsequent commands reproduce the external-comparator, descriptor-attribution-stability and independent rule-failure analyses used in the frozen study.

## 6. Tables and figures

Aggregate manuscript-supporting outputs are under `../results/frozen/`; table-ready copies are under `../tables/` where provided. Figure-generation source is retained under `../src/`.

```bash
python scripts/regenerate_manuscript_artifacts.py --help
```

Running figure-generation scripts after a clean retraining requires the corresponding generated prediction/diagnostic files. The repository does not claim that every publication figure can be regenerated from aggregate CSVs alone.

## 7. Exact-replay caveat

The frozen archive is sufficient to audit the reported results, but it does not preserve every object needed for byte-for-byte replay of the historical training run, such as every trained checkpoint, per-member prediction and original run-state object. The repository therefore distinguishes **result traceability/auditability** from exact computational replay and does not advertise an unsupported one-command byte-identical retraining claim.

A technically competent reader can inspect the implemented pipeline and frozen configuration, retrieve the public source data, and rerun the modules. Exact numerical identity across environments is not guaranteed where original run-state objects are absent.
