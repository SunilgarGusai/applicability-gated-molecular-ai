# Reproducibility

## Scope of the public repository

This repository exposes the research pipeline, frozen experiment configuration, provenance records, aggregate machine-readable outputs, and manuscript table/figure support scripts. Raw and per-molecule third-party molecular data are not redistributed.

The authoritative frozen submission archive contained additional local files (including per-molecule predictions and processed molecular tables) used for audit and manuscript production. Those records are not necessary to inspect the aggregate claims exposed here and are omitted to avoid republishing third-party molecular records without a source-by-source redistribution determination.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate applicability-gated-molecular-ai
```

or install the listed Python dependencies through `requirements.txt` in a compatible environment.

## 2. Retrieve the public data

Follow [`../data/sources.md`](../data/sources.md). Place source files into the locations expected by [`../config/experiment.yaml`](../config/experiment.yaml) if you intend to rerun the research pipeline.

When the exact historical source snapshot remains accessible, compare SHA-256 values to [`../data/checksums.csv`](../data/checksums.csv).

## 3. Inspect the frozen protocol

Read [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md). The source modules under `../src/` implement dataset standardization, molecular features, split construction, scientific-rule experts, learned models, calibration/uncertainty, diagnostics, and evaluation.

## 4. Validate the publicly exposed frozen outputs

```bash
python scripts/validate_frozen_package.py
```

The validator checks the frozen primary metric table for the expected 360 dataset/split/model/seed rows and verifies complete ESOL/BBBP coverage for five seeds over random, scaffold, and Butina split families.

## 5. Tables and figures

Machine-readable manuscript table sources are in `../tables/`. Aggregate manuscript-supporting output files are in `../results/frozen/`.

```bash
python scripts/regenerate_manuscript_artifacts.py --help
```

The public helper reports/manages the aggregate inputs needed for manuscript-artifact regeneration. The publication-ready figure files in the private/frozen submission archive were generated from these research outputs; third-party per-molecule molecular records are not required for inspection of the aggregate values published here.

## 6. Clean retraining caveat

The authoritative frozen archive is sufficient to audit the reported results, but it does not contain every object that would be required to byte-for-byte reproduce the original training run (for example, not every trained checkpoint/per-member prediction or original train/validation manifest is present). The repository therefore distinguishes **result reproducibility/auditability** from exact computational replay and does not advertise an unsupported one-command exact retraining claim.

A technically competent reader can inspect the implemented pipeline and frozen configuration, retrieve the public source data, and rerun the modules. Exact agreement with the original frozen predictions is not guaranteed across environments or where original run-state objects are absent.
