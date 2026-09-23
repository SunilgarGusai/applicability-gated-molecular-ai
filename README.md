# Applicability-Gated Molecular AI

Reproducibility materials for **Learning When to Trust Fallible Scientific Rules: Applicability-Gated Molecular Property Prediction under Chemical-Space Shift**.

## Overview

This repository accompanies a study of **Applicability-Gated Rule-Residual Learning (AGRR)**, a knowledge-guided artificial-intelligence framework in which an explicit scientific rule is retained as a fallible expert rather than treated as a universal law. A declared applicability signal and a learned trust term determine molecule-specific rule reliance; a graph/descriptor expert supplies learned correction. Ensemble uncertainty and a separately trained rule-failure score support auditing and selective prediction.

The molecular-property setting is used as a demanding test bed for a broader trustworthy-AI question: **when should a learned system trust, correct, review, or abstain from a scientific rule under distribution shift?**

The frozen manuscript results deliberately include mixed and negative findings. In particular, the principal AGRR configuration is a three-member ensemble while several comparator/control configurations are single networks, so observed performance differences do **not** isolate a causal effect of gating. The Delaney ESOL equation also has historical overlap with the ESOL source data. On BBBP, strong non-rule baselines and shuffled-rule controls remain competitive. These limitations are part of the reported scientific conclusion.

## Method

For a molecule \(x\), the study retains a scientific rule prediction \(r(x)\), combines declared applicability \(a(x)\) with learned trust \(t(x)\), and forms rule reliance \(\rho(x)\). A neural molecular expert \(n_\theta(x)\) then contributes the complementary prediction:

\[
\hat y=(1-\rho)n_\theta(x)+\rho r(x).
\]

The repository also contains the separate failure-diagnosis and uncertainty-analysis code used to audit rule error, calibration, and selective prediction. Failure diagnosis is an audit pathway; it does not alter the frozen AGRR prediction.

## Datasets

The study uses public third-party molecular benchmark data:

- ESOL / Delaney solubility
- BBBP / blood-brain-barrier permeability
- Lipophilicity and FreeSolv as no-rule/sensitivity tasks
- B3DB for external BBB transfer after exact standardized-SMILES overlap removal against the internal BBBP benchmark

Raw third-party molecular datasets are **not redistributed in this repository**. Exact source locations, retrieval notes, and SHA-256 checksums for the files used in the frozen study are provided in [`data/sources.md`](data/sources.md) and [`data/checksums.csv`](data/checksums.csv).

## Experimental Design

The frozen study evaluates random, Bemis-Murcko scaffold, and Butina similarity-cluster splits over five seeds. The repository preserves the code used for standardization, split construction, baselines, AGRR, calibration, uncertainty, diagnostics, and external evaluation. Aggregate frozen outputs used for manuscript tables/figures are under [`results/frozen/`](results/frozen/).

Key safeguards include audit-before-splitting, train-only preprocessing/rule fitting, validation-only calibration/model selection where applicable, and untouched test/external labels during fitting and tuning. See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Repository Structure

```text
config/                     frozen experiment configuration
data/                       source manifests, checksums and audit summaries
  manifests/                dataset/reference audit manifests
docs/                       protocol, provenance and reproduction notes
src/                        research pipeline source code
scripts/                    frozen-package validation and artifact regeneration
results/frozen/             aggregate machine-readable manuscript outputs
tables/                     machine-readable table sources
```

The repository intentionally does not contain the submitted manuscript itself or unrestricted copies of third-party raw datasets.

## Installation

The frozen environment specification is provided in [`environment.yml`](environment.yml), with a pip-oriented dependency list in [`requirements.txt`](requirements.txt).

Conda example:

```bash
conda env create -f environment.yml
conda activate applicability-gated-molecular-ai
```

## Reproducing the Main Experiments

This public repository is designed primarily to expose the frozen research pipeline and manuscript-supporting outputs without redistributing third-party molecular records. After retrieving the public datasets from the documented sources into the paths expected by the configuration, the research modules under `src/` can be used with [`config/experiment.yaml`](config/experiment.yaml).

Because the final archive does not contain a single authoritative one-command clean-retraining wrapper, this README does **not** advertise a fabricated `run_all` command. The exact frozen protocol and module responsibilities are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Reproducing Tables

Validate the aggregate frozen outputs and dataset/split coverage:

```bash
python scripts/validate_frozen_package.py
```

The manuscript table sources are also provided in [`tables/`](tables/).

## Reproducing Figures

The publication figures were produced from frozen machine-readable outputs. The artifact-regeneration helper is:

```bash
python scripts/regenerate_manuscript_artifacts.py --help
```

The helper maps the public aggregate files to manuscript-oriented summaries and reports which figure-generation inputs are available without exposing third-party per-molecule records. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the supported public-repository workflow.

## External B3DB Evaluation

The external transfer analysis uses B3DB after standardization and removal of molecules whose standardized SMILES exactly overlap the internal BBBP benchmark. The de-overlapped evaluation set contains **6,049 molecules** in the frozen study. Aggregate external metrics are in [`results/frozen/external_validation/`](results/frozen/external_validation/); data provenance and retrieval instructions are in [`data/sources.md`](data/sources.md).

The exact-SMILES de-overlap does not prove absence of all chemical similarity between datasets and should not be interpreted as such.

## Frozen Results

A compact inventory of manuscript-supporting frozen outputs is provided in [`docs/FROZEN_RESULTS.md`](docs/FROZEN_RESULTS.md). The public repository contains aggregate metrics, uncertainty/calibration summaries, ablation/control summaries, external-validation metrics, and statistical/audit summaries. Per-molecule third-party molecular records are intentionally excluded from the public repository.

## Data Provenance

See:

- [`data/sources.md`](data/sources.md)
- [`data/checksums.csv`](data/checksums.csv)
- [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)
- [`data/manifests/dataset_audit.csv`](data/manifests/dataset_audit.csv)

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License / Usage Notes

The research archive identifies the study code as intended for MIT release, but a repository-wide software license has **not** been asserted here because ownership/provenance of every contributed source component has not been independently re-verified for public relicensing. See [`docs/LICENSE_AND_USAGE.md`](docs/LICENSE_AND_USAGE.md). Third-party datasets remain subject to their original source terms and are not redistributed here.
