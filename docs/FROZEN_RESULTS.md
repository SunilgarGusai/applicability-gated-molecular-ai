# Frozen Result Inventory

The files under `results/frozen/` are aggregate machine-readable outputs copied or adapted directly from the authoritative frozen research archive. They are intended to make the manuscript-supporting numerical evidence inspectable without publishing third-party per-molecule molecular records.

## Main metrics

`main_metrics/metrics.csv` contains the primary dataset/split/model/seed metric table. The authoritative package audit records **360 rows** and confirms recomputation agreement against the frozen prediction records to within approximately `1.82e-7`.

Dataset-specific JSON summaries for ESOL and BBBP are included alongside the main metric table.

## Uncertainty and calibration

`uncertainty/` contains aggregate ensemble/calibration/selective-risk summaries used for the reliability analysis.

## Ablations and controls

`ablations/` contains the available aggregate ablation/control summaries, including shuffled-rule and related sensitivity analyses where present in the frozen package.

## External validation

`external_validation/` contains aggregate B3DB transfer metrics. The de-overlapped evaluation corpus contains **6,049 molecules** after exact standardized-SMILES overlap removal against BBBP.

## Statistical/audit outputs

`statistical_outputs/` contains available paired-comparison, failure-diagnosis, split-size, and evidence-recomputation summaries.

## Selected manuscript-level findings

These values are quoted only because they are explicitly supported by the frozen package:

- ESOL scaffold RMSE: AGRR `0.737 ± 0.054`; fixed Delaney rule `1.121 ± 0.035`; single-network unrestricted fusion `0.804 ± 0.093`.
- BBBP AUROC for AGRR is reported as `0.920`, `0.913`, and `0.885` on random, scaffold, and Butina splits, respectively, while shuffled-rule and strong baseline results constrain any knowledge-specific interpretation.
- External B3DB: Extra Trees has the highest observed AUROC (`0.907`), while AGRR has the lowest observed ECE (`0.047`) among the compared frozen configurations.
- The independent failure score correlates with held-out rule error at `0.409–0.592` for ESOL and `0.720–0.799` for BBBP; retaining the half with the lowest predicted rule error reduces observed mean rule error by `22–38%`.

These findings should be read with the ensemble-comparison and ESOL historical-overlap caveats described in the manuscript.
