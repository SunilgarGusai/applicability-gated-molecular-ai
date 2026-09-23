from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {20260711, 20260719, 20260727, 20260804, 20260812}
SPLITS = {"random", "scaffold", "butina"}

esol = pd.read_csv(ROOT / "results/frozen/main_metrics/esol_rmse_summary.csv")
bbbp = pd.read_csv(ROOT / "results/frozen/main_metrics/bbbp_auroc_summary.csv")
sizes = pd.read_csv(ROOT / "results/frozen/statistical_outputs/split_sizes.csv")
ext = pd.read_csv(ROOT / "results/frozen/external_validation/external_b3db_results.csv")
fail = pd.read_csv(ROOT / "results/frozen/statistical_outputs/failure_diagnostics_summary.csv")
unc = pd.read_csv(ROOT / "results/frozen/uncertainty/explanation_stability.csv")
paired = pd.read_csv(ROOT / "results/frozen/statistical_outputs/paired_comparisons.csv")

for df in (esol, bbbp):
    assert set(df["split"]) == SPLITS
    assert "Proposed_AGRR" in set(df["model"])
    assert "Shuffled_rule_control" in set(df["model"])

assert set(sizes["seed"].astype(int)) == SEEDS
assert set(sizes["split"]) == SPLITS
assert set(sizes["dataset"]) == {"ESOL", "BBBP"}

esol_scaf = esol[(esol.split == "scaffold") & (esol.model == "Proposed_AGRR")].iloc[0]
assert math.isclose(float(esol_scaf["mean"]), 0.7372314252936619, rel_tol=0, abs_tol=1e-12)
assert math.isclose(float(esol_scaf["std"]), 0.05377755297082071, rel_tol=0, abs_tol=1e-12)

agrr_ext = ext[ext.model == "Proposed_AGRR"].iloc[0]
extra_ext = ext[ext.model == "ExtraTrees_desc"].iloc[0]
assert int(agrr_ext.n_external) == 6049
assert int(agrr_ext.overlap_removed) == 1753
assert float(extra_ext.AUROC) > float(agrr_ext.AUROC)
assert float(agrr_ext.ECE) < float(extra_ext.ECE)

assert len(fail) == 6 and set(fail.dataset) == {"ESOL", "BBBP"}
assert len(unc) == 2 and set(unc.dataset) == {"ESOL", "BBBP"}
assert len(paired) == 6

print("Frozen public aggregate package: PASS")
print("Seeds:", ", ".join(map(str, sorted(SEEDS))))
print("ESOL scaffold AGRR RMSE: %.6f +/- %.6f" % (esol_scaf['mean'], esol_scaf['std']))
print("B3DB de-overlapped n:", int(agrr_ext.n_external))
print("B3DB Extra Trees AUROC: %.6f" % extra_ext.AUROC)
print("B3DB AGRR AUROC/ECE: %.6f / %.6f" % (agrr_ext.AUROC, agrr_ext.ECE))
