from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
metrics = pd.read_csv(ROOT / "results/frozen/main_metrics/metrics.csv")
summary = json.loads((ROOT / "results/frozen/main_metrics/result_summary.json").read_text())
expected_seeds = {20260711, 20260719, 20260727, 20260804, 20260812}

assert len(metrics) == 360, f"Expected 360 primary metric rows, found {len(metrics)}"
assert {"ESOL", "BBBP"} <= set(metrics["dataset"])
assert {"random", "scaffold", "butina"} <= set(metrics["split"])
assert set(metrics["seed"].astype(int)) == expected_seeds, "Frozen seed set does not match the authoritative archive"
for dataset in ("ESOL", "BBBP"):
    sub = metrics[metrics["dataset"] == dataset]
    assert set(sub["split"]) == {"random", "scaffold", "butina"}
    assert set(sub["seed"].astype(int)) == expected_seeds

print("Frozen aggregate results are present and structurally consistent.")
print("Primary metric rows:", len(metrics))
print("Frozen seeds:", ", ".join(map(str, sorted(expected_seeds))))
print("Summary keys:", ", ".join(sorted(summary.keys())[:12]))
