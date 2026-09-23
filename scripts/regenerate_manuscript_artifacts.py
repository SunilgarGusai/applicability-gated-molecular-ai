"""Public helper for manuscript artifact regeneration.

The scientific scripts are preserved under ``src/``. The raw public datasets
must first be retrieved from ``data/sources.md`` because this repository does
not redistribute them.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ROOT / "src" / "generate_artifacts.py",
    ROOT / "src" / "generate_review_figures.py",
    ROOT / "src" / "generate_case_figure.py",
]

def main() -> int:
    parser = argparse.ArgumentParser(
        description="List or execute the frozen manuscript-artifact generation scripts."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the scripts in order; requires the public datasets at the paths expected by the pipeline",
    )
    args = parser.parse_args()
    print("Artifact-generation scripts:")
    for step in STEPS:
        print(f"  - {step.relative_to(ROOT)}")
    if not args.execute:
        print("No scripts executed. Use --execute only after retrieving the source datasets documented in data/sources.md.")
        return 0
    for step in STEPS:
        print(f"Running {step.relative_to(ROOT)}")
        subprocess.run([sys.executable, str(step)], cwd=ROOT, check=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
