# Dataset Sources and Retrieval

The frozen study used public third-party molecular datasets. This repository records where they came from and how the frozen source files were identified; it does not redistribute the raw molecular tables.

| Dataset | Frozen source filename | Public source used/identified for retrieval | Frozen SHA-256 |
|---|---|---|---|
| ESOL / Delaney | `delaney-processed.csv` | DeepChem public data: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv` | `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189` |
| BBBP | `BBBP.csv` | DeepChem public data: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv` | `d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662` |
| Lipophilicity | `Lipophilicity.csv` | DeepChem public data: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv` | `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd` |
| FreeSolv | `FreeSolv_database.txt` | MobleyLab FreeSolv repository, `database.txt`: `https://github.com/MobleyLab/FreeSolv` | `2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260` |
| B3DB | package/source snapshot used by the frozen project | B3DB / qc-B3DB public release; retrieve from the original B3DB source or the versioned `qc-B3DB` distribution described in the manuscript | See the project audit/provenance records for the versioned external snapshot |

## Standardization and audit

Standardization, invalid-record handling, duplicate resolution, label-conflict handling, and split construction are implemented in the source modules under [`../src/`](../src/). Frozen audit counts are in [`manifests/dataset_audit.csv`](manifests/dataset_audit.csv).

The ESOL/Delaney scientific equation has a historical relationship with the source data used to create the benchmark. The manuscript therefore does not describe it as a fully independent external prior.

The B3DB transfer set was standardized and then filtered by exact standardized-SMILES overlap against BBBP; 6,049 molecules remained for the frozen external evaluation. Exact-identity removal should not be interpreted as eliminating broader chemical similarity.
