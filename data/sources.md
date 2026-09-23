# Dataset sources and provenance

| Dataset | Frozen source used | Retrieval / project page | Primary citation | Role |
|---|---|---|---|---|
| ESOL | DeepChem-hosted copy of Delaney supporting data | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv | Delaney (2004), https://doi.org/10.1021/ci034243x | Primary regression |
| BBBP | DeepChem/MoleculeNet BBBP CSV | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv | Martins et al. (2012), https://doi.org/10.1021/ci300124c | Primary classification |
| Lipophilicity | MoleculeNet/ChEMBL AstraZeneca file | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv | Wu et al. (2018), https://doi.org/10.1039/C7SC02664A | No-rule sensitivity |
| FreeSolv | Official MobleyLab `database.txt`, v0.52 | https://github.com/MobleyLab/FreeSolv/blob/master/database.txt | Mobley & Guthrie (2014), https://doi.org/10.1007/s10822-014-9747-x | No-rule sensitivity |
| B3DB transfer | `qc-B3DB` classification corpus v1.1.1 | https://pypi.org/project/qc-B3DB/ | Meng et al. (2021), https://doi.org/10.1038/s41597-021-01069-5 | External BBB transfer |

The frozen study deliberately used the official MobleyLab FreeSolv file rather than the DeepChem target copy. The B3DB project declares its distributed dataset under CC0; the other datasets retain their original source terms and citation requirements.

The manuscript also discloses an important provenance limitation: the ESOL rule is historically related to the benchmark source data, so the fixed rule is not described as a fully independent external prior.
