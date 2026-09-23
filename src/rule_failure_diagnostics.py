"""Train-only rule-failure diagnostics for the final review package.

The diagnostic model never changes the AGRR prediction. It estimates the
absolute error of the fixed rule from training molecules and is evaluated on
held-out molecules under the same frozen split protocol.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from splits import random_split, scaffold_key, butina_groups, _assign_groups

SEEDS = [20260711, 20260719, 20260727, 20260804, 20260812]
TASKS = {"ESOL": "regression", "BBBP": "classification"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def get_split(y, task, split, seed, cached_groups):
    if split == "random":
        return random_split(y, task, seed)
    return _assign_groups(cached_groups[split], len(y), seed)


def transform_rule(raw, y, train, task):
    if task == "regression":
        return raw.astype(float)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    lr.fit(raw[train].reshape(-1, 1), y[train].astype(int))
    return sigmoid(lr.decision_function(raw.reshape(-1, 1)))


def applicability(desc, train_desc, dataset):
    names = [
        "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotB", "RingCount",
        "AromaticRings", "FractionCSP3", "HeavyAtomCount", "FormalCharge",
        "HeteroAtoms", "NHOHCount", "NOCount", "LabuteASA", "BalabanJ",
        "BertzCT", "MolMR", "AromaticProportion", "MetalFlag",
    ]
    idx = {n:i for i,n in enumerate(names)}
    keys = ["MolLogP","MolWt","RotB","AromaticProportion"] if dataset == "ESOL" else ["MolLogP","TPSA","MolWt","FormalCharge"]
    cols = [idx[k] for k in keys]
    med = np.median(train_desc[:,cols],axis=0)
    q1,q3 = np.quantile(train_desc[:,cols],[0.25,0.75],axis=0)
    scale=np.maximum(q3-q1,1e-3)
    z=(desc[:,cols]-med)/scale
    domain=np.exp(-0.5*np.mean(z*z,axis=1))
    neutral=np.exp(-0.8*np.abs(desc[:,idx["FormalCharge"]]))
    organic=1.0-0.7*np.clip(desc[:,idx["MetalFlag"]],0,1)
    return np.clip(domain*neutral*organic,0,1)


def fp_sketch(fps, bins=64):
    return fps.reshape(len(fps), bins, -1).sum(axis=2) / (fps.shape[1]//bins)


def selective_aurc(error, score):
    order=np.argsort(score)
    coverages=np.linspace(0.1,1.0,19)
    risks=[]
    for c in coverages:
        n=max(2,int(round(c*len(error))))
        risks.append(float(np.mean(error[order[:n]])))
    return float(np.trapezoid(risks,coverages)/(coverages[-1]-coverages[0])), coverages, np.asarray(risks)


def main(root: Path):
    rows=[]; preds=[]; curves=[]
    for dataset,task in TASKS.items():
        z=np.load(root/f"data/processed/{dataset}_features.npz")
        desc=z["descriptors"].astype(float); fps=z["fingerprints"].astype(float)
        y=z["target"].astype(float); smiles=z["smiles"]; raw=z["rule"].astype(float)
        sketch=fp_sketch(fps)
        scaffold_buckets={}
        for i,s in enumerate(smiles): scaffold_buckets.setdefault(scaffold_key(str(s)),[]).append(i)
        cached_groups={"scaffold":list(scaffold_buckets.values()),"butina":butina_groups(smiles,cutoff=0.6)}
        for split in ["random","scaffold","butina"]:
            for seed in SEEDS:
                sp=get_split(y,task,split,seed,cached_groups)
                rp=transform_rule(raw,y,sp.train,task)
                err=np.abs(y-rp)
                app=applicability(desc,desc[sp.train],dataset)
                scaler=StandardScaler().fit(desc[sp.train])
                dz=scaler.transform(desc)
                nn=NearestNeighbors(n_neighbors=1,metric="euclidean").fit(dz[sp.train])
                distance=nn.kneighbors(dz[sp.test],return_distance=True)[0][:,0]
                X=dz
                model=ExtraTreesRegressor(n_estimators=40,min_samples_leaf=5,max_depth=12,max_features=0.8,n_jobs=1,random_state=seed)
                model.fit(X[sp.train],err[sp.train])
                q=np.maximum(0,model.predict(X[sp.test]))
                e=err[sp.test]
                corr_q=float(spearmanr(q,e).statistic)
                corr_app=float(spearmanr(1-app[sp.test],e).statistic)
                corr_dist=float(spearmanr(distance,e).statistic)
                aurc_q,cov,risk_q=selective_aurc(e,q)
                aurc_app,_,risk_app=selective_aurc(e,1-app[sp.test])
                aurc_dist,_,risk_dist=selective_aurc(e,distance)
                overall=float(np.mean(e))
                n50=max(2,int(round(0.5*len(e))))
                order=np.argsort(q)
                risk50=float(np.mean(e[order[:n50]]))
                rows.append(dict(dataset=dataset,split=split,seed=seed,n_test=len(sp.test),
                    failure_score_spearman=corr_q,declared_applicability_spearman=corr_app,
                    descriptor_distance_spearman=corr_dist,failure_score_aurc=aurc_q,
                    applicability_aurc=aurc_app,distance_aurc=aurc_dist,
                    full_rule_error=overall,rule_error_at_50pct=risk50,
                    relative_reduction_at_50pct=(overall-risk50)/overall if overall else np.nan))
                for j,idx in enumerate(sp.test):
                    preds.append(dict(dataset=dataset,split=split,seed=seed,sample_index=int(idx),
                        canonical_smiles=str(smiles[idx]),y_true=float(y[idx]),rule_prediction=float(rp[idx]),
                        rule_absolute_error=float(err[idx]),failure_score=float(q[j]),
                        declared_applicability=float(app[idx]),descriptor_distance=float(distance[j])))
                for method,risks in [("Failure score",risk_q),("Declared applicability",risk_app),("Descriptor distance",risk_dist)]:
                    for c,r in zip(cov,risks):
                        curves.append(dict(dataset=dataset,split=split,seed=seed,method=method,coverage=float(c),rule_risk=float(r)))
    out=root/"experiments/results"
    pd.DataFrame(rows).to_csv(out/"failure_diagnostics.csv",index=False)
    pd.DataFrame(preds).to_csv(out/"failure_predictions.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"failure_selective_curves.csv",index=False)
    summary=pd.DataFrame(rows).groupby(["dataset","split"]).agg(
        failure_score_spearman=("failure_score_spearman","mean"),
        failure_score_sd=("failure_score_spearman","std"),
        applicability_spearman=("declared_applicability_spearman","mean"),
        distance_spearman=("descriptor_distance_spearman","mean"),
        failure_score_aurc=("failure_score_aurc","mean"),
        applicability_aurc=("applicability_aurc","mean"),
        distance_aurc=("distance_aurc","mean"),
        relative_reduction_at_50pct=("relative_reduction_at_50pct","mean"),
    ).reset_index()
    summary.to_csv(out/"failure_diagnostics_summary.csv",index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
