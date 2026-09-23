"""Run the full benchmark and write immutable CSV/JSON result artifacts."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from classical_models import build_models, fit_predict
from data_pipeline import DESCRIPTOR_NAMES, process_all
from graph_models import mol_to_graph, train_model, predict_model
from metrics import regression_metrics, classification_metrics, selective_curve
from splits import random_split, scaffold_split, butina_split


SEEDS = [20260711, 20260719, 20260727, 20260804, 20260812]
PRIMARY = {"ESOL": "regression", "BBBP": "classification"}


def stable_sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1 / (1 + np.exp(-x))


def load_features(root: Path, dataset: str):
    z = np.load(root / "data" / "processed" / f"{dataset}_features.npz")
    return {k: z[k] for k in z.files}


def fit_rule_transform(rule_train, y_train, task):
    if task == "classification":
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        lr.fit(rule_train.reshape(-1, 1), y_train.astype(int))
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return 1.0, 0.0


def transform_rule(rule, params, task):
    a, b = params
    value = a * rule + b
    return value  # regression prediction or classification logit


def applicability_scores(desc, train_desc, dataset):
    idx = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}
    cols = [idx[x] for x in (["MolLogP", "MolWt", "RotB", "AromaticProportion"] if dataset == "ESOL" else ["MolLogP", "TPSA", "MolWt", "FormalCharge"])]
    train_x = train_desc[:, cols]
    x = desc[:, cols]
    med = np.median(train_x, axis=0)
    q1, q3 = np.quantile(train_x, [0.25, 0.75], axis=0)
    scale = np.maximum(q3 - q1, 1e-3)
    z = (x - med) / scale
    domain = np.exp(-0.5 * np.mean(z**2, axis=1))
    formal = np.abs(desc[:, idx["FormalCharge"]])
    neutral = np.exp(-0.8 * formal)
    organic = 1.0 - 0.7 * np.clip(desc[:, idx["MetalFlag"]], 0, 1)
    return np.clip(domain * neutral * organic, 0.0, 1.0).astype(np.float32)


def fit_regression_std_scale(y_valid, means, stds):
    stds = np.clip(stds, 1e-4, None)
    def objective(log_s):
        s = np.exp(log_s)
        sig = s * stds
        return np.mean(0.5 * np.log(2 * np.pi * sig**2) + 0.5 * ((y_valid - means) / sig) ** 2)
    res = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def temperature_scale(y_valid, logits):
    def objective(log_t):
        t = np.exp(log_t)
        p = stable_sigmoid(logits / t)
        return -np.mean(y_valid * np.log(np.clip(p, 1e-7, 1)) + (1-y_valid)*np.log(np.clip(1-p, 1e-7, 1)))
    res = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def write_prediction_rows(rows, dataset, split, seed, model, indices, y, pred, std=None, gate=None, app=None, rule=None, partition="test"):
    for j, idx in enumerate(indices):
        rows.append({
            "dataset": dataset, "split": split, "seed": seed, "model": model,
            "partition": partition, "sample_index": int(idx), "y_true": float(y[j]),
            "prediction": float(pred[j]), "uncertainty": float(std[j]) if std is not None else np.nan,
            "rule_reliance": float(gate[j]) if gate is not None else np.nan,
            "correction_reliance": float(1.0-gate[j]) if gate is not None else np.nan,
            "gate": float(gate[j]) if gate is not None else np.nan,
            "applicability": float(app[j]) if app is not None else np.nan,
            "rule_prediction": float(rule[j]) if rule is not None else np.nan,
        })


def run_primary(root: Path, quick: bool = False):
    result_dir = root / "experiments" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows, split_rows, training_rows = [], [], [], []
    seeds = SEEDS[:2] if quick else SEEDS
    ensemble_members = 2 if quick else 3
    epochs = 35 if quick else 70

    for dataset, task in PRIMARY.items():
        data = load_features(root, dataset)
        desc, fps, y = data["descriptors"], data["fingerprints"], data["target"]
        smiles, raw_rule = data["smiles"], data["rule"]
        for split_name in ["random", "scaffold", "butina"]:
            for seed in seeds:
                if split_name == "random":
                    sp = random_split(y, task, seed)
                elif split_name == "scaffold":
                    sp = scaffold_split(smiles, seed)
                else:
                    sp = butina_split(smiles, seed, cutoff=0.6)
                split_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"n_train":len(sp.train),"n_valid":len(sp.valid),"n_test":len(sp.test)})

                scaler = StandardScaler().fit(desc[sp.train])
                desc_z = scaler.transform(desc).astype(np.float32)
                app = applicability_scores(desc, desc[sp.train], dataset)
                rule_params = fit_rule_transform(raw_rule[sp.train], y[sp.train], task)
                rule_transformed = transform_rule(raw_rule, rule_params, task).astype(np.float32)

                for model_name, model in build_models(task, seed).items():
                    x = fps if model_name == "RF_Morgan" else desc_z
                    t0 = time.time()
                    pred = fit_predict(model, x[sp.train], y[sp.train], x[sp.test], task)
                    elapsed = time.time() - t0
                    metrics = regression_metrics(y[sp.test], pred) if task == "regression" else classification_metrics(y[sp.test], pred)
                    metric_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":model_name,**metrics})
                    training_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":model_name,"seconds":elapsed,"best_epoch":np.nan})
                    write_prediction_rows(prediction_rows,dataset,split_name,seed,model_name,sp.test,y[sp.test],pred)

                fixed_pred = rule_transformed[sp.test] if task == "regression" else stable_sigmoid(rule_transformed[sp.test])
                metrics = regression_metrics(y[sp.test], fixed_pred) if task == "regression" else classification_metrics(y[sp.test], fixed_pred)
                metric_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":"Fixed_rule",**metrics})
                write_prediction_rows(prediction_rows,dataset,split_name,seed,"Fixed_rule",sp.test,y[sp.test],fixed_pred,app=app[sp.test],rule=fixed_pred)

                graphs = [mol_to_graph(str(smiles[i]), desc_z[i], float(y[i]), float(rule_transformed[i]), float(app[i]), i) for i in range(len(y))]
                train_graphs = [graphs[i] for i in sp.train]
                valid_graphs = [graphs[i] for i in sp.valid]
                test_graphs = [graphs[i] for i in sp.test]

                for mode, label, lam in [
                    ("gnn","GCN",0.0), ("fusion","Graph_descriptor_fusion",0.0),
                    ("residual","Rule_residual_no_gate",0.0), ("gated","Gated_no_applicability_penalty",0.0),
                ]:
                    t0=time.time()
                    tr=train_model(train_graphs,valid_graphs,mode=mode,task=task,seed=seed,gate_lambda=lam,epochs=epochs)
                    out,gate,_,order=predict_model(tr.model,test_graphs)
                    pred = out if task=="regression" else stable_sigmoid(out)
                    metrics=regression_metrics(y[order],pred) if task=="regression" else classification_metrics(y[order],pred)
                    metric_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":label,**metrics})
                    training_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":label,"seconds":time.time()-t0,"best_epoch":tr.best_epoch})
                    write_prediction_rows(prediction_rows,dataset,split_name,seed,label,order,y[order],pred,gate=gate,app=app[order],rule=(rule_transformed[order] if task=="regression" else stable_sigmoid(rule_transformed[order])))

                valid_member_out, test_member_out, test_member_gate = [], [], []
                valid_orders, test_orders = None, None
                member_epochs=[]
                t0=time.time()
                for member in range(ensemble_members):
                    member_seed=seed+1009*member
                    tr=train_model(train_graphs,valid_graphs,mode="proposed",task=task,seed=member_seed,gate_lambda=0.002,epochs=epochs)
                    vo,vg,_,vorder=predict_model(tr.model,valid_graphs)
                    to,tg,_,torder=predict_model(tr.model,test_graphs)
                    valid_member_out.append(vo); test_member_out.append(to); test_member_gate.append(tg)
                    valid_orders=vorder; test_orders=torder; member_epochs.append(tr.best_epoch)
                valid_member_out=np.vstack(valid_member_out)
                test_member_out=np.vstack(test_member_out)
                gate_mean=np.mean(np.vstack(test_member_gate),axis=0)
                if task=="regression":
                    valid_mean=valid_member_out.mean(axis=0)
                    valid_epi=valid_member_out.std(axis=0,ddof=1) if ensemble_members>1 else np.zeros_like(valid_mean)
                    aleatoric=float(np.sqrt(np.mean((y[valid_orders]-valid_mean)**2)))
                    valid_std=np.sqrt(valid_epi**2+aleatoric**2)
                    scale=fit_regression_std_scale(y[valid_orders],valid_mean,valid_std)
                    pred=test_member_out.mean(axis=0)
                    test_epi=test_member_out.std(axis=0,ddof=1) if ensemble_members>1 else np.zeros_like(pred)
                    std=np.maximum(np.sqrt(test_epi**2+aleatoric**2)*scale,1e-4)
                    metrics=regression_metrics(y[test_orders],pred,std)
                else:
                    valid_logits=valid_member_out.mean(axis=0)
                    temp=temperature_scale(y[valid_orders],valid_logits)
                    member_probs=stable_sigmoid(test_member_out/temp)
                    pred=member_probs.mean(axis=0); std=member_probs.std(axis=0,ddof=1) if ensemble_members>1 else np.zeros_like(pred)
                    metrics=classification_metrics(y[test_orders],pred,std)
                metric_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":"Proposed_AGRR",**metrics})
                training_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":"Proposed_AGRR","seconds":time.time()-t0,"best_epoch":float(np.mean(member_epochs))})
                write_prediction_rows(prediction_rows,dataset,split_name,seed,"Proposed_AGRR",test_orders,y[test_orders],pred,std=std,gate=gate_mean,app=app[test_orders],rule=(rule_transformed[test_orders] if task=="regression" else stable_sigmoid(rule_transformed[test_orders])))

                rng=np.random.default_rng(seed)
                shuffled=rule_transformed.copy()
                shuffled[sp.train]=rng.permutation(shuffled[sp.train])
                shuffled[sp.valid]=rng.permutation(shuffled[sp.valid])
                shuffled[sp.test]=rng.permutation(shuffled[sp.test])
                shuffled_graphs=[mol_to_graph(str(smiles[i]),desc_z[i],float(y[i]),float(shuffled[i]),float(app[i]),i) for i in range(len(y))]
                tr=train_model([shuffled_graphs[i] for i in sp.train],[shuffled_graphs[i] for i in sp.valid],mode="proposed",task=task,seed=seed+777,gate_lambda=0.002,epochs=epochs)
                out,gate,_,order=predict_model(tr.model,[shuffled_graphs[i] for i in sp.test])
                pred=out if task=="regression" else stable_sigmoid(out)
                metrics=regression_metrics(y[order],pred) if task=="regression" else classification_metrics(y[order],pred)
                metric_rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":"Shuffled_rule_control",**metrics})
                write_prediction_rows(prediction_rows,dataset,split_name,seed,"Shuffled_rule_control",order,y[order],pred,gate=gate,app=app[order])

                pd.DataFrame(metric_rows).to_csv(result_dir/"metrics_partial.csv",index=False)
                pd.DataFrame(prediction_rows).to_csv(result_dir/"predictions_partial.csv",index=False)
                print(f"completed {dataset} {split_name} seed={seed}", flush=True)

    pd.DataFrame(metric_rows).to_csv(result_dir/"metrics.csv",index=False)
    pd.DataFrame(prediction_rows).to_csv(result_dir/"predictions.csv",index=False)
    pd.DataFrame(split_rows).to_csv(result_dir/"splits.csv",index=False)
    pd.DataFrame(training_rows).to_csv(result_dir/"training_times.csv",index=False)


def external_b3db(root: Path):
    """Train on all cleaned BBBP except validation calibration fold and evaluate on de-overlapped B3DB transfer corpus."""
    result_dir=root/"experiments"/"results"
    if not (root/"data"/"processed"/"B3DB_external_features.npz").exists(): return
    train=load_features(root,"BBBP"); ext=load_features(root,"B3DB_external")
    overlap=set(map(str,train["smiles"]))
    keep=np.array([str(s) not in overlap for s in ext["smiles"]])
    ext={k:v[keep] for k,v in ext.items()}
    y=train["target"]; sp=random_split(y,"classification",SEEDS[0],ratios=(0.8,0.2,0.0)) if False else None
    rng=np.random.default_rng(SEEDS[0]); idx=np.arange(len(y)); rng.shuffle(idx); cut=int(0.85*len(idx)); tr_idx,va_idx=np.sort(idx[:cut]),np.sort(idx[cut:])
    scaler=StandardScaler().fit(train["descriptors"][tr_idx])
    d_train=scaler.transform(train["descriptors"]).astype(np.float32); d_ext=scaler.transform(ext["descriptors"]).astype(np.float32)
    params=fit_rule_transform(train["rule"][tr_idx],y[tr_idx],"classification")
    r_train=transform_rule(train["rule"],params,"classification").astype(np.float32); r_ext=transform_rule(ext["rule"],params,"classification").astype(np.float32)
    app_train=applicability_scores(train["descriptors"],train["descriptors"][tr_idx],"BBBP"); app_ext=applicability_scores(ext["descriptors"],train["descriptors"][tr_idx],"BBBP")
    graphs_train=[mol_to_graph(str(train["smiles"][i]),d_train[i],float(y[i]),float(r_train[i]),float(app_train[i]),i) for i in range(len(y))]
    graphs_ext=[mol_to_graph(str(ext["smiles"][i]),d_ext[i],float(ext["target"][i]),float(r_ext[i]),float(app_ext[i]),i) for i in range(len(ext["target"]))]
    members=[]; gates=[]; valid_logits=[]
    for m in range(3):
        tr=train_model([graphs_train[i] for i in tr_idx],[graphs_train[i] for i in va_idx],mode="proposed",task="classification",seed=SEEDS[0]+1009*m,gate_lambda=0.002,epochs=70)
        vo,_,_,_=predict_model(tr.model,[graphs_train[i] for i in va_idx]); to,tg,_,_=predict_model(tr.model,graphs_ext)
        valid_logits.append(vo); members.append(to); gates.append(tg)
    temp=temperature_scale(y[va_idx],np.vstack(valid_logits).mean(axis=0))
    probs=stable_sigmoid(np.vstack(members)/temp); pred=probs.mean(axis=0); std=probs.std(axis=0,ddof=1)
    m=classification_metrics(ext["target"],pred,std)
    pd.DataFrame([{ "dataset":"B3DB_external","split":"external","seed":SEEDS[0],"model":"Proposed_AGRR", **m, "n_external":len(pred), "overlap_removed":int((~keep).sum()) }]).to_csv(result_dir/"external_metrics.csv",index=False)
    pd.DataFrame({"sample_index":np.arange(len(pred)),"y_true":ext["target"],"prediction":pred,"uncertainty":std,"gate":np.mean(gates,axis=0),"applicability":app_ext,"smiles":ext["smiles"]}).to_csv(result_dir/"external_predictions.csv",index=False)


def sensitivity_controls(root: Path, quick=False):
    """Classical and fusion baselines on two no-rule regression endpoints."""
    rows=[]
    for dataset in ["FreeSolv","Lipophilicity"]:
        data=load_features(root,dataset); y=data["target"]; desc=data["descriptors"]; fps=data["fingerprints"]; smiles=data["smiles"]
        seeds=SEEDS[:2] if quick else SEEDS[:3]
        for split_name in ["random","scaffold"]:
            for seed in seeds:
                sp=random_split(y,"regression",seed) if split_name=="random" else scaffold_split(smiles,seed)
                scaler=StandardScaler().fit(desc[sp.train]); dz=scaler.transform(desc).astype(np.float32)
                for name,model in build_models("regression",seed).items():
                    if name not in {"Ridge_desc","RF_desc","XGBoost_desc","RF_Morgan"}: continue
                    x=fps if name=="RF_Morgan" else dz
                    pred=fit_predict(model,x[sp.train],y[sp.train],x[sp.test],"regression")
                    rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":name,**regression_metrics(y[sp.test],pred)})
                zero=np.zeros(len(y),dtype=np.float32); app=np.zeros(len(y),dtype=np.float32)
                graphs=[mol_to_graph(str(smiles[i]),dz[i],float(y[i]),0.0,0.0,i) for i in range(len(y))]
                tr=train_model([graphs[i] for i in sp.train],[graphs[i] for i in sp.valid],mode="fusion",task="regression",seed=seed,epochs=35 if quick else 60)
                out,_,_,order=predict_model(tr.model,[graphs[i] for i in sp.test])
                rows.append({"dataset":dataset,"split":split_name,"seed":seed,"model":"Graph_descriptor_fusion",**regression_metrics(y[order],out)})
    pd.DataFrame(rows).to_csv(root/"experiments"/"results"/"sensitivity_metrics.csv",index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    process_all(root)
    run_primary(root,quick=args.quick)
    sensitivity_controls(root,quick=args.quick)
    external_b3db(root)

if __name__=="__main__": main()
