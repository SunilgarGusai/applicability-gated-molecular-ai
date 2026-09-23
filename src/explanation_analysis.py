"""Representative descriptor-pathway attribution stability analysis.

The analysis deliberately avoids claiming atom-level mechanistic explanation. It quantifies
whether descriptor sensitivities are stable across independently initialized ensemble members.
"""
from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from data_pipeline import DESCRIPTOR_NAMES
from run_experiments import load_features, applicability_scores, fit_rule_transform, transform_rule, SEEDS
from graph_models import mol_to_graph, train_model
from splits import scaffold_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'results'


def attribution(model, graph):
    batch=next(iter(DataLoader([graph],batch_size=1,shuffle=False)))
    batch.desc=batch.desc.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out,gate,_=model(batch)
    out.sum().backward()
    grad=batch.desc.grad.detach().cpu().numpy().reshape(-1)
    x=batch.desc.detach().cpu().numpy().reshape(-1)
    return grad*x, grad, float(out.item()), float(gate.item())


def main():
  rows=[]; summary=[]
  for dataset,task in [('ESOL','regression'),('BBBP','classification')]:
    data=load_features(ROOT,dataset); y=data['target']; desc=data['descriptors']; smiles=data['smiles']; rule=data['rule']
    sp=scaffold_split(smiles,SEEDS[0])
    scaler=StandardScaler().fit(desc[sp.train]); dz=scaler.transform(desc).astype(np.float32)
    app=applicability_scores(desc,desc[sp.train],dataset)
    params=fit_rule_transform(rule[sp.train],y[sp.train],task); rt=transform_rule(rule,params,task).astype(np.float32)
    graphs=[mol_to_graph(str(smiles[i]),dz[i],float(y[i]),float(rt[i]),float(app[i]),i) for i in range(len(y))]
    models=[]
    for m in range(3):
      tr=train_model([graphs[i] for i in sp.train],[graphs[i] for i in sp.valid],mode='proposed',task=task,seed=SEEDS[0]+1009*m,gate_lambda=.002,epochs=70)
      models.append(tr.model)
    rng=np.random.default_rng(SEEDS[0]); sample=rng.choice(sp.test,size=min(120,len(sp.test)),replace=False)
    sample_stab=[]; sample_jacc=[]
    for idx in sample:
      attrs=[]
      for mi,model in enumerate(models):
        a,g,o,gate=attribution(model,graphs[int(idx)]); attrs.append(a)
        for j,name in enumerate(DESCRIPTOR_NAMES):
          rows.append({'dataset':dataset,'sample_index':int(idx),'member':mi,'descriptor':name,'attribution':float(a[j]),'gradient':float(g[j]),'output':o,'gate':gate,'applicability':float(app[idx])})
      for a,b in itertools.combinations(attrs,2):
        sample_stab.append(float(spearmanr(np.abs(a),np.abs(b)).statistic))
        ta=set(np.argsort(np.abs(a))[-5:]); tb=set(np.argsort(np.abs(b))[-5:]); sample_jacc.append(len(ta&tb)/len(ta|tb))
    summary.append({'dataset':dataset,'split':'scaffold','seed':SEEDS[0],'n_molecules':len(sample),'pairwise_abs_attribution_spearman':float(np.nanmean(sample_stab)),'top5_jaccard':float(np.mean(sample_jacc))})
  pd.DataFrame(rows).to_csv(OUT/'descriptor_attributions.csv',index=False)
  pd.DataFrame(summary).to_csv(OUT/'explanation_stability.csv',index=False)

if __name__=='__main__': main()
