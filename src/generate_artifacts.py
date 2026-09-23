"""Generate manuscript tables, figures, and machine-readable summaries from frozen results."""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
RES=Path(os.environ.get('KBS_RESULTS_DIR', ROOT/'experiments'/'results'))
FIG=ROOT/'manuscript'/'figures'
TAB=ROOT/'manuscript'/'tables'
FIG.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)

PRIMARY_MODELS=[
    'Fixed_rule','RF_desc','ExtraTrees_desc','XGBoost_desc','RF_Morgan','GCN',
    'Graph_descriptor_fusion','Rule_residual_no_gate','Gated_no_applicability_penalty',
    'Proposed_AGRR','Shuffled_rule_control'
]
DISPLAY={
 'Fixed_rule':'Fixed rule','RF_desc':'RF descriptors','ExtraTrees_desc':'Extra Trees descriptors',
 'XGBoost_desc':'XGBoost descriptors','RF_Morgan':'RF Morgan','GCN':'GCN',
 'Graph_descriptor_fusion':'Graph--descriptor fusion','Rule_residual_no_gate':'Residual, no gate',
 'Gated_no_applicability_penalty':'Gated, no applicability penalty','Proposed_AGRR':'Proposed AGRR',
 'Shuffled_rule_control':'Shuffled-rule control','Ridge_desc':'Ridge descriptors','Logistic_desc':'Logistic descriptors'
}


def mean_sd(x):
    x=pd.Series(x).dropna().astype(float)
    return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}" if len(x)>1 else f"{x.mean():.3f}"


def bootstrap_diff(values, n=10000, seed=20260711):
    arr=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(arr,size=len(arr),replace=True).mean() for _ in range(n)])
    return float(arr.mean()), float(np.quantile(means,.025)), float(np.quantile(means,.975))


def make_architecture():
    """Create an uncluttered, publication-ready architecture diagram."""
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    boxes = [
        (0.025, 0.60, 0.14, 0.18, 'Molecular graph', 'atoms and bonds'),
        (0.025, 0.22, 0.14, 0.18, 'Descriptors', 'physicochemical'),
        (0.21, 0.60, 0.15, 0.18, 'Graph encoder', 'two-layer GCN'),
        (0.21, 0.22, 0.15, 0.18, 'Descriptor encoder', 'MLP'),
        (0.405, 0.41, 0.15, 0.20, 'Fused representation', r'$h=[h_g\Vert h_d]$'),
        (0.61, 0.57, 0.13, 0.19, 'Neural expert', r'$n(x)$'),
        (0.61, 0.16, 0.13, 0.22, 'Rule expert', r'$r(x)$; applicability $a(x)$'),
        (0.785, 0.35, 0.11, 0.24, 'Rule reliance', r'$t(x,a)$; $\rho=a t$'),
        (0.925, 0.35, 0.065, 0.24, 'Prediction', r'$n+\rho(r-n)$'),
    ]
    for x, y, w, h, title, sub in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.010', linewidth=1.35, facecolor='white', edgecolor='black')
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*0.63, title, ha='center', va='center', fontweight='bold', fontsize=9.4)
        ax.text(x + w/2, y + h*0.28, sub, ha='center', va='center', fontsize=8.5)
    arrows = [
        ((0.165, 0.69), (0.21, 0.69)), ((0.165, 0.31), (0.21, 0.31)),
        ((0.36, 0.69), (0.405, 0.54)), ((0.36, 0.31), (0.405, 0.48)),
        ((0.555, 0.53), (0.61, 0.665)), ((0.555, 0.47), (0.61, 0.27)),
        ((0.74, 0.665), (0.785, 0.52)), ((0.74, 0.27), (0.785, 0.42)),
        ((0.895, 0.47), (0.925, 0.47)),
    ]
    for source, target in arrows:
        ax.add_patch(FancyArrowPatch(source, target, arrowstyle='-|>', mutation_scale=12, linewidth=1.15))
    ax.text(0.405, 0.89, 'Training objective', fontweight='bold', fontsize=9.6)
    ax.text(0.405, 0.82, r'prediction loss $+\;\lambda_g a(x)[1-\rho(x)]$', fontsize=9.2)
    ax.text(0.025, 0.94, 'Applicability-gated rule-residual learning', fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=0.01, right=0.995, top=0.98, bottom=0.02)
    fig.savefig(FIG/'framework_architecture.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(FIG/'framework_architecture.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    metrics=pd.read_csv(RES/'metrics.csv')
    preds=pd.read_csv(RES/'predictions.csv')
    audit=pd.read_csv(ROOT/'data'/'processed'/'audit.csv')
    aud=audit[audit.dataset.isin(['ESOL','BBBP','Lipophilicity','FreeSolv','B3DB_external'])].copy()
    aud['role']=aud.dataset.map({'ESOL':'Primary rule task','BBBP':'Primary rule task','Lipophilicity':'No-rule sensitivity','FreeSolv':'No-rule sensitivity','B3DB_external':'De-overlapped BBB transfer validation'})
    aud[['dataset','raw_rows','missing_or_invalid','duplicate_groups','conflicting_groups','final_rows','task','role']].to_csv(TAB/'dataset_audit.csv',index=False)
    with open(TAB/'dataset_audit.tex','w') as f:
        f.write(aud[['dataset','raw_rows','missing_or_invalid','duplicate_groups','conflicting_groups','final_rows','task','role']].to_latex(index=False,escape=True,column_format='lrrrrrrl'))
    summaries={}
    for dataset,metric in [('ESOL','RMSE'),('BBBP','AUROC')]:
        sub=metrics[(metrics.dataset==dataset)&metrics.model.isin(PRIMARY_MODELS)].copy()
        agg=sub.groupby(['split','model'])[metric].agg(['mean','std']).reset_index()
        agg['display']=agg['model'].map(DISPLAY)
        agg.to_csv(TAB/f'{dataset.lower()}_{metric.lower()}_summary.csv',index=False)
        pivot=agg.pivot(index='display',columns='split',values='mean')
        order=[DISPLAY[m] for m in PRIMARY_MODELS if DISPLAY[m] in pivot.index]
        pivot=pivot.reindex(order)
        pivot.to_latex(TAB/f'{dataset.lower()}_{metric.lower()}_summary.tex',float_format='%.3f',escape=False)
        summaries[dataset]={}
        for split in ['random','scaffold','butina']:
            ss=sub[sub.split==split]
            for model in PRIMARY_MODELS:
                vals=ss[ss.model==model][metric]
                if len(vals): summaries[dataset].setdefault(split,{})[model]={'mean':float(vals.mean()),'sd':float(vals.std(ddof=1))}
    main_models=['Fixed_rule','ExtraTrees_desc','RF_Morgan','Graph_descriptor_fusion','Rule_residual_no_gate','Gated_no_applicability_penalty','Proposed_AGRR','Shuffled_rule_control']
    lines=[r"\begin{tabular}{llccc}",r"\toprule",r"Task & Model & Random & Scaffold & Butina \",r"\midrule"]
    for dataset,metric in [('ESOL','RMSE'),('BBBP','AUROC')]:
        sub=metrics[(metrics.dataset==dataset)&metrics.model.isin(main_models)]
        for j,model in enumerate(main_models):
            vals=[]
            for split in ['random','scaffold','butina']:
                x=sub[(sub.model==model)&(sub.split==split)][metric].dropna()
                vals.append(mean_sd(x) if len(x) else '--')
            task=dataset if j==0 else ''
            lines.append(f"{task} & {DISPLAY[model]} & {vals[0]} & {vals[1]} & {vals[2]} " + r"\\")
        if dataset=='ESOL': lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (TAB/'primary_performance.tex').write_text('\n'.join(lines))
    comparison=[]
    for dataset,metric,direction in [('ESOL','RMSE',-1),('BBBP','AUROC',1)]:
      for split in ['random','scaffold','butina']:
        ss=metrics[(metrics.dataset==dataset)&(metrics.split==split)]
        prop=ss[ss.model=='Proposed_AGRR'].set_index('seed')[metric]
        fusion=ss[ss.model=='Graph_descriptor_fusion'].set_index('seed')[metric]
        diff=(prop-fusion).dropna() if direction==1 else (fusion-prop).dropna()
        dmean,lo,hi=bootstrap_diff(diff)
        classical=ss[ss.model.isin(['RF_desc','ExtraTrees_desc','XGBoost_desc','RF_Morgan','Ridge_desc','Logistic_desc'])].groupby('model')[metric].mean()
        best_model=classical.idxmin() if direction==-1 else classical.idxmax()
        best=ss[ss.model==best_model].set_index('seed')[metric]
        diff2=(prop-best).dropna() if direction==1 else (best-prop).dropna()
        bmean,blo,bhi=bootstrap_diff(diff2)
        comparison.append({'dataset':dataset,'split':split,'metric':metric,'proposed_minus_fusion_benefit':dmean,'ci_low':lo,'ci_high':hi,'best_classical':best_model,'proposed_minus_best_classical_benefit':bmean,'best_ci_low':blo,'best_ci_high':bhi})
    pd.DataFrame(comparison).to_csv(TAB/'paired_comparisons.csv',index=False)
    cols=['dataset','split','model','NLL','Coverage95','Width95','Brier','ECE','UncertaintyErrorSpearman']
    cal=metrics[metrics.model=='Proposed_AGRR'][[c for c in cols if c in metrics.columns]].groupby(['dataset','split','model']).agg(['mean','std'])
    cal.to_csv(TAB/'uncertainty_summary.csv')
    diag=[]; qrows=[]
    for dataset in ['ESOL','BBBP']:
      for split in ['random','scaffold','butina']:
        p=preds[(preds.dataset==dataset)&(preds.split==split)&(preds.model=='Proposed_AGRR')].dropna(subset=['gate','applicability'])
        p['abs_error']=np.abs(p.y_true-p.prediction)
        diag.append({'dataset':dataset,'split':split,'n':len(p),'rule_reliance_mean':p.gate.mean(),'correction_reliance_mean':1-p.gate.mean(),'applicability_mean':p.applicability.mean(),'reliance_applicability_spearman':spearmanr(p.gate,p.applicability).statistic,'reliance_error_spearman':spearmanr(p.gate,p.abs_error).statistic})
        p['quartile']=pd.qcut(p.applicability,4,labels=['Q1 low','Q2','Q3','Q4 high'],duplicates='drop')
        fusion=preds[(preds.dataset==dataset)&(preds.split==split)&(preds.model=='Graph_descriptor_fusion')][['seed','sample_index','prediction']].rename(columns={'prediction':'fusion_pred'})
        p=p.merge(fusion,on=['seed','sample_index'],how='left')
        p['proposed_error']=(p.y_true-p.prediction).abs(); p['fusion_error']=(p.y_true-p.fusion_pred).abs()
        for q,g in p.groupby('quartile',observed=True):
          qrows.append({'dataset':dataset,'split':split,'quartile':str(q),'n':len(g),'applicability':g.applicability.mean(),'rule_reliance':g.gate.mean(),'proposed_abs_error':g.proposed_error.mean(),'fusion_abs_error':g.fusion_error.mean(),'error_benefit':g.fusion_error.mean()-g.proposed_error.mean()})
    pd.DataFrame(diag).to_csv(TAB/'gate_diagnostics.csv',index=False)
    pd.DataFrame(qrows).to_csv(TAB/'applicability_quartiles.csv',index=False)
    external={}
    if (RES/'external_metrics.csv').exists():
        ext=pd.read_csv(RES/'external_metrics.csv'); ext.to_csv(TAB/'external_b3db_results.csv',index=False)
        external=ext.to_dict(orient='records')
    if (RES/'sensitivity_metrics.csv').exists():
        sens=pd.read_csv(RES/'sensitivity_metrics.csv')
        sens.groupby(['dataset','split','model'])['RMSE'].agg(['mean','std']).reset_index().to_csv(TAB/'no_rule_sensitivity.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,(dataset,metric) in zip(axes,[('ESOL','RMSE'),('BBBP','AUROC')]):
        sub=metrics[(metrics.dataset==dataset)&metrics.model.isin(['Fixed_rule','ExtraTrees_desc','RF_Morgan','Graph_descriptor_fusion','Proposed_AGRR'])]
        agg=sub.groupby(['split','model'])[metric].agg(['mean','std']).reset_index()
        x=np.arange(3); width=.15
        models=['Fixed_rule','ExtraTrees_desc','RF_Morgan','Graph_descriptor_fusion','Proposed_AGRR']
        for j,m in enumerate(models):
            g=agg[agg.model==m].set_index('split').reindex(['random','scaffold','butina'])
            ax.bar(x+(j-2)*width,g['mean'],width,yerr=g['std'],capsize=2,label=DISPLAY[m])
        ax.set_xticks(x,['Random','Scaffold','Butina']); ax.set_ylabel(metric); ax.set_title(dataset); ax.grid(axis='y',alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1)); fig.savefig(FIG/'predictive_performance.pdf',bbox_inches='tight'); fig.savefig(FIG/'predictive_performance.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    p=preds[(preds.dataset=='BBBP')&(preds.model=='Proposed_AGRR')]
    bins=np.linspace(0,1,11); xs=[]; ys=[]; ns=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(p.prediction>=lo)&(p.prediction<(hi if hi<1 else hi+1e-9))
        if m.any(): xs.append(p.loc[m,'prediction'].mean()); ys.append(p.loc[m,'y_true'].mean()); ns.append(m.sum())
    fig,ax=plt.subplots(figsize=(5.2,4.6)); ax.plot([0,1],[0,1],'--'); ax.plot(xs,ys,marker='o')
    for x,y,n in zip(xs,ys,ns): ax.annotate(str(n),(x,y),textcoords='offset points',xytext=(4,4),fontsize=7)
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed positive fraction'); ax.set_title('BBBP reliability (all held-out folds)'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/'bbbp_reliability.pdf'); fig.savefig(FIG/'bbbp_reliability.png',dpi=300); plt.close(fig)
    coverage=np.linspace(.2,1,20)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,(dataset,task) in zip(axes,[('ESOL','regression'),('BBBP','classification')]):
      curves=[]
      pp=preds[(preds.dataset==dataset)&(preds.model=='Proposed_AGRR')].dropna(subset=['uncertainty'])
      for (_,_,_),g in pp.groupby(['split','seed','model']):
        order=np.argsort(g.uncertainty.to_numpy()); risks=[]
        for c in coverage:
          take=order[:max(2,int(round(c*len(g))))]
          if task=='regression': risks.append(np.sqrt(np.mean((g.y_true.to_numpy()[take]-g.prediction.to_numpy()[take])**2)))
          else: risks.append(np.mean((g.prediction.to_numpy()[take]>=.5).astype(int)!=g.y_true.to_numpy()[take]))
        curves.append(risks)
      arr=np.asarray(curves); ax.plot(coverage,arr.mean(axis=0)); ax.fill_between(coverage,arr.mean(axis=0)-arr.std(axis=0),arr.mean(axis=0)+arr.std(axis=0),alpha=.2)
      ax.set_xlabel('Coverage retained'); ax.set_ylabel('RMSE' if task=='regression' else 'Classification error'); ax.set_title(dataset); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/'selective_risk.pdf'); fig.savefig(FIG/'selective_risk.png',dpi=300); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,dataset in zip(axes,['ESOL','BBBP']):
      p=preds[(preds.dataset==dataset)&(preds.model=='Proposed_AGRR')].dropna(subset=['gate','applicability'])
      ax.hexbin(p.applicability,p.gate,gridsize=25,mincnt=1); ax.set_xlabel('Applicability score'); ax.set_ylabel(r'Rule reliance $\rho$'); ax.set_title(dataset); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG/'rule_reliance_applicability.pdf'); fig.savefig(FIG/'rule_reliance_applicability.png',dpi=300); plt.close(fig)
    make_architecture()
    with open(RES/'result_summary.json','w') as f: json.dump({'main':summaries,'paired':comparison,'external':external},f,indent=2)

if __name__=='__main__': main()
