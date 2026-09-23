from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import Draw

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'manuscript/figures'; FIG.mkdir(parents=True,exist_ok=True)
COL={'navy':'#17365D','blue':'#3C78D8','teal':'#2A9D8F','gold':'#E9C46A','orange':'#F4A261','red':'#D55E00','gray':'#6B7280','light':'#F4F7FB','green':'#4E9F3D','purple':'#7E57C2'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=0.03)
    fig.savefig(FIG/f'{name}.png',dpi=360,bbox_inches='tight',pad_inches=0.03)
    plt.close(fig)

def box(ax,xy,w,h,title,sub='',fc='white',ec=COL['navy'],lw=1.2,fontsize=9):
    x,y=xy
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.018',facecolor=fc,edgecolor=ec,linewidth=lw)
    ax.add_patch(p)
    ax.text(x+w/2,y+h*0.64,title,ha='center',va='center',weight='bold',fontsize=fontsize,color=COL['navy'])
    if sub: ax.text(x+w/2,y+h*0.30,sub,ha='center',va='center',fontsize=fontsize-1,color='#374151')
    return p

def arrow(ax,a,b,color=COL['navy'],style='-',lw=1.3,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,linewidth=lw,color=color,linestyle=style,connectionstyle=f'arc3,rad={rad}'))

fig,ax=plt.subplots(figsize=(12.8,6.4)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.text(0.02,0.985,'Applicability-gated rule-residual learning with independent failure diagnosis',weight='bold',fontsize=14,color=COL['navy'],va='top')
ax.text(0.02,0.925,'Solid arrows: inference pathway    Dashed arrows: training-only supervision or audit pathway',fontsize=8.5,color=COL['gray'])
for y,h,label,c in [(0.54,0.32,'A. Learned molecular expert',COL['blue']),(0.28,0.22,'B. Explicit fallible-rule pathway',COL['orange']),(0.035,0.19,'C. Reliability and decision layer',COL['teal'])]:
    ax.add_patch(FancyBboxPatch((0.012,y),0.976,h,boxstyle='round,pad=0.006',facecolor=c+'12',edgecolor=c,linewidth=0.8))
    ax.text(0.022,y+h+0.010,label,weight='bold',fontsize=9,color=c,va='top')
box(ax,(0.035,0.58),0.125,0.10,'Molecular graph','atoms, bonds',fc='white'); box(ax,(0.035,0.72),0.125,0.10,'Descriptors','physicochemical',fc='white')
box(ax,(0.205,0.58),0.13,0.10,'Graph encoder',r'$f_g(G)$',fc='#EAF2FD'); box(ax,(0.205,0.72),0.13,0.10,'Descriptor encoder',r'$f_d(d)$',fc='#EAF2FD')
box(ax,(0.385,0.64),0.14,0.12,'Fused representation',r'$h=[h_g\Vert h_d]$',fc='#EAF2FD'); box(ax,(0.575,0.64),0.13,0.12,'Neural expert',r'$n_\theta(x)$',fc='#EAF2FD')
arrow(ax,(0.16,0.63),(0.205,0.63)); arrow(ax,(0.16,0.77),(0.205,0.77)); arrow(ax,(0.335,0.63),(0.385,0.68)); arrow(ax,(0.335,0.77),(0.385,0.72)); arrow(ax,(0.525,0.70),(0.575,0.70))
box(ax,(0.035,0.325),0.155,0.11,'Scientific rule expert','$r(d)$; equation retained',fc='#FFF4E6',ec=COL['orange']); box(ax,(0.235,0.325),0.155,0.11,'Declared applicability',r'$a(x)\in[0,1]$',fc='#FFF4E6',ec=COL['orange']); box(ax,(0.435,0.325),0.155,0.11,'Learned trust',r'$t_\phi(h,a)$',fc='#FFF4E6',ec=COL['orange']); box(ax,(0.635,0.325),0.135,0.11,'Rule reliance',r'$\rho=a\,t$',fc='#FFF4E6',ec=COL['orange'])
arrow(ax,(0.19,0.38),(0.635,0.38),color=COL['orange']); arrow(ax,(0.39,0.38),(0.435,0.38),color=COL['orange']); arrow(ax,(0.59,0.38),(0.635,0.38),color=COL['orange']); arrow(ax,(0.455,0.64),(0.505,0.435),color=COL['orange'],rad=0.1)
box(ax,(0.815,0.59),0.16,0.18,'Auditable prediction',r'$\hat y=(1-\rho)n+\rho r$'+'\n'+r'rule contribution $\rho(r-n)$',fc='#F0F7EF',ec=COL['green'],fontsize=10)
arrow(ax,(0.705,0.70),(0.815,0.70),color=COL['green'],lw=1.7); arrow(ax,(0.77,0.38),(0.87,0.59),color=COL['green'],lw=1.7)
box(ax,(0.035,0.075),0.16,0.09,'Training rule error','$|y-r|$',fc='#EAF8F5',ec=COL['teal']); box(ax,(0.235,0.075),0.17,0.09,'Failure-score model',r'$q_\psi(x)\approx E[|y-r|\mid x]$',fc='#EAF8F5',ec=COL['teal']); box(ax,(0.445,0.075),0.14,0.09,'Ensemble uncertainty','$u(x)$',fc='#EAF8F5',ec=COL['teal']); box(ax,(0.63,0.065),0.15,0.11,'Audit record',r'$r,a,t,\rho,q,u,\hat y$',fc='#EAF8F5',ec=COL['teal']); box(ax,(0.825,0.055),0.15,0.13,'Decision support','trust  |  correct\nreview / abstain',fc='#EAF8F5',ec=COL['teal'])
arrow(ax,(0.195,0.12),(0.235,0.12),color=COL['teal'],style='--'); arrow(ax,(0.405,0.12),(0.63,0.12),color=COL['teal']); arrow(ax,(0.585,0.12),(0.63,0.12),color=COL['teal']); arrow(ax,(0.78,0.12),(0.825,0.12),color=COL['teal'],lw=1.7); arrow(ax,(0.455,0.64),(0.32,0.165),color=COL['teal'],style='--',rad=0.18); arrow(ax,(0.895,0.59),(0.705,0.175),color=COL['teal'],style='--',rad=-0.15)
ax.text(0.60,0.515,'Prediction objective',fontsize=8,weight='bold',color=COL['gray']); ax.text(0.60,0.487,'task loss + weak high-applicability retention penalty',fontsize=7.6,color=COL['gray']); save(fig,'framework_architecture_enhanced')

fig,ax=plt.subplots(figsize=(12.5,4.5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.text(0.02,0.95,'Leakage-controlled experimental and reproducibility workflow',fontsize=14,weight='bold',color=COL['navy'],va='top')
steps=[('Public sources','ESOL · BBBP\nFreeSolv · Lipophilicity\nB3DB transfer'),('Molecular audit','parse · clean · fragment\ncanonicalize · de-duplicate\nremove label conflicts'),('Frozen partitions','5 seeds × random\nBemis–Murcko scaffold\nButina clusters'),('Fit on train/validation','scaling · rule calibration\nmodels · early stopping\nuncertainty calibration'),('Untouched evaluation','prediction · calibration\nselective risk · negative controls\nB3DB de-overlapped transfer'),('Immutable outputs','split manifests · predictions\nmetrics · figures · DOI ledger\nchecksums · environment')]
xs=np.linspace(0.02,0.84,len(steps)); w=0.145; y=0.39; h=0.34
for i,(t,s) in enumerate(steps):
    fc=['#EEF4FC','#FFF5E8','#F3EEFB','#EAF8F5','#FDEEEE','#F3F4F6'][i]; ec=[COL['blue'],COL['orange'],COL['purple'],COL['teal'],COL['red'],COL['gray']][i]
    box(ax,(xs[i],y),w,h,t,s,fc=fc,ec=ec,fontsize=9); ax.text(xs[i]+0.012,y+h-0.045,str(i+1),fontsize=13,weight='bold',color=ec)
    if i<len(steps)-1: arrow(ax,(xs[i]+w,y+h/2),(xs[i+1],y+h/2),color=COL['navy'],lw=1.4)
ax.add_patch(FancyBboxPatch((0.20,0.10),0.60,0.16,boxstyle='round,pad=0.012',facecolor='white',edgecolor=COL['navy'],linewidth=1.1)); ax.text(0.50,0.205,'Leakage controls',weight='bold',ha='center',color=COL['navy'],fontsize=10); ax.text(0.50,0.145,'all duplicate handling precedes splitting  ·  preprocessing and rule transforms use train only  ·  calibration uses validation only  ·  test and B3DB labels never tune the model',ha='center',va='center',fontsize=8.2,color='#374151',wrap=True); save(fig,'experimental_workflow')

pc=pd.read_csv(ROOT/'manuscript/tables/paired_comparisons.csv'); fig,axes=plt.subplots(1,2,figsize=(11.8,4.4),sharey=False)
for ax,dataset,metric in zip(axes,['ESOL','BBBP'],['RMSE benefit','AUROC benefit']):
    d=pc[pc.dataset==dataset].copy(); d['label']=d['split'].str.capitalize(); y=np.arange(len(d))[::-1]; ax.axvline(0,color='black',lw=0.8)
    ax.errorbar(d.proposed_minus_fusion_benefit,y,xerr=[d.proposed_minus_fusion_benefit-d.ci_low,d.ci_high-d.proposed_minus_fusion_benefit],fmt='o',capsize=3,label='AGRR vs fusion',color=COL['blue']); ax.errorbar(d.proposed_minus_best_classical_benefit,y-0.18,xerr=[d.proposed_minus_best_classical_benefit-d.best_ci_low,d.best_ci_high-d.proposed_minus_best_classical_benefit],fmt='s',capsize=3,label='AGRR vs best classical',color=COL['orange']); ax.set_yticks(y-0.09,d.label); ax.set_xlabel(metric); ax.set_title(dataset); ax.grid(axis='x',alpha=.25)
axes[0].legend(loc='lower right',frameon=False); fig.suptitle('Paired seed-level benefits with 95% bootstrap intervals',fontsize=13,weight='bold',color=COL['navy']); fig.tight_layout(rect=[0,0,1,.93]); save(fig,'paired_effect_sizes')

fd=pd.read_csv(ROOT/'experiments/results/failure_diagnostics_summary.csv'); fig,axes=plt.subplots(1,2,figsize=(12,4.8)); methods=[('failure_score_spearman','Learned failure score',COL['blue']),('applicability_spearman','Declared applicability',COL['orange']),('distance_spearman','Descriptor distance',COL['gray'])]
for ax,dataset in zip(axes,['ESOL','BBBP']):
    d=fd[fd.dataset==dataset].set_index('split').loc[['random','scaffold','butina']]; x=np.arange(3); width=.24
    for k,(col,label,c) in enumerate(methods): ax.bar(x+(k-1)*width,d[col],width,label=label,color=c,alpha=.9)
    ax.axhline(0,color='black',lw=.7); ax.set_xticks(x,['Random','Scaffold','Butina']); ax.set_ylim(-.35,1.0); ax.set_ylabel('Spearman correlation with held-out rule error'); ax.set_title(dataset); ax.grid(axis='y',alpha=.2)
axes[0].legend(frameon=False,loc='upper right'); fig.suptitle('A separately trained failure score ranks rule error more reliably than declared applicability',fontsize=12.5,weight='bold',color=COL['navy']); fig.tight_layout(rect=[0,0,1,.92]); save(fig,'failure_score_diagnostics')

cur=pd.read_csv(ROOT/'experiments/results/failure_selective_curves.csv'); fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,dataset in zip(axes,['ESOL','BBBP']):
    d=cur[cur.dataset==dataset].groupby(['method','coverage'],as_index=False).rule_risk.mean()
    for method,c in [('Failure score',COL['blue']),('Declared applicability',COL['orange']),('Descriptor distance',COL['gray'])]:
        z=d[d.method==method]; ax.plot(z.coverage,z.rule_risk,marker='o',ms=3,label=method,color=c,lw=1.8)
    ax.set_xlabel('Coverage retained'); ax.set_ylabel('Mean absolute rule error'); ax.set_title(dataset); ax.grid(alpha=.25)
axes[0].legend(frameon=False); fig.suptitle('Selective use of the scientific rule after sorting by predicted failure',fontsize=12.5,weight='bold',color=COL['navy']); fig.tight_layout(rect=[0,0,1,.92]); save(fig,'failure_selective_risk')

ext=pd.read_csv(ROOT/'manuscript/tables/external_b3db_results.csv'); labels={'Fixed_rule':'Fixed rule','ExtraTrees_desc':'Extra Trees','RF_Morgan':'RF Morgan','Graph_descriptor_fusion':'Fusion','Proposed_AGRR':'AGRR'}; ext['label']=ext.model.map(labels); fig,axes=plt.subplots(1,2,figsize=(11.8,4.5)); colors=[COL['gray'],COL['orange'],COL['purple'],COL['teal'],COL['blue']]; x=np.arange(len(ext))
axes[0].bar(x-0.18,ext.AUROC,.36,label='AUROC',color=colors,edgecolor='white'); axes[0].bar(x+0.18,ext.AUPRC,.36,label='AUPRC',color=colors,alpha=.55,edgecolor='white',hatch='//'); axes[0].set_ylim(.72,.96); axes[0].set_xticks(x,ext.label,rotation=25,ha='right'); axes[0].set_title('Transfer discrimination'); axes[0].legend(frameon=False); axes[0].grid(axis='y',alpha=.2)
axes[1].bar(x-0.18,ext.Brier,.36,label='Brier (lower)',color=colors,edgecolor='white'); axes[1].bar(x+0.18,ext.ECE,.36,label='ECE (lower)',color=colors,alpha=.55,edgecolor='white',hatch='//'); axes[1].set_ylim(0,.21); axes[1].set_xticks(x,ext.label,rotation=25,ha='right'); axes[1].set_title('Transfer calibration'); axes[1].legend(frameon=False); axes[1].grid(axis='y',alpha=.2); fig.suptitle('De-overlapped B3DB transfer: classical ensembles lead discrimination; AGRR has lowest ECE',fontsize=12.5,weight='bold',color=COL['navy']); fig.tight_layout(rect=[0,0,1,.92]); save(fig,'external_b3db_comparison')

pred=pd.read_csv(ROOT/'experiments/results/predictions.csv'); fail=pd.read_csv(ROOT/'experiments/results/failure_predictions.csv'); proc={d:pd.read_csv(ROOT/f'data/processed/{d}.csv') for d in ['ESOL','BBBP']}; fig,axes=plt.subplots(2,4,figsize=(13,6.5))
for row,dataset in enumerate(['ESOL','BBBP']):
    p=pred[(pred.dataset==dataset)&(pred.split=='scaffold')&(pred.seed==20260711)&(pred.model=='Proposed_AGRR')].copy(); f=fail[(fail.dataset==dataset)&(fail.split=='scaffold')&(fail.seed==20260711)].copy(); z=p.merge(f[['sample_index','failure_score','rule_absolute_error']],on='sample_index'); z['model_error']=abs(z.y_true-z.prediction); z['improvement']=z.rule_absolute_error-z.model_error; trust=z.sort_values(['failure_score','rule_reliance'],ascending=[True,False]).head(2); correct=z.sort_values(['improvement','rule_absolute_error'],ascending=False).head(2); sel=pd.concat([trust,correct]).drop_duplicates('sample_index').head(4)
    while len(sel)<4: sel=pd.concat([sel,z.sort_values('failure_score',ascending=False)]).drop_duplicates('sample_index').head(4)
    for col,(_,r) in enumerate(sel.iterrows()):
        ax=axes[row,col]; ax.axis('off'); smi=proc[dataset].iloc[int(r.sample_index)].canonical_smiles; mol=Chem.MolFromSmiles(smi); img=Draw.MolToImage(mol,size=(320,220)); ax.imshow(img); label='trust' if col<2 else 'correct'; ax.set_title(f'{dataset}: {label}',fontsize=9,weight='bold',color=COL['green'] if label=='trust' else COL['blue'])
        txt=(f"y={r.y_true:.2f}  rule={r.rule_prediction:.2f}\nAGRR={r.prediction:.2f}  ρ={r.rule_reliance:.2f}\nq={r.failure_score:.2f}  u={r.uncertainty:.2f}" if dataset=='ESOL' else f"label={int(r.y_true)}  rule p={r.rule_prediction:.2f}\nAGRR p={r.prediction:.2f}  ρ={r.rule_reliance:.2f}\nq={r.failure_score:.2f}  u={r.uncertainty:.2f}"); ax.text(.5,-.02,txt,transform=ax.transAxes,ha='center',va='top',fontsize=7.6)
fig.suptitle('Representative scaffold-held-out audit records (illustrative, not mechanistic explanations)',fontsize=12.5,weight='bold',color=COL['navy']); fig.tight_layout(rect=[0,.03,1,.93]); save(fig,'molecule_audit_cases')

print('generated',len(list(FIG.glob('*.pdf'))),'PDF figures')
