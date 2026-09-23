from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
ROOT=Path(__file__).resolve().parents[1]; FIG=ROOT/'manuscript/figures'
COL={'navy':'#17365D','blue':'#3C78D8','green':'#4E9F3D'}
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
pred=pd.read_csv(ROOT/'experiments/results/predictions.csv')
fail=pd.read_csv(ROOT/'experiments/results/failure_predictions.csv')
proc={d:pd.read_csv(ROOT/f'data/processed/{d}.csv') for d in ['ESOL','BBBP']}
fig,axes=plt.subplots(2,2,figsize=(10.5,7.2))
for row,dataset in enumerate(['ESOL','BBBP']):
    p=pred[(pred.dataset==dataset)&(pred.split=='scaffold')&(pred.seed==20260711)&(pred.model=='Proposed_AGRR')].copy()
    f=fail[(fail.dataset==dataset)&(fail.split=='scaffold')&(fail.seed==20260711)].copy()
    z=p.merge(f[['sample_index','failure_score','rule_absolute_error']],on='sample_index')
    z['model_error']=abs(z.y_true-z.prediction); z['improvement']=z.rule_absolute_error-z.model_error
    trust=z.sort_values(['failure_score','rule_reliance'],ascending=[True,False]).iloc[0]
    correct=z[(z.improvement>0)].sort_values(['improvement','rule_absolute_error'],ascending=False).iloc[0]
    for col,(kind,r) in enumerate([('trust',trust),('correct',correct)]):
        ax=axes[row,col]; ax.axis('off')
        smi=proc[dataset].iloc[int(r.sample_index)].canonical_smiles
        mol=Chem.MolFromSmiles(smi); img=Draw.MolToImage(mol,size=(560,340))
        ax.imshow(img)
        ax.set_title(f'{dataset}: {kind}',fontsize=12,weight='bold',color=COL['green'] if kind=='trust' else COL['blue'])
        if dataset=='ESOL':
            txt=(f"Observed logS: {r.y_true:.2f}    Rule: {r.rule_prediction:.2f}    AGRR: {r.prediction:.2f}\n"
                 f"Rule reliance ρ: {r.rule_reliance:.2f}    Failure score q: {r.failure_score:.2f}    Uncertainty u: {r.uncertainty:.2f}")
        else:
            txt=(f"Observed class: {int(r.y_true)}    Rule probability: {r.rule_prediction:.2f}    AGRR probability: {r.prediction:.2f}\n"
                 f"Rule reliance ρ: {r.rule_reliance:.2f}    Failure score q: {r.failure_score:.2f}    Uncertainty u: {r.uncertainty:.2f}")
        ax.text(.5,-.015,txt,transform=ax.transAxes,ha='center',va='top',fontsize=9.2,color='#202124')
fig.suptitle('Representative scaffold-held-out audit records',fontsize=15,weight='bold',color=COL['navy'])
fig.text(.5,.018,'Illustrative audit records; molecular drawings and scores are not mechanistic explanations.',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.05,1,.93],h_pad=2.8,w_pad=1.5)
fig.savefig(FIG/'molecule_audit_cases.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(FIG/'molecule_audit_cases.png',dpi=360,bbox_inches='tight',pad_inches=.03)
