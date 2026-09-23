"""Add leakage-controlled B3DB transfer comparators to the frozen external results."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from classical_models import build_models, fit_predict
from graph_models import mol_to_graph, train_model, predict_model
from metrics import classification_metrics
from run_experiments import SEEDS, load_features, fit_rule_transform, transform_rule, applicability_scores, stable_sigmoid, temperature_scale


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / 'experiments' / 'results'
    train = load_features(root, 'BBBP')
    ext0 = load_features(root, 'B3DB_external')

    overlap = set(map(str, train['smiles']))
    keep = np.array([str(s) not in overlap for s in ext0['smiles']])
    ext = {k: v[keep] for k, v in ext0.items()}

    y = train['target']
    rng = np.random.default_rng(SEEDS[0])
    idx = np.arange(len(y)); rng.shuffle(idx)
    cut = int(0.85 * len(idx))
    tr_idx, va_idx = np.sort(idx[:cut]), np.sort(idx[cut:])

    scaler = StandardScaler().fit(train['descriptors'][tr_idx])
    d_train = scaler.transform(train['descriptors']).astype(np.float32)
    d_ext = scaler.transform(ext['descriptors']).astype(np.float32)
    params = fit_rule_transform(train['rule'][tr_idx], y[tr_idx], 'classification')
    r_train = transform_rule(train['rule'], params, 'classification').astype(np.float32)
    r_ext = transform_rule(ext['rule'], params, 'classification').astype(np.float32)
    app_train = applicability_scores(train['descriptors'], train['descriptors'][tr_idx], 'BBBP')
    app_ext = applicability_scores(ext['descriptors'], train['descriptors'][tr_idx], 'BBBP')

    rows = []
    base = {
        'dataset': 'B3DB_external', 'split': 'transfer', 'seed': SEEDS[0],
        'n_external': len(ext['target']), 'overlap_removed': int((~keep).sum()),
        'n_positive': int(np.sum(ext['target'] == 1)), 'n_negative': int(np.sum(ext['target'] == 0)),
    }

    pred = stable_sigmoid(r_ext)
    rows.append({**base, 'model': 'Fixed_rule', **classification_metrics(ext['target'], pred)})

    models = build_models('classification', SEEDS[0])
    for name in ['ExtraTrees_desc', 'RF_Morgan']:
        model = models[name]
        xtr = train['fingerprints'] if name == 'RF_Morgan' else d_train
        xext = ext['fingerprints'] if name == 'RF_Morgan' else d_ext
        pred = fit_predict(model, xtr[tr_idx], y[tr_idx], xext, 'classification')
        rows.append({**base, 'model': name, **classification_metrics(ext['target'], pred)})

    graphs_train = [mol_to_graph(str(train['smiles'][i]), d_train[i], float(y[i]), float(r_train[i]), float(app_train[i]), i) for i in range(len(y))]
    graphs_ext = [mol_to_graph(str(ext['smiles'][i]), d_ext[i], float(ext['target'][i]), float(r_ext[i]), float(app_ext[i]), i) for i in range(len(ext['target']))]
    valid_logits, members = [], []
    for m in range(3):
        trained = train_model([graphs_train[i] for i in tr_idx], [graphs_train[i] for i in va_idx], mode='fusion', task='classification', seed=SEEDS[0] + 1009*m, epochs=70)
        vo, _, _, _ = predict_model(trained.model, [graphs_train[i] for i in va_idx])
        to, _, _, _ = predict_model(trained.model, graphs_ext)
        valid_logits.append(vo); members.append(to)
    temp = temperature_scale(y[va_idx], np.vstack(valid_logits).mean(axis=0))
    probs = stable_sigmoid(np.vstack(members) / temp)
    pred = probs.mean(axis=0); std = probs.std(axis=0, ddof=1)
    rows.append({**base, 'model': 'Graph_descriptor_fusion', **classification_metrics(ext['target'], pred, std)})

    proposed_path = result_dir / 'external_metrics.csv'
    if proposed_path.exists():
        old = pd.read_csv(proposed_path)
        proposed = old[old['model'].eq('Proposed_AGRR')]
        if len(proposed):
            rec = proposed.iloc[0].to_dict()
            rec.update(base); rec['model'] = 'Proposed_AGRR'
            rows.append(rec)

    out = pd.DataFrame(rows)
    order = ['Fixed_rule', 'ExtraTrees_desc', 'RF_Morgan', 'Graph_descriptor_fusion', 'Proposed_AGRR']
    out['_order'] = out['model'].map({m:i for i,m in enumerate(order)})
    out = out.sort_values('_order').drop(columns='_order')
    out.to_csv(proposed_path, index=False)
    print(out[['model','n_external','AUROC','AUPRC','MCC','Brier','ECE']].to_string(index=False))


if __name__ == '__main__':
    main()
