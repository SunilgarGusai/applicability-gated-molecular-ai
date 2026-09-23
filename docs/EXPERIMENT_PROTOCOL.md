# Frozen Experimental Protocol

## Research question

The study asks when an explicit but fallible scientific rule should be trusted, corrected, or treated cautiously under molecular chemical-space shift.

## Representation and rule integration

AGRR combines graph and descriptor information with an explicit rule expert. The final prediction uses molecule-specific rule reliance rather than enforcing the rule globally. The scientific-rule pathway, applicability signal, learned trust/reliance, learned molecular expert, ensemble uncertainty, and independent rule-failure score are retained as separately auditable quantities.

## Datasets and controls

- ESOL: regression rule-guided task.
- BBBP: classification task using a Clark-derived passive-diffusion prior rather than treating the Clark relationship as a ground-truth binary mechanism.
- Lipophilicity and FreeSolv: sensitivity/no-rule tasks.
- B3DB: external BBB transfer corpus after exact standardized-SMILES de-overlap against BBBP.

## Splits

Primary internal evaluation uses:

- random splits;
- Bemis-Murcko scaffold splits;
- Butina similarity-cluster splits;
- five frozen seeds: 11, 29, 47, 83, 131.

## Leakage controls

Molecular auditing and duplicate/conflict handling occur before splitting. Training-only transformations and fitting are isolated from test labels. Validation data are used for model selection/calibration where applicable. Test/external labels are not used to tune the reported predictors.

## Comparators and controls

The study includes fixed scientific rules/priors, descriptor/fingerprint classical models, graph/descriptor learned models, unrestricted or ungated correction variants, AGRR, and rule/gate control variants where supported by the frozen package.

A central interpretation constraint is that the primary AGRR predictor is a three-member ensemble while multiple comparator/control models are single networks. Therefore, differences between those rows cannot be attributed uniquely to the applicability-gating mechanism.

## Reliability analyses

The frozen evaluation includes:

- predictive metrics under multiple split families;
- calibration and ensemble uncertainty;
- selective prediction/risk-coverage analyses;
- rule reliance/applicability diagnostics;
- a separately trained rule-failure score;
- ablation/control analysis;
- external B3DB transfer;
- explanation-repeatability sensitivity analysis.

## Interpretation discipline

Mixed findings are retained. The study does not claim universal model superiority, does not attribute all gains to gating, does not present the ESOL equation as an independent external prior, and does not make a deployment-safety guarantee.
