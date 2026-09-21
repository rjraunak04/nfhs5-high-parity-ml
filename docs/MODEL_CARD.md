# Model card

## Summary

This repository contains two clearly separated evidence layers:

1. **Research evidence:** manuscript-reported NFHS-5 India and Nepal DHS metrics stored in `research/manuscript_metrics.json`.
2. **Runnable public demo:** a random-forest pipeline trained on deterministic synthetic DHS-shaped data.

The demo proves the software path works; it is not a substitute for the confidential/licensed-data research fit.

## Intended use

- Reproducible ML engineering demonstration.
- Cross-sectional research on observed high parity at survey interview.
- Teaching leakage prevention, nested validation, probability evaluation, and geographic validation.

## Out-of-scope use

- Predicting a woman's future fertility.
- Clinical diagnosis, counselling, benefit eligibility, family-planning targeting, or automated decision-making.
- Causal claims about age, education, wealth, caste, religion, residence, or geography.
- Direct use of India-trained probabilities in Nepal without local recalibration and governance review.

## Outcome

`fertility_class = 1` when children ever born (`v201`) is at least three. Records with missing outcome or invalid age are excluded rather than silently assigned to the negative class.

## Feature sets

| Model | Features | Where used |
|---|---|---|
| Primary India research model | age, residence, education, wealth, religion, caste/social group, state | Offline research only |
| Harmonised transport model | age, residence, education, wealth, union status | India-to-Nepal validation |
| Public synthetic demo | same five harmonised variables | API and dashboard |

Fertility-history variables and outcome-derived features are blocked from the primary predictors. Survey weights, PSU, and stratum are retained for descriptive/statistical work but are not model predictors.

## Evaluation design

- 80:20 stratified India train/holdout split.
- Hyperparameter selection inside inner folds.
- Performance estimation in outer folds.
- Final holdout used after model selection.
- Threshold chosen from training out-of-fold probabilities, never from Nepal or the protected holdout.
- AUC ROC, Brier score, bootstrap interval, calibration, and subgroup results considered together.

## Manuscript-reported performance

| Measure | Result |
|---|---:|
| Primary India nested AUC | 0.8811 +/- 0.0028 |
| Primary India holdout AUC / Brier | 0.8839 / 0.1445 |
| Harmonised India holdout AUC / Brier | 0.8708 / 0.1531 |
| Nepal AUC / Brier | 0.8488 / 0.1523 |
| Nepal AUC 95% bootstrap CI | 0.8426-0.8542 |
| Nepal calibration intercept / slope | 0.3846 / 0.7272 |

The drop in discrimination and slope below one show that transportability is limited and calibration is not preserved.

## Threshold clarification

- `0.4890` belongs to the primary India full-feature model.
- `0.4694` belongs to the separate India common-feature transport model.
- The synthetic demo has its own threshold computed from synthetic out-of-fold predictions.

These thresholds must not be swapped or used to compare AUC between countries.

## Fairness and risk

The India research model includes caste, religion, and state as contextual variables. Their model contributions can reflect correlated structural conditions and must not be interpreted as inherent individual risk. They are excluded from the public UI. Before any real operational study, evaluate subgroup discrimination, calibration, error rates, data governance, consent, human oversight, and the consequences of false positives and false negatives.

## Explainability

Global feature importance and SHAP can describe how the fitted model used recorded variables. They do not estimate the effect of changing a person's education, residence, wealth, caste, religion, or state.

## Known limitations

- Cross-sectional accumulated outcome, not a longitudinal forecast.
- Self-reported survey variables may contain recall and reporting error.
- Predictive evaluation is not design-based national estimation.
- Nepal validation covers only the five harmonised features.
- Age-stratified Nepal performance was substantially weaker for older groups.
- The public demo is synthetic and has no empirical public-health validity.
