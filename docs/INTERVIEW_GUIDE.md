# Interview guide

## 30-second introduction

"I converted my NFHS-5 research into a production-style ML repository. The research used 724,115 India records, leakage-aware predictors, nested cross-validation, an untouched holdout, and a separate five-feature India-to-Nepal geographic validation. Because DHS microdata cannot be published, I built a synthetic demo artifact with the same software contract, then served it through FastAPI and Streamlit with Docker, tests, CI, a model card, and strict input validation."

## Questions you must be able to answer

### Why did the reported AUC fall from the older experiment?

The older workflow included fertility-history variables that were too close to accumulated parity and used ordinary SMOTE with categorical survey codes. Removing those shortcuts reduced AUC but made the result more credible and deployable.

### What is target leakage here?

The outcome is children ever born. Variables such as age at first birth, prior birth history, and outcome-derived counts can reconstruct that accumulated outcome. The leakage guard blocks them from the primary predictor set.

### Why nested cross-validation plus a holdout?

Inner folds choose hyperparameters; outer folds estimate selection-aware performance. The final untouched holdout is opened only after selection. This reduces optimistic bias.

### Why AUC and Brier score together?

AUC measures ranking across thresholds. Brier score checks probability accuracy. A model can rank well and still produce poorly calibrated probabilities, which happened in the Nepal transport assessment.

### What does Nepal calibration slope 0.7272 mean?

The transported probabilities were too extreme. The model retained useful ordering but should be recalibrated locally before anyone uses the probability values.

### Why not expose caste and religion in the app?

They are sensitive/contextual research variables that can encode structural conditions. The public demonstration does not need them. Keeping them offline creates a clearer responsible-use boundary.

### Is the deployed model the paper model?

No. The deployed artifact is trained on synthetic data so the engineering can be demonstrated without distributing licensed microdata or research artifacts. The README labels manuscript metrics separately.

### What would you do next in a real company?

Define an operational target and time point, collect prospective outcomes, validate subgroup calibration and error costs, add a registry and monitoring, perform local recalibration, establish human oversight, and run a controlled impact evaluation.

## Honest ownership statement

Say: "I designed and can explain the research problem, feature policy, evaluation choices, and implementation. I used coding assistance for refactoring and review, then verified the data contract, tests, API behavior, and scientific claims myself."

Do not say that every line was typed without tools. Interviewers care whether you understand, can debug, and can defend the system.
