# Day 1 completion report

Date: 13 September 2026

## Outcome

The original research notebook has been converted into a clean, testable ML foundation suitable for a public portfolio repository. Day 1 is complete.

## Completed

- Audited the 147-cell research notebook and revised manuscript.
- Defined the scientifically correct target: observed high parity at interview, not future fertility risk.
- Created a versioned Python package and experiment configuration.
- Added memory-safe Stata column loading for the multi-gigabyte NFHS file.
- Added validity checks and explicit outcome-proximal leakage protection.
- Added fold-local imputation, encoding, random-forest construction, nested validation, OOF threshold selection, bootstrap AUC, and protected holdout evaluation.
- Added primary India and harmonised transport feature contracts.
- Added a deterministic synthetic dataset and runnable demo model so the public repository works without licensed DHS microdata.
- Added a versioned model bundle with threshold, features, metrics, software version, and intended-use metadata.
- Added a clean notebook walkthrough and command-line training/validation scripts.
- Added unit tests plus a dependency-light Day-1 acceptance command.
- Added README, data card, model card, architecture, research metric source, and interview explanations.
- Protected raw `.DTA`, caches, secrets, research models, and row-level outputs through Git ignore rules.

## Verified today

- Synthetic model builds deterministically.
- Single and batch inference return probabilities in `[0, 1]`.
- Ages outside 15-49 are rejected.
- `v201` and other outcome-proximal variables are blocked from predictors.
- Missing outcomes are excluded instead of silently becoming class 0.
- Research metrics are read from one versioned JSON source.
- Python modules compile and notebook/JSON files parse successfully.
- Raw DHS formats, private model files, and `.env` are ignored by Git.

## Correct boundary

The real NFHS/Nepal training run was not repeated because licensed raw microdata are deliberately outside this repository. The training and external-validation commands are ready for an approved local environment. Manuscript metrics remain distinct from the synthetic demo metrics.

## Day 2 entry point

Day 2 will commit and runtime-test the FastAPI service, Streamlit dashboard, Docker configuration, and recruiter-facing screenshot. Those files are already prepared in the working tree but remain outside the Day-1 commit history.
