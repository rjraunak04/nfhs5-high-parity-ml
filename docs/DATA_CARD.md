# Data card

## Sources

- India: National Family Health Survey 2019-2021, women's Individual Recode file.
- Nepal: Nepal Demographic and Health Survey 2022, women's Individual Recode file.

The author reported 724,115 India records and 14,845 Nepal records after the study's checks. Microdata are available only to approved researchers under the DHS Program data-use agreement.

## Repository data policy

This public repository contains only `data/demo_batch.csv`, generated from a fixed random seed. It contains no respondent records and no values copied from DHS microdata.

Never commit:

- `.DTA`, `.sav`, or other raw survey extracts;
- row-level predictions or SHAP values from real participants;
- cached dataframes, Optuna databases, or fitted research models;
- Google Drive file IDs or direct download links for controlled microdata.

The root `.gitignore` blocks these formats. Still inspect `git status` before every push.

## Variable contract

| Clean name | DHS source | Meaning | Public demo range |
|---|---|---|---:|
| current_age | v012 | age at interview | 15-49 |
| residence | v025 | urban/rural | 1-2 |
| education | v106 | education level | 0-3 |
| wealth | v190 | household wealth quintile | 1-5 |
| in_union | derived from v501 | married/living together | 0-1 |
| fertility_class | derived from v201 | children ever born >= 3 | 0-1 |

The primary India offline feature set additionally uses religion (`v130`), state (`v024`), and caste/social group (`s116`).

## Quality checks

- Resolve column names case-insensitively, then read only required columns.
- Convert columns to numeric without converting value labels into model strings.
- Exclude missing `v201` outcomes.
- Keep only ages 15-49 for the stated study population.
- Derive `in_union` from DHS codes 1 and 2.
- Validate that blocked outcome-proximal variables are absent from predictors.
- Fit imputation and categorical encoding inside each training fold.

## Reproducibility fingerprint

The training command records file name, byte size, and SHA-256 over the first and last 1 MiB. This fast fingerprint can detect accidental source changes without reading the full multi-gigabyte file a second time. It is not a cryptographic proof of every byte.
