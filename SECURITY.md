# Security policy

## Supported version

Security updates are applied to the latest version on the `main` branch.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could expose data, credentials, or deployment infrastructure.

Use GitHub's private vulnerability reporting feature when it is available. Include:

- the affected file, endpoint, or version;
- clear reproduction steps;
- the expected and observed behaviour;
- the potential impact;
- any safe remediation suggestion.

Please allow reasonable time for investigation before public disclosure.

## Data and secret handling

This repository must not contain:

- NFHS/DHS respondent-level microdata;
- direct identifiers or exported participant records;
- private research-model artifacts;
- API keys, tokens, passwords, or service-account files;
- submitted dashboard profiles or agent conversations.

Use `.env.example` only as a template. Store real secrets in environment variables or the deployment platform's secret manager.

## Scope

The public model is trained on synthetic data and is intended for software demonstration. Findings involving licensed DHS data should be reproduced only in an authorised research environment.
