# Day 3 release report

Date: 2026-09-21

## Release gate

- Ruff: passed.
- Pytest: 13 passed.
- Source coverage: 55% overall; serving, schema, modelling, and inference paths are covered.
- API process: `/health` returned HTTP 200.
- Dashboard process: Streamlit health returned HTTP 200.
- Prediction smoke test: returned a typed synthetic-demo prediction.
- Sensitive-data scan: no tracked DHS/Stata/SAV/SAS files.
- Secret-pattern scan: no detected GitHub, AWS, or private-key signatures.
- Git ignore verification: `.DTA`, `.env`, and local outputs are excluded.
- Streamlit telemetry disabled and CORS protection enabled.
- Missing synthetic demo artifacts regenerate deterministically on first startup.

## Release contents

- Reproducible Python package and synthetic demo artifact.
- FastAPI single/batch prediction service with OpenAPI documentation.
- Streamlit single-record and CSV-batch interface.
- Docker and Docker Compose configuration.
- CI workflow for lint, tests, runtime acceptance, and container build.
- Model card, data card, architecture, deployment guide, and interview guide.

## Re-run locally

```bash
python -m pip install -e ".[app,dev]"
python scripts/train_demo_model.py
ruff check .
pytest --cov=src --cov-report=term-missing
python scripts/day3_acceptance.py
```

The public release must retain the research disclaimer and must not include licensed DHS microdata, reviewer correspondence, unpublished manuscript files, or fitted controlled-data models.
