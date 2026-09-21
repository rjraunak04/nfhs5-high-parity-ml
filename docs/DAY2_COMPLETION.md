# Day 2 completion report

Date: 2026-09-18

## Completed

- FastAPI endpoints for health, model metadata, feature schema, single prediction, and batches of up to 500 records.
- Strict Pydantic validation rejects out-of-range values and unexpected fields.
- Streamlit interface supports both one-record scoring and validated CSV batch scoring.
- Batch results can be reviewed in the browser and downloaded as CSV.
- Research-only disclaimer and synthetic-artifact status remain visible in the interface and API.
- Dockerfile packages the API, defines a health check, and uses the synthetic model artifact.
- Docker Compose defines separate API and dashboard services on ports 8000 and 8501.
- API tests cover discovery, feature contract, single prediction, invalid age, batch prediction, and empty batches.
- Offline acceptance script checks routes, schemas, inference, dashboard safeguards, and container configuration.

## Verification evidence

Run from the repository root:

```bash
PYTHONPATH=src python scripts/day2_acceptance.py
python -m compileall -q src app scripts tests
```

Expected result: `DAY 2 ACCEPTANCE: PASS`.

The current managed workspace cannot download FastAPI/Streamlit or start Docker. Full integration checks remain encoded in `tests/test_api.py` and `.github/workflows/ci.yml`; GitHub Actions installs the application extras, runs lint and tests, and builds the container.

## Safety boundary

The deployed artifact uses synthetic data. It classifies a synthetic representation of observed high parity at interview and must not be described as a fertility forecast, diagnosis, treatment tool, or causal model.
