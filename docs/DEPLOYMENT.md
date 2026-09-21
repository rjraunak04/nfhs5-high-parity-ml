# Deployment guide

## Local production-like run

```bash
python scripts/train_demo_model.py
docker compose up --build
```

Confirm:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/model
```

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a Streamlit app from that repository.
3. Set the entry point to `app/dashboard.py`.
4. Keep `requirements.txt`, `models/demo_transport_model.joblib`, and the package source in the repository.
5. After deployment, open the app from an incognito window and test one valid and one invalid input.

No secret or DHS file is required for the synthetic demo.

## Container service

Build and run only the API image:

```bash
docker build -t nfhs5-high-parity-api:0.1.0 .
docker run --rm -p 8000:8000 nfhs5-high-parity-api:0.1.0
```

A managed container host must route its public port to container port `8000` and use `/health` for health checks.

## Release evidence

Before sharing the repository with recruiters, add:

- live dashboard URL in the GitHub About section;
- a dashboard screenshot below the README architecture;
- a green Actions badge using the repository's real workflow URL;
- a tagged release (`v0.1.0`);
- a short demo recording in LinkedIn Featured.
