# Deployment guide

The repository supports two public surfaces:

- **Streamlit dashboard** for interactive scoring, batch uploads, and the research agent.
- **FastAPI service** for typed programmatic access.

The default deployment uses a deterministic synthetic model. It does not require DHS data or an API key.

## Local production-style check

```bash
python scripts/train_demo_model.py
docker compose up --build
```

Verify the API:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/model
curl -X POST http://localhost:8000/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How was the model validated?"}'
```

Open the dashboard at `http://localhost:8501`.

## Streamlit Community Cloud

1. Connect this GitHub repository in Streamlit Community Cloud.
2. Choose the `main` branch.
3. Set the entry point to `app/dashboard.py`.
4. Deploy without secrets for deterministic local routing.
5. Test all three tabs from an incognito window.

The application generates its safe synthetic artifact automatically when it is absent. Licensed data and research-model artifacts must never be uploaded to Streamlit Cloud.

### Optional LLM routing

The agent remains deterministic unless all three deployment secrets are configured:

```text
FERTILITY_ENABLE_LLM=true
FERTILITY_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<deployment secret>
```

Never place an API key in source files, screenshots, Docker build arguments, or logs. If the provider is unavailable, routing automatically falls back to the local English/Hinglish planner.

## Container API

```bash
docker build -t nfhs5-high-parity-api:0.2.0 .
docker run --rm -p 8000:8000 nfhs5-high-parity-api:0.2.0
```

Configure the hosting platform to:

- expose container port `8000`;
- use `/health` for health checks;
- restart unhealthy instances;
- supply secrets through its secret manager;
- retain application logs without recording submitted profiles.

## Pre-release checklist

```bash
ruff check .
pytest
python scripts/evaluate_agent.py
docker build -t nfhs5-high-parity-api:check .
```

Then verify:

- GitHub Actions is green on the release commit;
- the live dashboard loads all tabs;
- one valid and one invalid record behave correctly;
- the agent displays its tool trace and disclaimer;
- the README, package version, and release notes agree;
- no raw data, model secrets, local outputs, or credentials are tracked.
