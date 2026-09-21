# Three-day recruiter-ready execution plan

The repository is already scaffolded. Use this sequence to understand it, make honest commits, deploy it, and present it confidently. Do not split one tiny edit into many fake commits; recruiters care more about a coherent history and working software than a green calendar.

## Day 1 - Research to reusable ML package

**Goal:** the code runs without Colab-only state, and the scientific contract is explicit.

### Morning (3 hours)

1. Create the repository and copy this project into it.
2. Read `README.md`, `docs/DATA_CARD.md`, and `docs/MODEL_CARD.md`.
3. Create a virtual environment and install `.[app,dev,train]`.
4. Run `python scripts/train_demo_model.py` and `pytest`.
5. Trace one record through `schemas.py -> inference.py -> model bundle`.

Suggested commit:

```bash
git add README.md pyproject.toml .gitignore LICENSE CITATION.cff configs data research
git commit -m "docs: define high-parity research and data contracts"
```

### Afternoon (4 hours)

1. Study `data.py`: column-subset reading, outcome filtering, mapping, and leakage guard.
2. Study `modeling.py` and `training.py`: fold-local preprocessing, nested validation, threshold selection, and protected holdout.
3. Run the notebook walkthrough.
4. If approved DHS data are locally available, run `--mode quick`. Then train the transport feature set and exercise `validate_nepal.py`. Never move a `.DTA` file into Git.

Suggested commits:

```bash
git add src/fertility_risk/data.py src/fertility_risk/constants.py
git commit -m "feat: add memory-safe DHS ingestion and leakage guards"

git add src/fertility_risk/modeling.py src/fertility_risk/evaluation.py src/fertility_risk/training.py
git commit -m "feat: add nested validation and versioned model training"

git add scripts notebooks
git commit -m "feat: add reproducible training commands and walkthrough"
```

### End-of-day gate

- No raw DHS data or Drive IDs are tracked.
- `pytest` passes.
- You can explain why `v201`, age at first birth, and family-planning history are blocked.
- You can explain why the synthetic demo metrics are not the manuscript metrics.

## Day 2 - Serving, UX, containerisation

**Goal:** a recruiter can run the system without reading the research notebook.

### Morning (3 hours)

1. Start `uvicorn app.api:app --reload`.
2. Exercise `/health`, `/v1/model`, `/v1/predict`, and `/v1/predict/batch` in `/docs`.
3. Deliberately submit age 52 and an extra field to observe schema rejection.
4. Read `tests/test_api.py` and add one edge case in your own words.

Suggested commit:

```bash
git add app/api.py src/fertility_risk/schemas.py src/fertility_risk/inference.py tests
git commit -m "feat: serve validated single and batch predictions"
```

### Afternoon (4 hours)

1. Start `streamlit run app/dashboard.py` and capture one clean screenshot.
2. Verify that the disclaimer, synthetic-data label, and feature labels are visible.
3. Run `docker compose up --build` and test both ports.
4. Add the screenshot under `docs/assets/` and link it from the README.

Suggested commits:

```bash
git add app/dashboard.py .streamlit docs/assets
git commit -m "feat: add responsible interactive model dashboard"

git add Dockerfile docker-compose.yml .dockerignore
git commit -m "build: containerize API and dashboard"
```

### End-of-day gate

- API returns typed JSON and rejects invalid inputs.
- Dashboard works from a clean environment.
- Docker health check passes.
- The deployed UI never calls the outcome a diagnosis or future prediction.

## Day 3 - Quality, deployment, portfolio presentation

**Goal:** public repository, live demo, and interview-ready explanation.

### Morning (3 hours)

1. Run `ruff check .`, `pytest --cov=src`, and `docker build -t nfhs5-high-parity-ml .`.
2. Inspect `git status` and use `git check-ignore` on a local `.DTA` path.
3. Push to a private GitHub repository first and confirm the Actions workflow is green.
4. Run a secret scan in GitHub before making the repository public.

Suggested commit:

```bash
git add .github Makefile CONTRIBUTING.md SECURITY.md
git commit -m "ci: enforce lint tests and container build"
```

### Afternoon (3 hours)

1. Deploy the Streamlit dashboard using `requirements.txt` and `app/dashboard.py`.
2. Put the live link in the README repository description and LinkedIn Featured section.
3. Add one 45-60 second screen recording: input validation, prediction, disclaimer, model information.
4. Rehearse the answers in `docs/INTERVIEW_GUIDE.md`.

Suggested final commits:

```bash
git add README.md docs
git commit -m "docs: add deployment evidence and technical walkthrough"

git tag -a v0.1.0 -m "Recruiter-ready portfolio release"
git push origin main --tags
```

### Final public-release checklist

- Repository name: `nfhs5-high-parity-ml`
- One-line description: `Production-minded ML system for explainable high-parity classification with nested validation, FastAPI, Streamlit, Docker, and CI.`
- Topics: `machine-learning`, `data-science`, `public-health`, `fastapi`, `streamlit`, `docker`, `explainable-ai`, `survey-data`, `scikit-learn`, `mlops`
- Pin this repository on GitHub.
- Add the live dashboard URL under the repository About section.
- Do not claim the associated paper is published until it is accepted and publicly available.
