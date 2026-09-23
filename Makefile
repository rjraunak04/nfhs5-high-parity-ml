.PHONY: install demo test agent-eval lint api dashboard docker clean

install:
	python -m pip install -e ".[app,dev,train]"

demo:
	python scripts/train_demo_model.py

test: demo
	pytest

agent-eval:
	python scripts/evaluate_agent.py

lint:
	ruff check .

api: demo
	uvicorn app.api:app --reload

dashboard: demo
	streamlit run app/dashboard.py

docker:
	docker compose up --build

clean:
	python scripts/clean_local_outputs.py
