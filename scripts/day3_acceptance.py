"""Day 3 end-to-end release acceptance for API and dashboard processes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wait_for(url: str, attempts: int = 60) -> tuple[int, str]:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return response.status, response.read().decode()
        except Exception:  # the child service may still be starting
            time.sleep(0.25)
    raise RuntimeError(f"Service did not become healthy: {url}")


def main() -> None:
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        ),
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/dashboard.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        ),
    ]

    try:
        api_status, api_body = wait_for("http://127.0.0.1:8000/health")
        dashboard_status, dashboard_body = wait_for("http://127.0.0.1:8501/_stcore/health")

        payload = {
            "current_age": 31,
            "residence": 2,
            "education": 2,
            "wealth": 3,
            "in_union": 1,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/predict",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            prediction_status = response.status
            prediction = json.loads(response.read().decode())

        assert api_status == 200 and json.loads(api_body)["status"] == "ok"
        assert dashboard_status == 200 and dashboard_body == "ok"
        assert prediction_status == 200
        assert prediction["demo_only"] is True
        assert 0 <= prediction["probability"] <= 1

        print("DAY 3 ACCEPTANCE: PASS")
        print(json.dumps(prediction, indent=2))
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
