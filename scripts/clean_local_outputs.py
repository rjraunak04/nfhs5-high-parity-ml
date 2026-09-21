"""Remove only reproducible local build/test outputs inside this repository."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [root / "outputs", root / ".pytest_cache", root / ".ruff_cache"]
    targets.extend(
        path for path in root.rglob("__pycache__") if ".venv" not in path.relative_to(root).parts
    )
    for target in targets:
        if target.exists() and target.is_relative_to(root):
            shutil.rmtree(target)
            print(f"Removed reproducible local output: {target.relative_to(root)}")


if __name__ == "__main__":
    main()
