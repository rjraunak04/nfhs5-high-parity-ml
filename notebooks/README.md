# Notebook policy

`01_project_walkthrough.ipynb` is a short entry point for exploration. Reusable logic lives in `src/fertility_risk`; do not duplicate production logic across notebook cells.

The original 147-cell Colab research notebook is not published here because it contains legacy branches, Drive-specific state, and controlled-data download logic. Its final leakage-aware methodology has been refactored into tested modules and documented in the model card.
