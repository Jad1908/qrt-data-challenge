# QRT Data Challenge

A modeling pipeline for the QRT data challenge: feature engineering, model training/evaluation, and submission generation.

This is a work in progress and will evolve as the approach develops.

## Project layout

- `src/` — core library (config, data loading, features, models, validation, evaluation, experiments, submission)
- `notebooks/` — exploration
- `data/` — raw/processed data (not tracked)
- `outputs/` — generated artifacts (predictions, reports, etc.)
- `main.py` — entry point for running an experiment end-to-end

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Usage

```bash
uv run python main.py
```

Edit `main.py` to configure and run a different experiment via `ExperimentConfig`.
