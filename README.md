# PrisonersDilemmaSim

**Prisoner’s Dilemma Simulation** is a public-facing web app for exploring repeated Prisoner’s Dilemma strategies, running live experiments, and comparing behaviors with analytics-first visualizations.

Read more about the Prisoner’s Dilemma here: `https://plato.stanford.edu/entries/prisoner-dilemma/`

## What’s in the app

The Dash app (`pages/app.py`) includes:

- **Overview** (`/`): purpose, resources, and “what to do next”
- **Explore** (`/explore`)
  - Total points by strategy
  - Summary table
  - Strategy similarity heatmap (feature distance)
  - Export: **CSV / PNG / PDF**
- **Profiles** (`/profiles`)
  - Strategy descriptions + performance KPIs
  - Opponent breakdown and behavior-by-round charts
  - Export: **CSV / PNG / PDF**
- **Experiment** (`/experiment`)
  - Live tournament runner (timelines + leaderboard + summary modal)
  - “Play a match” vs a selected strategy
  - Build a custom strategy (rule-based) and test it
- **Donate** (`/donate`): PayPal hosted button

## Run locally

### 1) Install dependencies

This project uses Python + Dash. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Start the app

From the repo root:

```bash
python pages/app.py
```

Then open:

- `http://127.0.0.1:8050/`

## Notes

- **Generated artifacts** (like `results.csv` and Python `__pycache__`) are intentionally ignored via `.gitignore`.
- The simulation is run in-memory via `pages/game_logic.py` (`simulate_tournament()`), so the UI is always driven by the current settings (rounds, repetitions, seed).

## Deploy on Render

This repo is set up for Render with `render.yaml`.

- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn pages.app:server --bind 0.0.0.0:$PORT`

In the Render dashboard, create a new **Web Service** from this repo. Render will detect `render.yaml` automatically (or you can paste the build/start commands manually).

## Screenshots (older runs)

![strategy_pair_figure](https://github.com/user-attachments/assets/265ce637-3d95-4198-b0e0-e6049be8c7e0)

![image](https://github.com/user-attachments/assets/02bcb1f7-6ea7-46b8-890f-5ae9b2db10f1)

![image](https://github.com/user-attachments/assets/804fc7dc-f2d8-487f-9bb8-425f49cdc77a)

![image](https://github.com/user-attachments/assets/0a17e9d7-23e0-43c1-82e2-21aa92e13935)
