# PrisonersDilemmaSim

An interactive analytics dashboard for exploring repeated Prisoner’s Dilemma strategies and tournament results.

Read more about the Prisoner’s Dilemma here: https://plato.stanford.edu/entries/prisoner-dilemma/

## What’s in the demo

The Dash app (`pages/app.py`) provides:

- **Overview page** (`/`)
  - Total points per strategy (bar chart)
  - Summary table (total points, average points/round, cooperate rate)
- **Profile Overview page** (`/profiles`)
  - Human-friendly **description / origin / notes** for each strategy
  - Computed KPIs (avg points/round, match win/tie rate, cooperate rate)
  - Performance breakdown vs each opponent + behavior by round

## Run locally

### 1) Install dependencies

This project uses Python + Dash. You’ll need these packages installed:

- `dash`
- `dash-bootstrap-components`
- `pandas`
- `plotly`

Install with pip:

```bash
pip install dash dash-bootstrap-components pandas plotly
```

### 2) Start the app

From the repo root:

```bash
python pages/app.py
```

Then open:

- `http://127.0.0.1:8050/` (Overview)
- `http://127.0.0.1:8050/profiles` (Profile Overview)

## Notes

- **Generated artifacts** (like `results.csv` and Python `__pycache__`) are intentionally ignored via `.gitignore`.
- The simulation is run in-memory via `pages/game_logic.py` (`simulate_tournament()`), so the UI is always driven by the current settings (rounds, repetitions, seed).

## Screenshots (older runs)

![strategy_pair_figure](https://github.com/user-attachments/assets/265ce637-3d95-4198-b0e0-e6049be8c7e0)

![image](https://github.com/user-attachments/assets/02bcb1f7-6ea7-46b8-890f-5ae9b2db10f1)

![image](https://github.com/user-attachments/assets/804fc7dc-f2d8-487f-9bb8-425f49cdc77a)

![image](https://github.com/user-attachments/assets/0a17e9d7-23e0-43c1-82e2-21aa92e13935)
