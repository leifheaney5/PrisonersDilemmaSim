# Prisoner’s Dilemma Simulation

**Prisoner’s Dilemma Simulation** is a public-facing web app for exploring the *Iterated* Prisoner’s Dilemma: run tournaments, inspect strategies, compare head‑to‑head behavior, and play matches yourself.

- Learn the basics: [Stanford Encyclopedia of Philosophy — Prisoner’s Dilemma](https://plato.stanford.edu/entries/prisoner-dilemma/)

> Live demo (Render): *(add your Render URL here)*

## What you can do

- **Run a live tournament** (round‑robin) and watch leaderboards + timelines update in real time
- **Play a match** (you vs any strategy) and see outcomes unfold round‑by‑round
- **Explore strategy profiles** with behavior visualizations and a quick “scorecard”
- **Compare two strategies** head‑to‑head
- **Build a custom strategy** (safe rule-based controls) and test it in tournaments
- **Export results and charts** as CSV / PNG / PDF

## Pages / routes

| Route | Purpose |
|------:|---------|
| `/` | Overview, background, and key concepts |
| `/experiment` | Live tournament runner + “Play a match” + custom strategy builder |
| `/profiles` | Strategy profiles + analytics + comparisons |
| `/donate` | PayPal hosted button |
| `/explore` | Legacy deep-link (redirects to `/experiment`) |

## How the simulation works (quick)

### Payoff matrix (per round)
Each round uses the classic payoff structure shown in-app:
- Mutual cooperation: **3, 3**
- Temptation to defect: **5, 0**
- Sucker’s payoff: **0, 5**
- Mutual defection: **1, 1**

### Tournament structure
- A **match** is two strategies playing for **N rounds**
- A **tournament** is a **round‑robin** across the selected strategies
- **Repetitions** repeat each pairing to reduce randomness and estimate typical performance

### Settings that matter
- **Rounds per match**: longer matches make “forgiveness” and “retaliation” dynamics more visible
- **Repetitions**: stabilizes results (especially with stochastic strategies)
- **Seed**: makes randomness reproducible
- **Known match length**: when enabled, strategies may know the total round count (enabling end‑game behavior)

For performance and clarity, live tournaments are capped at **10 strategies per run**.

## Run locally

### 1) Install dependencies

This project uses Python + Dash. Install dependencies from the repo root:

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

## Deploy on Render

This repo is set up for Render with `render.yaml`.

- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn pages.app:server --bind 0.0.0.0:$PORT`

In the Render dashboard, create a new **Web Service** from this repo. Render will detect `render.yaml` automatically (or you can paste the build/start commands manually).

## Screenshots

<img width="2348" height="1237" alt="firefox_O8TFtVMfpL" src="https://github.com/user-attachments/assets/784ce6c9-016f-4d00-9cbd-941c246b9f3c" />
<img width="2345" height="1230" alt="firefox_8S6U49I7b4" src="https://github.com/user-attachments/assets/09b8134b-9ddd-427c-8d0b-67071d74247c" />
<img width="2349" height="1158" alt="firefox_arKULWTOkB" src="https://github.com/user-attachments/assets/cabf8873-a6bd-418f-8891-255b5a90033f" />
<img width="2345" height="1235" alt="firefox_j566ruD900" src="https://github.com/user-attachments/assets/68c1c73e-8b3f-43cd-9cbe-24a294ce90c0" />
<img width="2347" height="1232" alt="firefox_ygMpUqeHqO" src="https://github.com/user-attachments/assets/4255c2af-bdf8-4b16-9f1b-866dd4fab99e" />

## Resources
- [Axelrod-Python (Iterated Prisoner’s Dilemma library)](https://axelrod.readthedocs.io/en/stable/)
- Axelrod, R. (1981). *The Evolution of Cooperation*. Science. https://doi.org/10.1126/science.7466396
- Nowak, M. A. (2006). *Five Rules for the Evolution of Cooperation*. Science. https://doi.org/10.1126/science.1133755
- Press, W. H., & Dyson, F. J. (2012). *Iterated Prisoner’s Dilemma contains strategies that dominate any evolutionary opponent*. PNAS. https://doi.org/10.1073/pnas.1206569109

## Repo notes
- Generated artifacts (e.g., Python `__pycache__`) are intentionally ignored via `.gitignore`.
- Core simulation logic lives in `pages/game_logic.py`; the Dash UI is in `pages/app.py`.
