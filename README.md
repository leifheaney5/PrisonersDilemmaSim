# Prisoner’s Dilemma Simulation

**Prisoner’s Dilemma Simulation** is a public-facing web app for exploring the *Iterated* Prisoner’s Dilemma: run tournaments, inspect strategies, compare head‑to‑head behavior, and play matches yourself.

- Learn the basics: [Stanford Encyclopedia of Philosophy: Prisoner’s Dilemma](https://plato.stanford.edu/entries/prisoner-dilemma/)


## What you can do

- **Run a live tournament** (round‑robin) and watch leaderboards + timelines update in real time
- **Play a match** (you vs any strategy) and see outcomes unfold round‑by‑round
- **Explore strategy profiles** with behavior visualizations and a quick “scorecard”
- **Compare two strategies** head‑to‑head
- **Build a custom strategy** (safe rule-based controls) and test it in tournaments
- **Export results and charts** as CSV / PNG / PDF
- **Learn the game** through an interactive payoff example and short lessons on repetition, mistakes, and result interpretation

Live tournament charts normalize points and moves by rounds played so partial
round-robin progress does not favor strategies that happened to play first. The
dashboard also plots payoff efficiency against cooperation, while human matches
pair cumulative scores with a round-by-round cooperation/defection ribbon.
The profile page includes a selectable pairwise matchup matrix for payoff,
cooperation, win rate, and score margin. The custom builder previews unsaved rules
against scripted opponents, and complete experiment settings can be exported and
restored as a versioned JSON file.

The strategy-builder preview also exposes a round-by-round decision trace for its
base response, safety override, reputation threshold, endgame rule, and noise.
This makes composed strategies inspectable before they are saved or entered into
a tournament.

The Experiment page is organized into four steps: **Choose, Configure, Run, and
Review**. Optional tournament settings are separate from the basic setup. Run
controls remain visible while results update and adapt to narrow screens.

Tournament and human-match controls can add reproducible execution errors. The
simulator records both the move a player intended and the move that was actually
executed, so accidental defections and recovery behavior remain inspectable.
Live tournaments can also include self-play, adding each strategy's diagonal
matchup to the usual round robin. Both settings are preserved in exported
experiment configurations.

The UI labels the default settings as the **Classic IPD format**. Enabling
self-play or a non-zero execution-error probability displays an **Experimental
variant** notice so extensions to the app's original unique-pairing,
error-free format are never mistaken for the baseline game. These variants keep
the same two legal actions and payoff matrix.

## Pages / routes

| Route | Purpose |
|------:|---------|
| `/` | Overview, background, and key concepts |
| `/learn` | Interactive introduction to incentives, repeated play, mistakes, and interpreting results |
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

## Built-in strategy catalog

The app includes 48 built-in strategies. They range from standard IPD baselines
to project-specific experimental policies:

- **Baselines:** MrNiceGuy (always cooperate), BadCop (always defect), ImSoRandom
- **Reciprocity:** TitForTat, SuspiciousTitForTat, GenerousTitForTat, Joss, WinStayLoseShift
- **Retaliation variants:** TitForTwoTats, TwoTitsForTat, HardTitForTat, Gradual, HoldingAGrudge, PastTrauma
- **Majority and threshold policies:** SoftMajority, HardMajority, CalculatedDefector, ForgiveButDontForget
- **Testing and adaptive heuristics:** Prober, LongTermRelationship, KeepingPeace, ParkBus, Illuminati
- **Experimental learners:** DebtCollector (repayable trust ledger), PatternHunter (transition prediction),
  and EntropyBroker (recent-behavior uncertainty shield)
- **Schedules and novelty policies:** BadAlternator, RitualDefection, TripleThreat, Pattern, RandomPrime,
  Fibonacci, DefectiveFriedman, CooperativeProth, FriendlySquare, and Shootout
- **Phase and commitment policies:** Pushover, Thief, NeverSwitchUp, LosingMyMind, and DefectiveDeputy
- **Opponent classification and endgame policies:** BadJudgeOfCharacter, BadDivorce, RandomStranger,
  MarkedMan, and Lottery

The similarly named retaliation strategies intentionally differ:

| Strategy | Rule |
|---|---|
| **TitForTwoTats** | Defect only after two consecutive opponent defections |
| **TwoTitsForTat** | Defect when the opponent defected in either of the previous two rounds |
| **HardTitForTat** | Defect when the opponent defected in any of the previous three rounds |
| **Gradual** | Add increasingly long punishments after successive defections, then cooperate twice to calm the interaction |

SoftMajority cooperates on tied histories, including the opening move;
HardMajority defects on ties. Strategy profile pages show whether a policy is
deterministic or stochastic, reactive or scheduled, and how much memory it uses.

### Strategy configuration and reproducibility

Built-in profiles, aliases, traits, and horizon-aware declarations live in
`pages/strategy_catalog.py`. New built-ins must also provide batch and incremental
implementations and parity coverage in `tests/test_strategies.py`.

Batch tournaments use a tournament-local random generator, so a seed produces
repeatable results without changing Python's process-global random state. Live
tournaments keep their random state inside the JSON-serializable tournament state,
allowing a paused experiment to resume exactly.

Custom strategies accept the following normalized fields:

| Field | Accepted value |
|---|---|
| `start_move` | `cooperate` or `defect` |
| `use_tft` | Boolean |
| `use_grudge` | Boolean |
| `response_mode` | `fixed`, `tft`, `anti_tft`, `soft_majority`, or `hard_majority` |
| `retaliation_window` | Non-negative integer |
| `threshold_enabled` | Boolean |
| `defect_rate_threshold` | Number from 0 through 1 |
| `min_history` | Non-negative integer |
| `endgame_after_turn` | Non-negative integer |
| `noise` | Number from 0 through 1 |

Unknown fields and invalid values are rejected before a tournament or human match
starts. The visual builder includes recipe starters and composes rules in an explicit
order: base response, retaliation, reputation threshold, endgame, then noise. Live
tournaments accept 2–10 strategies, with bounded round, repetition,
recent-event, and timeline settings to protect server and browser state.

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
