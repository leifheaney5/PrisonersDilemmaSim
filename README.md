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
The live leaderboard is paired with a deterministic written ranking that follows
the selected points, win-rate, or cooperation metric and updates as play advances.
The profile page includes a selectable pairwise matchup matrix for payoff,
cooperation, win rate, score margin, combined payoff, inequality, mutual outcomes,
and exploitation. Selecting a matrix cell opens a round-by-round matchup replay
with action and cumulative-score charts. The replay also opens a Match Arena: a
round selector reveals large action tokens, intended-versus-executed decisions,
round payoffs, cumulative scores, and an accessible factual description. The custom builder previews unsaved rules
against scripted opponents, and complete experiment settings can be exported and
restored as a versioned JSON file.

The Profiles page also opens with an interactive strategy landscape. Each point is
a strategy: horizontal position shows its change in cooperation after cooperative
versus defective opponents, vertical position shows action stability, color shows
overall cooperation, and point size shows payoff efficiency. Selecting a point
opens that strategy's full profile.

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

The Strategy Lab also includes an **Evolution** tab. This is labeled as an
experimental extension because strategy shares change between generations rather
than remaining fixed in a round-robin field. Users can choose deterministic
replicator dynamics or a seeded finite-population Moran birth-death process. Both
models support selection and mutation. Individual matches retain the classic two
actions and payoff matrix. Moran runs also record integer population counts and are
limited by a maximum number of birth-death events.

Evolution populations can use equal shares, seeded random shares, an invasion
preset, or editable starting shares. Evolution settings export separately from
tournament settings and identify the experiment type, game, and game version.
Only selected strategies are simulated when building the evolution matchup table.
Completed evolution runs are stored as versioned, JSON-safe state. A generation
inspector shows population share, change, fitness, payoff, and cooperation without
rerunning matches. A deterministic event log records changes in population leaders,
majority shares, low shares, and cooperation thresholds.
Completed evolution results can also be downloaded and opened later in read-only
inspection mode. Imported results are validated and never trigger a simulation.
Completed runs also receive factual highlights covering population leadership,
the largest share changes, final payoff, cooperation, and diversity.

The Strategy Lab includes a **Robustness** tab for reproducible seed sweeps with
workload limits, seed-level payoff, cooperation and rank records, plus descriptive
summaries such as quartiles, rank ranges, win rates, and top-three rates. Box plots
show payoff spread and a heatmap shows rank for every seed. Results can be downloaded
as versioned JSON and reopened later without rerunning tournaments. A seed inspector
shows the payoff, cooperation rate, rank, and participation count for every strategy
in one underlying seed. This analysis is an experimental extension because it
compares multiple independent seeded tournaments rather than changing the classic
game. Result highlights identify the highest mean payoff, lowest payoff variation,
best average rank, and widest observed rank range without making causal claims. An
interactive ranking explorer can reorder strategies by payoff, variation, average
rank, first-place rate, or cooperation. Selecting a bar adds a deterministic
statement about that strategy's exact position and supporting metrics. Equal values
share a competition rank, and behavioral metrics such as cooperation are described
without claiming that higher behavior is inherently better.

The Evolution generation inspector also includes an interactive population ranking.
Moving the generation slider redraws the ranking and produces a deterministic,
ordered statement from the population shares already stored in the result artifact;
it never reruns the evolution model.

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
| `/health` | Lightweight service health response |
| `/health/live` | Process liveness response for deployment monitoring |
| `/health/ready` | Readiness response covering the strategy catalog, assets, and artifact schemas |
| `/version` | App, game, and artifact schema versions |

Unknown routes display a dedicated not-found page with links back to the main
learning and experiment areas.

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

The app includes 60 built-in strategies. They range from standard IPD baselines
to project-specific experimental policies:

- **Baselines:** MrNiceGuy (always cooperate), BadCop (always defect), ImSoRandom
- **Reciprocity:** TitForTat, SuspiciousTitForTat, GenerousTitForTat, Joss, WinStayLoseShift
- **Retaliation variants:** TitForTwoTats, TwoTitsForTat, HardTitForTat, Gradual, HoldingAGrudge, PastTrauma
- **Majority and threshold policies:** SoftMajority, HardMajority, CalculatedDefector, ForgiveButDontForget
- **Testing and adaptive heuristics:** Prober, LongTermRelationship, KeepingPeace, ParkBus, Illuminati
- **Experimental learners:** DebtCollector (repayable trust ledger), PatternHunter (transition prediction),
  EntropyBroker (recent-behavior uncertainty shield), AdaptiveBestResponse, and HedgeMetaStrategy
- **Finite punishment and recovery:** ForgetfulGrudger, Appeaser, Forgiver, and StochasticPavlov
- **Memory-one research policies:** ReactivePlayer, MemoryOnePlayer, ZDExtort2, ZDGenerous2, and ZDEqualizer
- **Recognition:** Handshake, which uses a legal C-D-C-C opening sequence to identify matching copies
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

The Render blueprint uses `/health/ready` for deployment health checks and declares
the production environment and Python runtime. `/version` also reports the public
environment label, commit SHA when supplied by the host, and deployment timestamp.
Security and privacy reporting guidance is available in `SECURITY.md` and
`PRIVACY.md`. The PayPal SDK loads only when the donation page is opened.
The web process also applies a request-size ceiling, request identifiers, and a
small per-process submission limit for robustness and evolution callbacks. This is
an initial public-preview safeguard; a shared Redis-backed limiter and background
worker remain required before increasing public experiment workloads.

CI also runs Playwright smoke tests against a real Gunicorn-hosted Dash process in
desktop and mobile Chromium viewports. The suite covers public routes, Learn page
interaction, the 60-strategy landscape, primary experiment modes, service health,
and response security headers. Failure screenshots, video, traces, and the HTML
report are retained as workflow artifacts.

To run the browser suite locally after installing Node.js:

```bash
npm install --ignore-scripts
npx playwright install chromium
npm run test:e2e
```

Playwright starts a one-worker Gunicorn process automatically unless
`E2E_BASE_URL` points to an existing staging deployment.

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
- Runtime and development dependencies are pinned in `requirements.txt` and
  `requirements-dev.txt` so local, CI, and hosted installations use the same versions.
- Core simulation logic lives in `pages/game_logic.py`; reusable tournament analytics
  live in `pages/analytics.py`; and Dash-independent robustness and evolution result
  presentation lives in `pages/experiment_views.py`. The Dash UI and callback
  registration remain in `pages/app.py` while the larger layout/callback refactor
  continues incrementally.
