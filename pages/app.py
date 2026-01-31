from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
import random

import dash
from dash import Input, Output, State, callback, dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import time
from flask import g, redirect, request

try:
    # When running under gunicorn (Render) as a package import.
    from .game_logic import (
        init_human_match_state,
        init_tournament_state,
        list_strategy_names,
        simulate_tournament,
        step_human_match,
        step_tournament,
        strategy_summary,
    )
except ImportError:
    # When running locally via `python pages/app.py`.
    from game_logic import (  # type: ignore
        init_human_match_state,
        init_tournament_state,
        list_strategy_names,
        simulate_tournament,
        step_human_match,
        step_tournament,
        strategy_summary,
    )


# ----------------------------
# Strategy profiles (metadata)
# ----------------------------

STRATEGY_PROFILES: dict[str, dict[str, str]] = {
    "MrNiceGuy": {
        "description": "Always cooperates. Baseline for measuring how strategies exploit unconditional cooperation.",
        "origin": "Project baseline strategy (unconditional cooperator).",
        "notes": "Scores well against other cooperators, but is heavily exploited by defect-heavy strategies.",
    },
    "BadCop": {
        "description": "Always defects. Baseline for measuring robustness against exploitation.",
        "origin": "Project baseline strategy (unconditional defector).",
        "notes": "Often wins short-term vs cooperators; performs poorly in mutual-defection matchups.",
    },
    "TitForTat": {
        "description": "Cooperates first, then copies the opponent’s previous move.",
        "origin": "Classic strategy popularized by the Axelrod tournaments (submitted by Anatol Rapoport).",
        "notes": "Typically strong in repeated games: nice, retaliatory, forgiving, and clear.",
    },
    "ImSoRandom": {
        "description": "Randomly cooperates or defects each turn (50/50).",
        "origin": "Project baseline strategy (stochastic behavior).",
        "notes": "Useful to test whether strategies handle noise/unpredictability.",
    },
    "CalculatedDefector": {
        "description": "Cooperates unless the opponent defects “too often” (threshold-based).",
        "origin": "Project-defined heuristic.",
        "notes": "Designed to tolerate occasional defection but punish sustained defection.",
    },
    "HoldingAGrudge": {
        "description": "Cooperates until the opponent defects once, then defects forever.",
        "origin": "Project-defined grudge strategy (grim-trigger style).",
        "notes": "Very punishing; can do well against defectors but can lock into mutual defection after a single defection.",
    },
    "ForgiveButDontForget": {
        "description": "Defects if the opponent’s historical defection rate is high; otherwise cooperates.",
        "origin": "Project-defined forgiveness heuristic.",
        "notes": "More forgiving than a pure grudge; still punishes frequent defectors.",
    },
    "BadAlternator": {
        "description": "Alternates cooperate/defect each turn (C, D, C, D...).",
        "origin": "Project-defined deterministic cycle strategy.",
        "notes": "Can confuse reactive opponents; performance depends strongly on opponent’s response to alternation.",
    },
    "RitualDefection": {
        "description": "Mostly cooperates, but defects on a fixed schedule (every 5th move).",
        "origin": "Project-defined periodic strategy.",
        "notes": "Introduces predictable “ritual” defection; tests opponent retaliation/forgiveness.",
    },
    "TripleThreat": {
        "description": "Defects for a 3-turn block in a 6-turn cycle (CCC DDD repeating).",
        "origin": "Project-defined cyclic strategy.",
        "notes": "Creates sustained defection bursts; can trigger long retaliation cycles in grudge-like opponents.",
    },
    "Pushover": {
        "description": "Starts responsive, then eventually gives in and cooperates regardless of the opponent.",
        "origin": "Project-defined 'softening' strategy.",
        "notes": "Can reduce long retaliation cycles, but risks being exploited late in the match.",
    },
    "Thief": {
        "description": "Builds cooperation early, then shifts behavior later to try to take advantage.",
        "origin": "Project-defined 'phase shift' strategy.",
        "notes": "Useful for studying end-game betrayal and how retaliation-based opponents react.",
    },
    "Pattern": {
        "description": "Repeats a fixed pattern: 3 defects, then 3 cooperates, then repeat (DDD CCC ...).",
        "origin": "Project-defined deterministic pattern strategy.",
        "notes": "Predictable by design; tests whether opponents adapt to periodic behavior. (Intentionally distinct from TripleThreat.)",
    },
    "NeverSwitchUp": {
        "description": "Randomly chooses cooperate or defect once, then sticks with it for the entire match.",
        "origin": "Project-defined commitment strategy (stochastic initialization).",
        "notes": "A controlled way to test 'committed' behavior vs reactive opponents.",
    },
    "WinStayLoseShift": {
        "description": "Repeats its last move if it was rewarded; otherwise switches (Pavlov / WSLS).",
        "origin": "Classic IPD baseline (Pavlov / Win‑Stay, Lose‑Shift).",
        "notes": "Often strong in noisy settings; can quickly return to cooperation after mutual cooperation.",
    },
    "TitForTwoTats": {
        "description": "Cooperates by default; defects only after two consecutive opponent defections.",
        "origin": "Classic forgiving TFT variant (TF2T).",
        "notes": "More forgiving than TFT; less likely to spiral into retaliation after a single defection.",
    },
    "SuspiciousTitForTat": {
        "description": "Defects on the first move, then mirrors the opponent’s previous move.",
        "origin": "Classic TFT variant (STFT).",
        "notes": "A 'hostile start' version of TFT; useful for testing strategies against early aggression.",
    },
    "GenerousTitForTat": {
        "description": "Like TFT, but sometimes forgives defections and cooperates anyway (stochastic).",
        "origin": "Classic TFT variant (GTFT).",
        "notes": "Designed to sustain cooperation under noise; introduces controlled forgiveness.",
    },
    "Joss": {
        "description": "TFT with occasional random defection ('spite') even after opponent cooperation.",
        "origin": "Classic stochastic variant of TFT (Joss).",
        "notes": "Injects unpredictability; can exploit overly trusting opponents but may reduce cooperation stability.",
    },
    "Prober": {
        "description": "Probes early (D, C, C), then exploits if the opponent never retaliates; otherwise switches to TFT.",
        "origin": "Classic 'tester' strategy (Prober).",
        "notes": "Aims to detect unconditional cooperators; otherwise behaves similarly to TFT.",
    },
    "RandomPrime": {
        "description": "Defects by default; on prime-numbered turns it plays randomly.",
        "origin": "Project-defined novelty strategy (number-based turn schedule).",
        "notes": "A mostly-defect strategy with periodic randomness tied to primes.",
    },
    "Fibonacci": {
        "description": "Starts with a random base choice; plays it on Fibonacci-numbered turns, otherwise plays the opposite.",
        "origin": "Project-defined novelty strategy (Fibonacci turn schedule).",
        "notes": "Creates structured alternation driven by the Fibonacci sequence.",
    },
    "DefectiveFriedman": {
        "description": "Defects on turns whose round number is a Friedman number; otherwise cooperates.",
        "origin": "Project-defined novelty strategy inspired by Friedman numbers.",
        "notes": "Rare, irregular defections determined by a curated Friedman set.",
    },
    "CooperativeProth": {
        "description": "Cooperates on Proth-numbered turns; otherwise defects.",
        "origin": "Project-defined novelty strategy inspired by Proth numbers.",
        "notes": "Structured cooperation tied to a number-theory predicate.",
    },
    "LongTermRelationship": {
        "description": "Cooperates when overall cooperation is high, defects when it’s low, and randomizes in the middle.",
        "origin": "Project-defined relationship-health heuristic.",
        "notes": "Uses overall cooperation rate as a proxy for 'trust' and adapts accordingly.",
    },
    "Parrot": {
        "description": "Starts random, then copies the opponent for 5 turns, then goes random for 1 turn, repeating.",
        "origin": "Project-defined periodic mimic strategy.",
        "notes": "Mostly reactive (copying) with occasional 'reset' randomness.",
    },
    "OneStepBehind": {
        "description": "Starts random, then always plays the opposite of the opponent’s previous move.",
        "origin": "Project-defined anti-mirroring strategy.",
        "notes": "Tries to 'beat' what the opponent did last round; can destabilize cooperation loops.",
    },
    "FriendlySquare": {
        "description": "Cooperates on perfect-square turns (1, 4, 9, 16, …); otherwise plays randomly.",
        "origin": "Project-defined novelty strategy (square-number schedule).",
        "notes": "Mostly random with occasional deterministic cooperation markers.",
    },
    "LosingMyMind": {
        "description": "Starts fully cooperative and becomes increasingly random each turn.",
        "origin": "Project-defined gradual-noise strategy.",
        "notes": "Models 'deteriorating consistency' over time without relying on match-length knowledge.",
    },
    "KeepingPeace": {
        "description": "Starts cooperative and tries to keep the match as close to a tie as possible in points.",
        "origin": "Project-defined 'tie-seeking' heuristic.",
        "notes": "Tracks its own and the opponent’s points and adjusts to reduce score imbalance.",
    },
    "BadJudgeOfCharacter": {
        "description": "Starts defecting; after 3 rounds it either defects forever or randomizes based on early opponent behavior.",
        "origin": "Project-defined early-judgment strategy.",
        "notes": "If the opponent defects more than cooperates in the first 3 rounds, it commits to defecting forever.",
    },
    "DefectiveDeputy": {
        "description": "Defect-leaning strategy that becomes more likely to defect each turn.",
        "origin": "Project-defined ramping defector.",
        "notes": "A gradually-hardening policy that trends toward always defecting.",
    },
    "BadDivorce": {
        "description": "Defects almost every turn, with one surprise cooperation on a random round.",
        "origin": "Project-defined endgame-flavored strategy.",
        "notes": "Defects N−1 times and cooperates once (never on the first move).",
    },
    "RandomStranger": {
        "description": "Mostly random, but defects at the end to try to 'get one over' on the opponent.",
        "origin": "Project-defined endgame betrayal strategy.",
        "notes": "Random for most of the match; defects on the final turn when the horizon is known.",
    },
    "PastTrauma": {
        "description": "Cooperates until the opponent defects 3 total times, then defects forever.",
        "origin": "Project-defined threshold grudge strategy.",
        "notes": "The three defections do not need to be consecutive.",
    },
    "MarkedMan": {
        "description": "Defects about 90% of the time, cooperates about 10% (paranoia breaks occasionally).",
        "origin": "Project-defined stochastic paranoia strategy.",
        "notes": "A fixed-probability mixture policy (mostly defect).",
    },
    "Lottery": {
        "description": "Defects throughout, then plays randomly on the final turn (if the horizon is known).",
        "origin": "Project-defined endgame gamble strategy.",
        "notes": "If match length is unknown, it behaves as an always-defect policy.",
    },
    "Shootout": {
        "description": "Cooperates on the first move, then defects every other turn.",
        "origin": "Project-defined alternating duel strategy.",
        "notes": "Creates a predictable C/D rhythm after the opening cooperate.",
    },
    "ParkBus": {
        "description": "Defects until it gets ahead on points, then cooperates forever.",
        "origin": "Project-defined 'lock-in lead' strategy.",
        "notes": "Attempts to secure an early advantage and then play defensively (cooperate) to preserve it.",
    },
    "Illuminati": {
        "description": "Classified (black box).",
        "origin": "Project-defined hidden strategy.",
        "notes": "This strategy’s logic is intentionally not disclosed in the UI.",
    },
}


# ----------------------------
# Data helpers
# ----------------------------


@lru_cache(maxsize=16)
def get_results(rounds_per_match: int, repetitions: int, seed: int, horizon_known: bool) -> pd.DataFrame:
    return simulate_tournament(
        rounds_per_match=rounds_per_match,
        repetitions=repetitions,
        seed=seed,
        horizon_known=bool(horizon_known),
    )


def perspective_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ordered (strategy_1/strategy_2) rows into per-player perspective rows:
    columns: repetition, round, strategy, opponent, move, opp_move, points, opp_points
    """

    a = df.rename(
        columns={
            "strategy_1": "strategy",
            "strategy_2": "opponent",
            "move_1": "move",
            "move_2": "opp_move",
            "points_1": "points",
            "points_2": "opp_points",
        }
    )[
        ["repetition", "round", "strategy", "opponent", "move", "opp_move", "points", "opp_points"]
    ]

    b = df.rename(
        columns={
            "strategy_2": "strategy",
            "strategy_1": "opponent",
            "move_2": "move",
            "move_1": "opp_move",
            "points_2": "points",
            "points_1": "opp_points",
        }
    )[
        ["repetition", "round", "strategy", "opponent", "move", "opp_move", "points", "opp_points"]
    ]

    out = pd.concat([a, b], ignore_index=True)
    out["strategy"] = out["strategy"].astype("string")
    out["opponent"] = out["opponent"].astype("string")
    out["move"] = out["move"].astype("string")
    out["opp_move"] = out["opp_move"].astype("string")
    return out


def match_level(persp: pd.DataFrame) -> pd.DataFrame:
    m = (
        persp.groupby(["repetition", "strategy", "opponent"], as_index=False)
        .agg(points=("points", "sum"), opp_points=("opp_points", "sum"), rounds=("points", "size"))
    )
    m["outcome"] = "tie"
    m.loc[m["points"] > m["opp_points"], "outcome"] = "win"
    m.loc[m["points"] < m["opp_points"], "outcome"] = "loss"
    return m


# ----------------------------
# Strategy classification (tags)
# ----------------------------


def strategy_scorecard(name: str) -> dict[str, object]:
    """
    Static-ish strategy classification for UI badges/scorecards.

    This is intentionally simple (human-readable, not 'perfect' taxonomy).
    """

    nm = str(name or "")
    nm = {
        "ThePushover": "Pushover",
        "TheThief": "Thief",
        "ParrotPicker": "Parrot",
        "KeepingThePeace": "KeepingPeace",
    }.get(nm, nm)

    deterministic = nm in {
        "MrNiceGuy",
        "BadCop",
        "TitForTat",
        "WinStayLoseShift",
        "TitForTwoTats",
        "SuspiciousTitForTat",
        "CalculatedDefector",
        "HoldingAGrudge",
        "ForgiveButDontForget",
        "BadAlternator",
        "RitualDefection",
        "TripleThreat",
        "Pushover",
        "Thief",
        "Pattern",
        "OneStepBehind",
        "KeepingPeace",
        "ParkBus",
        "Shootout",
        "Illuminati",
    }

    stochastic = nm in {
        "ImSoRandom",
        "NeverSwitchUp",
        "GenerousTitForTat",
        "Joss",
        "Prober",
        "RandomPrime",
        "Fibonacci",
        "LongTermRelationship",
        "Parrot",
        "FriendlySquare",
        "LosingMyMind",
        "BadJudgeOfCharacter",
        "DefectiveDeputy",
        "BadDivorce",
        "RandomStranger",
        "MarkedMan",
        "Lottery",
    }

    # Rough "memory depth": 0, 1, or "many"
    if nm in {"MrNiceGuy", "BadCop", "BadAlternator", "RitualDefection", "TripleThreat", "Pattern", "FriendlySquare", "Shootout"}:
        memory: object = 0
    elif nm in {"TitForTat", "SuspiciousTitForTat", "WinStayLoseShift", "Joss", "OneStepBehind", "Parrot"}:
        memory = 1
    elif nm in {
        "HoldingAGrudge",
        "CalculatedDefector",
        "ForgiveButDontForget",
        "TitForTwoTats",
        "Prober",
        "LongTermRelationship",
        "KeepingPeace",
        "ParkBus",
        "Illuminati",
        "BadJudgeOfCharacter",
        "PastTrauma",
    }:
        memory = "many"
    else:
        memory = "unknown"

    # Primary tendency (very rough)
    primary_coop = nm in {"MrNiceGuy", "TitForTat", "WinStayLoseShift", "TitForTwoTats", "GenerousTitForTat", "Pushover", "KeepingPeace"}
    primary_defect = nm in {"BadCop", "CalculatedDefector", "HoldingAGrudge", "Thief", "OneStepBehind", "DefectiveDeputy", "BadDivorce", "ParkBus"}

    # Uses turn counter / schedule
    time_based = nm in {
        "BadAlternator",
        "RitualDefection",
        "TripleThreat",
        "Pattern",
        "NeverSwitchUp",
        "RandomPrime",
        "Fibonacci",
        "DefectiveFriedman",
        "CooperativeProth",
        "Parrot",
        "FriendlySquare",
        "LosingMyMind",
        "Shootout",
        "BadDivorce",
        "RandomStranger",
        "Lottery",
    }

    reactive = nm in {
        "TitForTat",
        "SuspiciousTitForTat",
        "TitForTwoTats",
        "WinStayLoseShift",
        "Joss",
        "Prober",
        "CalculatedDefector",
        "HoldingAGrudge",
        "ForgiveButDontForget",
        "Pushover",
        "Parrot",
        "OneStepBehind",
        "LongTermRelationship",
        "KeepingPeace",
        "ParkBus",
        "Illuminati",
        "PastTrauma",
    }

    return {
        "deterministic": bool(deterministic and not stochastic),
        "stochastic": bool(stochastic),
        "memory": memory,
        "primarily_cooperative": bool(primary_coop),
        "primarily_defective": bool(primary_defect),
        "reactive": bool(reactive),
        "time_based": bool(time_based),
    }


# ----------------------------
# App + layout
# ----------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent
_ASSETS_DIR = _BASE_DIR / "assets"


app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=str(_ASSETS_DIR),
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[
        "https://www.paypal.com/sdk/js?client-id=BAAl7kWTxi6DEkHN3OfgGG2D1JqpQdHd22tivmtDGJ574TMPPUoXoCqg0OlGQmeDM2aS4wbzBd0emGM7As&components=hosted-buttons&enable-funding=venmo&currency=USD"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Prisoner's Dilemma Simulation"

# Use a PNG favicon (served from /assets).
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/png" href="/assets/favicon.png">
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

# Expose the underlying Flask server for Render/Gunicorn:
server = app.server

_log = logging.getLogger("pd")
if not _log.handlers:
    logging.basicConfig(level=logging.INFO)


@server.before_request
def _pd_start_timer():
    g._pd_start_time = time.perf_counter()


@server.after_request
def _pd_log_slow_requests(resp):
    start = getattr(g, "_pd_start_time", None)
    if start is None:
        return resp
    ms = (time.perf_counter() - float(start)) * 1000.0
    # Useful for quick triage in Render without adding a full profiler.
    resp.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    try:
        clen = resp.calculate_content_length()
    except Exception:
        clen = None
    if request.path.startswith("/_dash-update-component") and ms >= 800:
        _log.info("SLOW dash update: %s %s ms=%.1f bytes=%s", request.method, request.path, ms, clen)
    elif ms >= 2000:
        _log.info("SLOW request: %s %s ms=%.1f bytes=%s", request.method, request.path, ms, clen)
    return resp


def label_with_help(label: str, help_id: str, help_text: str) -> html.Div:
    """
    Consistent label + help icon + tooltip pattern.
    """

    return html.Div(
        [
            dbc.Label(label, className="mb-1"),
            html.Span("ⓘ", id=help_id, className="ms-2 muted", style={"cursor": "help", "userSelect": "none"}),
            dbc.Tooltip(help_text, target=help_id, placement="right", trigger="hover focus click"),
        ],
        className="d-flex align-items-center",
    )


def navbar_links() -> list[dbc.NavLink]:
    def link(label: str, href: str):
        return dbc.NavLink(label, href=href, active="exact", external_link=False)

    return [
        link("Overview", "/"),
        link("Profiles", "/profiles"),
        link("Experiment", "/experiment"),
    ]


pio.templates.default = "plotly_white"

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}

app.layout = html.Div(
    id="app-shell",
    className="app-shell",
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(
            id="sim-settings",
            data={"rounds_per_match": 10, "repetitions": 10, "seed": 0, "horizon_known": True},
        ),
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand(
                        [
                            html.Img(
                                src="/assets/logo.png",
                                alt="Prisoner's Dilemma Simulation logo",
                                style={"height": "30px", "width": "30px", "marginRight": "10px"},
                            ),
                            html.Span("Prisoner's Dilemma Simulation", className="fw-semibold"),
                        ],
                        href="/",
                        external_link=False,
                        className="d-flex align-items-center",
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0, className="ms-auto"),
                    dbc.Collapse(
                        dbc.Nav(
                            [
                                dbc.NavItem(dbc.NavLink("Overview", href="/", active="exact", external_link=False)),
                                dbc.NavItem(dbc.NavLink("Profiles", href="/profiles", active="exact", external_link=False)),
                                dbc.NavItem(dbc.NavLink("Experiment", href="/experiment", active="exact", external_link=False)),
                                dbc.NavItem(
                                    dbc.Button(
                                        "GitHub",
                                        href="https://github.com/leifheaney5/PrisonersDilemmaSim",
                                        target="_blank",
                                        outline=True,
                                        color="primary",
                                        className="ms-lg-3 mt-2 mt-lg-0",
                                    )
                                ),
                                dbc.NavItem(
                                    dbc.Button(
                                        "Donate",
                                        href="/donate",
                                        outline=True,
                                        color="primary",
                                        external_link=False,
                                        className="ms-lg-2 mt-2 mt-lg-0",
                                    )
                                ),
                            ],
                            className="ms-lg-2",
                            navbar=True,
                        ),
                        id="navbar-collapse",
                        is_open=False,
                        navbar=True,
                    ),
                ],
                fluid=True,
            ),
            className="app-navbar",
            dark=False,
            sticky="top",
            expand="lg",
        ),
        dbc.Container(
            [
                html.Br(),
                html.Div(html.Div(id="page-content"), id="page-wrapper", className="app-main"),
                html.Br(),
            ],
            fluid=True,
        ),
    ],
)

@callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
)
def toggle_navbar(n, is_open):
    if n:
        return not is_open
    return is_open


# Ensure deep-links like `/explore` work on Render (serve the Dash index for unknown routes).
@server.route("/<path:path>")
def catch_all(path):
    # Let Dash/Flask handle its own special routes.
    if path.startswith("_dash-") or path.startswith("assets/") or path.startswith("_favicon"):
        return ("Not Found", 404)
    if path == "explore":
        return redirect("/experiment")
    return app.index()


# ----------------------------
# Pages
# ----------------------------

def about_page() -> html.Div:
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H2("Prisoner’s Dilemma Simulation", className="mb-2"),
                                        html.P(
                                            "Explore how cooperation (and exploitation) emerges in repeated interactions. "
                                            "Run tournaments, inspect behavior round-by-round, and compare strategies under "
                                            "reproducible settings.",
                                            className="muted",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Open Experiment",
                                                        href="/experiment",
                                                        color="primary",
                                                        size="lg",
                                                        className="w-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Open Profiles",
                                                        href="/profiles",
                                                        color="secondary",
                                                        size="lg",
                                                        className="w-100",
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-2 mt-2",
                                        ),
                                    ],
                                    md=7,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("Payoff matrix (per round)", className="mb-2"),
                                                dbc.Table(
                                                    [
                                                        html.Thead(
                                                            html.Tr(
                                                                [
                                                                    html.Th("You \\ Opponent"),
                                                                    html.Th("Cooperate"),
                                                                    html.Th("Defect"),
                                                                ]
                                                            )
                                                        ),
                                                        html.Tbody(
                                                            [
                                                                html.Tr([html.Th("Cooperate"), html.Td("3 , 3"), html.Td("0 , 5")]),
                                                                html.Tr([html.Th("Defect"), html.Td("5 , 0"), html.Td("1 , 1")]),
                                                            ]
                                                        ),
                                                    ],
                                                    bordered=True,
                                                    responsive=True,
                                                    size="sm",
                                                    className="mb-0",
                                                ),
                                                html.Div(
                                                    "In a single round, defection dominates. Over many rounds, reputation and reciprocity can make cooperation viable.",
                                                    className="muted mt-2",
                                                ),
                                            ]
                                        ),
                                        className="glass-card",
                                    ),
                                    md=5,
                                ),
                            ],
                            className="g-3 align-items-stretch",
                        ),
                        html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("What you can do", className="mb-2"),
                                                html.Ul(
                                                    [
                                                        html.Li([html.Strong("Tournament (live): "), "Watch scores and win/loss trends update in real time."]),
                                                        html.Li([html.Strong("Play a match: "), "Make your own moves against any strategy and see the outcome unfold."]),
                                                        html.Li([html.Strong("Profiles & comparisons: "), "Inspect behavior patterns and head‑to‑head results."]),
                                                        html.Li([html.Strong("Build a strategy: "), "Create rule-based custom strategies and test them in tournaments."]),
                                                    ],
                                                    className="muted mb-0",
                                                ),
                                            ]
                                        ),
                                        className="glass-card",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("Key ideas to watch", className="mb-2"),
                                                html.Ul(
                                                    [
                                                        html.Li([html.Strong("Reciprocity: "), "Does a strategy reward cooperation and punish defection?"]),
                                                        html.Li([html.Strong("Forgiveness: "), "Can it return to cooperation after conflict?"]),
                                                        html.Li([html.Strong("Robustness: "), "Does it resist being exploited by defect-heavy opponents?"]),
                                                        html.Li([html.Strong("Stability: "), "Does its performance hold across seeds and settings?"]),
                                                    ],
                                                    className="muted mb-0",
                                                ),
                                            ]
                                        ),
                                        className="glass-card",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("Background (why this game matters)", className="mb-2"),
                                                html.P(
                                                    "The Prisoner’s Dilemma models a tension between short‑term incentives and long‑term outcomes. "
                                                    "In a single round, defection is the dominant action (it never does worse immediately). "
                                                    "But in repeated interactions, strategies can build (or destroy) cooperation through "
                                                    "reciprocity, punishment, forgiveness, and reputation-like dynamics.",
                                                    className="muted mb-2",
                                                ),
                                                html.P(
                                                    "This is why Iterated Prisoner’s Dilemma became a classic testbed in game theory and evolutionary thinking: "
                                                    "it’s a simple rule set that still produces rich, surprising behavior.",
                                                    className="muted mb-0",
                                                ),
                                            ]
                                        ),
                                        className="glass-card",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("How this simulator works", className="mb-2"),
                                                html.Ul(
                                                    [
                                                        html.Li(
                                                            [
                                                                html.Strong("A match"),
                                                                ": two strategies play for N rounds. Each round awards points using the payoff matrix above.",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("A tournament"),
                                                                ": every selected strategy plays every other strategy (round‑robin).",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("Repetitions"),
                                                                ": the same pairing is repeated multiple times to reduce randomness/noise and estimate typical outcomes.",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("Live mode"),
                                                                ": the app advances the tournament in small steps so charts update while it runs.",
                                                            ]
                                                        ),
                                                    ],
                                                    className="muted mb-0",
                                                ),
                                            ]
                                        ),
                                        className="glass-card",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("Settings (what they change)", className="mb-2"),
                                                html.Ul(
                                                    [
                                                        html.Li(
                                                            [
                                                                html.Strong("Rounds per match"),
                                                                ": longer matches give more time for retaliation and forgiveness to matter.",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("Repetitions"),
                                                                ": more repeats make results smoother and less dependent on luck.",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("Random seed"),
                                                                ": makes stochastic behavior reproducible. If you change the seed, you’ll often see different outcomes.",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("Known match length"),
                                                                ": when on, strategies may know the total number of rounds (enabling end‑game behavior). "
                                                                "When off, strategies don’t get that information, which can make cooperation more stable.",
                                                            ]
                                                        ),
                                                    ],
                                                    className="muted mb-0",
                                                ),
                                            ]
                                        ),
                                        className="glass-card",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("How to interpret results", className="mb-2"),
                                                html.Ul(
                                                    [
                                                        html.Li(
                                                            [
                                                                html.Strong("Total points ≠ 'niceness'"),
                                                                ": defect-heavy strategies can score well in some fields, but often collapse into low mutual scores.",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("Win/loss vs points"),
                                                                ": a strategy can win many matches but still have a lower average payoff (or vice versa).",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("Stability"),
                                                                ": try multiple seeds and settings. A strategy that only wins under one seed is less convincing.",
                                                            ]
                                                        ),
                                                        html.Li(
                                                            [
                                                                html.Strong("Head‑to‑head"),
                                                                ": use Profiles to see who a strategy beats/loses to and whether it changes behavior over time.",
                                                            ]
                                                        ),
                                                    ],
                                                    className="muted mb-0",
                                                ),
                                            ]
                                        ),
                                        className="glass-card",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Br(),
                        html.H5("Resources"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Strong("Stanford Encyclopedia of Philosophy: "),
                                        html.A(
                                            "Prisoner’s Dilemma",
                                            href="https://plato.stanford.edu/entries/prisoner-dilemma/",
                                            target="_blank",
                                        ),
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Axelrod tournaments (Iterated PD): "),
                                        html.A(
                                            "Axelrod-Python tournament library",
                                            href="https://axelrod.readthedocs.io/en/stable/",
                                            target="_blank",
                                        ),
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Axelrod (Science, 1981): "),
                                        html.A(
                                            "The Evolution of Cooperation",
                                            href="https://doi.org/10.1126/science.7466396",
                                            target="_blank",
                                        ),
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Press & Dyson (PNAS, 2012): "),
                                        html.A(
                                            "Iterated Prisoner’s Dilemma contains strategies that dominate any evolutionary opponent",
                                            href="https://doi.org/10.1073/pnas.1206569109",
                                            target="_blank",
                                        ),
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Nowak (Science, 2006): "),
                                        html.A(
                                            "Five Rules for the Evolution of Cooperation",
                                            href="https://doi.org/10.1126/science.1133755",
                                            target="_blank",
                                        ),
                                    ]
                                ),
                            ],
                            className="muted",
                        ),
                    ]
                ),
                className="glass-card",
            )
        ]
    )


def donate_page() -> html.Div:
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H2("Support the project"),
                        html.P(
                            "If you find this simulator useful, you can support development via PayPal / Venmo.",
                            className="muted",
                        ),
                        html.Hr(),
                        html.Div(
                            [
                                html.Div(
                                    id="paypal-container-YCXC33LAEKQ78",
                                    style={"minHeight": "60px"},
                                )
                            ]
                        ),
                        html.Hr(),
                        html.P(
                            "Thank you — your support helps me keep improving the simulator and adding new features.",
                            className="muted",
                        ),
                    ]
                ),
                className="glass-card",
            )
        ]
    )


def explore_page(rounds_per_match: int, repetitions: int, seed: int, horizon_known: bool) -> html.Div:
    df = get_results(rounds_per_match, repetitions, seed, horizon_known)
    summary = strategy_summary(df)

    fig = px.bar(
        summary.sort_values("total_points", ascending=True),
        x="total_points",
        y="strategy",
        orientation="h",
        title="Total points by strategy (across all roles)",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=450)

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in summary.columns],
        data=summary.round(4).to_dict("records"),
        page_size=10,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "system-ui",
            "fontSize": 14,
            "padding": "8px",
            "backgroundColor": "var(--card-bg)",
            "color": "var(--app-text)",
            "border": "1px solid var(--card-border)",
        },
        style_header={
            "fontWeight": "700",
            "backgroundColor": "var(--card-bg)",
            "color": "var(--app-text)",
            "border": "1px solid var(--card-border)",
        },
    )

    return html.Div(
        [
            dcc.Download(id="explore-download"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="explore-total-points-fig", figure=fig), md=9),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Export", className="mb-2"),
                                        html.P("Download results or figures.", className="muted"),
                                        dbc.Button("Summary CSV", id="explore-export-csv", color="primary", outline=True, className="w-100 mb-2"),
                                        dbc.Button("Figure PNG", id="explore-export-png", color="primary", outline=True, className="w-100 mb-2"),
                                        dbc.Button("Figure PDF", id="explore-export-pdf", color="primary", outline=True, className="w-100"),
                                    ]
                                ),
                                className="glass-card",
                            )
                        ],
                        md=3,
                    ),
                ],
                className="g-3",
            ),
            html.Hr(),
            html.H4("Summary table"),
            table,
            html.Hr(),
            html.H4("Strategy similarity (feature distance)"),
            html.P(
                "A quick comparison based on behavioral features (cooperate rate, conditional cooperation, switch rate, and points). "
                "Lower distance means more similar behavior in this run.",
                className="muted",
            ),
            dcc.Graph(figure=_strategy_similarity_heatmap(df)),
        ]
    )


def _strategy_similarity_heatmap(df: pd.DataFrame):
    """
    Compute a simple feature vector per strategy and render a distance matrix heatmap.
    """

    persp = perspective_rows(df)
    if persp.empty:
        fig = px.imshow([[0]], text_auto=True, title="Similarity heatmap")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # Overall cooperation rate
    features = persp.groupby("strategy", as_index=False).agg(
        cooperate_rate=("move", lambda s: float((s == "cooperate").mean())),
        avg_points=("points", lambda s: float(s.mean())),
    )

    # Conditional cooperation
    coop_vs_def = (
        persp[persp["opp_move"] == "defect"]
        .groupby("strategy", as_index=False)
        .agg(coop_after_opp_defect=("move", lambda s: float((s == "cooperate").mean())))
    )
    coop_vs_coop = (
        persp[persp["opp_move"] == "cooperate"]
        .groupby("strategy", as_index=False)
        .agg(coop_after_opp_cooperate=("move", lambda s: float((s == "cooperate").mean())))
    )
    features = features.merge(coop_vs_def, on="strategy", how="left").merge(coop_vs_coop, on="strategy", how="left")
    features["coop_after_opp_defect"] = features["coop_after_opp_defect"].fillna(0.0)
    features["coop_after_opp_cooperate"] = features["coop_after_opp_cooperate"].fillna(0.0)

    # Switch rate within each (repetition, strategy, opponent) sequence
    def _switch_rate(group: pd.DataFrame) -> float:
        g = group.sort_values("round")
        if len(g) <= 1:
            return 0.0
        return float((g["move"].shift(1) != g["move"]).iloc[1:].mean())

    switch = (
        persp.groupby(["repetition", "strategy", "opponent"], as_index=False)
        .apply(_switch_rate)
        .rename(columns={None: "switch_rate"})
    )
    switch = switch.groupby("strategy", as_index=False).agg(switch_rate=("switch_rate", "mean"))
    features = features.merge(switch, on="strategy", how="left")
    features["switch_rate"] = features["switch_rate"].fillna(0.0)

    # Distance matrix (euclidean on normalized features)
    feats = features.set_index("strategy")[["cooperate_rate", "coop_after_opp_defect", "coop_after_opp_cooperate", "switch_rate", "avg_points"]]
    # normalize each column to [0,1] when possible
    norm = feats.copy()
    for col in norm.columns:
        mn = float(norm[col].min())
        mx = float(norm[col].max())
        if mx > mn:
            norm[col] = (norm[col] - mn) / (mx - mn)
        else:
            norm[col] = 0.0

    strategies = list(norm.index)
    mat = []
    for a in strategies:
        row = []
        va = norm.loc[a].to_list()
        for b in strategies:
            vb = norm.loc[b].to_list()
            d = sum((float(x) - float(y)) ** 2 for x, y in zip(va, vb)) ** 0.5
            row.append(d)
        mat.append(row)

    fig = px.imshow(
        mat,
        x=strategies,
        y=strategies,
        color_continuous_scale="Blues",
        title="Behavior distance matrix",
        aspect="auto",
    )
    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def profiles_page() -> html.Div:
    names = list_strategy_names()
    return html.Div(
        [
            html.H3("Profile Overview"),
            html.P("Pick a strategy to see what it does and how it performs in the current simulation settings."),
            dcc.Download(id="profile-download"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Strategy"),
                            dcc.Dropdown(
                                id="profile-strategy",
                                options=[{"label": n, "value": n} for n in names],
                                value=names[0] if names else None,
                                clearable=False,
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Export"),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("CSV", id="profile-export-csv", outline=True, color="primary"),
                                    dbc.Button("PNG", id="profile-export-png", outline=True, color="primary"),
                                    dbc.Button("PDF", id="profile-export-pdf", outline=True, color="primary"),
                                ],
                                size="sm",
                            ),
                        ],
                        md=6,
                        className="d-flex align-items-end justify-content-end",
                    ),
                ],
                className="g-2",
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(html.Div(id="profile-metadata")), className="glass-card"), md=4),
                    dbc.Col(dbc.Card(dbc.CardBody(html.Div(id="profile-kpis")), className="glass-card"), md=4),
                    dbc.Col(dbc.Card(dbc.CardBody(html.Div(id="profile-scorecard")), className="glass-card"), md=4),
                ],
                className="g-3",
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="profile-vs-opponents", config=GRAPH_CONFIG), md=12),
                ],
                className="g-3",
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="profile-round-behavior", config=GRAPH_CONFIG), md=12),
                ],
                className="g-3",
            ),
            html.Br(),
            html.H4("Opponent breakdown"),
            dash_table.DataTable(
                id="profile-opponent-table",
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "system-ui",
                    "fontSize": 14,
                    "padding": "8px",
                    "backgroundColor": "var(--card-bg)",
                    "color": "var(--app-text)",
                    "border": "1px solid var(--card-border)",
                },
                style_header={
                    "fontWeight": "700",
                    "backgroundColor": "var(--card-bg)",
                    "color": "var(--app-text)",
                    "border": "1px solid var(--card-border)",
                },
            ),
            html.Hr(),
            html.H3("Compare two strategies"),
            html.P("Compare head-to-head results and behaviors side-by-side.", className="muted"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Strategy A"),
                            dcc.Dropdown(
                                id="compare-a",
                                options=[{"label": n, "value": n} for n in names],
                                value=names[0] if names else None,
                                clearable=False,
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Strategy B"),
                            dcc.Dropdown(
                                id="compare-b",
                                options=[{"label": n, "value": n} for n in names],
                                value=names[1] if len(names) > 1 else (names[0] if names else None),
                                clearable=False,
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="g-2",
            ),
            html.Br(),
            dbc.Row([dbc.Col(dcc.Graph(id="compare-headtohead", config=GRAPH_CONFIG), md=12)], className="g-3"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="compare-a-vs-opponents", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="compare-b-vs-opponents", config=GRAPH_CONFIG), md=6),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="compare-a-behavior", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="compare-b-behavior", config=GRAPH_CONFIG), md=6),
                ],
                className="g-3",
            ),
        ]
    )


def experiment_page() -> html.Div:
    names = list_strategy_names()
    default_names = names[:10]

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H3("Run an experiment (real-time)"),
                        html.P(
                            "Run tournaments incrementally (live) or play a match yourself against a strategy.",
                            className="muted",
                        ),
                        html.Hr(),
                        dcc.Store(id="tournament-state"),
                        # Slightly slower tick reduces callback pressure (especially on Render / mobile),
                        # while the backend advances multiple rounds per tick.
                        dcc.Interval(id="tournament-interval", interval=650, disabled=True),
                        dcc.Store(id="human-match-state"),
                        dcc.Store(id="custom-strategies", data=[]),
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    label="Tournament (live)",
                                    tab_id="tab-tournament",
                                    children=[
                                        html.Br(),
                                        dbc.Alert(
                                        "Select your strategies (2 minimum, 10 maximum).",
                                            color="info",
                                            className="mb-3",
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Tournament summary")),
                                                dbc.ModalBody(id="tournament-summary-body"),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        "Close",
                                                        id="tournament-summary-close",
                                                        color="secondary",
                                                        outline=True,
                                                        className="ms-auto",
                                                    )
                                                ),
                                            ],
                                            id="tournament-summary-modal",
                                            is_open=False,
                                            size="lg",
                                            centered=True,
                                            scrollable=True,
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        label_with_help(
                                                            "Strategies",
                                                            "help-tournament-strategies",
                                                            "Select up to 10 strategies for a live tournament run. "
                                                            "Reducing the number of strategies makes the live runner faster and easier to interpret.",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="tournament-strategies",
                                                            options=[{"label": n, "value": n} for n in names],
                                                            value=default_names,
                                                            multi=True,
                                                        ),
                                                        html.Div(
                                                            [
                                                                dbc.Button(
                                                                    "Random 10",
                                                                    id="tournament-random-10",
                                                                    color="secondary",
                                                                    outline=True,
                                                                    size="sm",
                                                                ),
                                                                html.Div(id="tournament-random-feedback", className="muted"),
                                                            ],
                                                            className="d-flex align-items-center gap-2 mt-2 flex-wrap",
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        label_with_help(
                                                            "Rounds per match",
                                                            "help-tournament-rounds",
                                                            "How many rounds each pair of strategies plays per match. "
                                                            "Higher values reduce randomness but increase runtime.",
                                                        ),
                                                        dbc.Input(id="tournament-rounds", type="number", value=10, min=5, max=20, step=1),
                                                    ],
                                                    md=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        label_with_help(
                                                            "Repetitions",
                                                            "help-tournament-reps",
                                                            "How many times each pairing is repeated. Higher values stabilize rankings but increase runtime.",
                                                        ),
                                                        dbc.Input(id="tournament-reps", type="number", value=10, min=1, max=30, step=1),
                                                    ],
                                                    md=3,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        label_with_help(
                                                            "Seed",
                                                            "help-tournament-seed",
                                                            "Controls randomness for reproducibility. Same seed + same settings should produce the same results.",
                                                        ),
                                                        dbc.Input(id="tournament-seed", type="number", value=0, step=1),
                                                    ],
                                                    md=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        label_with_help(
                                                            "Match length visibility",
                                                            "help-tournament-horizon-known",
                                                            "If enabled, strategies are treated as knowing the match length (N). "
                                                            "This can change endgame-style strategies that behave differently on the final turn.",
                                                        ),
                                                        dbc.Switch(
                                                            id="tournament-horizon-known",
                                                            value=True,
                                                            label="Strategies know the number of rounds",
                                                        ),
                                                    ],
                                                    md=5,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                dbc.Button("Start", id="tournament-start", color="success"),
                                                                dbc.Button("Stop", id="tournament-stop", color="secondary", outline=True),
                                                                dbc.Button("Reset", id="tournament-reset", color="danger", outline=True),
                                                            ],
                                                            className="btn-row",
                                                        ),
                                                    ],
                                                    md=9,
                                                    className="d-flex align-items-end",
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                        html.Br(),
                                        html.Div(id="tournament-status", className="muted"),
                                        dbc.Progress(id="tournament-progress", value=0, striped=True, animated=True, className="mt-2"),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(dcc.Graph(id="tournament-leaderboard", config=GRAPH_CONFIG), md=6),
                                                dbc.Col(dcc.Graph(id="tournament-points-timeline", config=GRAPH_CONFIG), md=6),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(dcc.Graph(id="tournament-move-counts", config=GRAPH_CONFIG), md=12),
                                            ],
                                            className="g-3",
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(dcc.Graph(id="tournament-wl-timeline", config=GRAPH_CONFIG), md=8),
                                                dbc.Col(dcc.Graph(id="tournament-points-pie", config=GRAPH_CONFIG), md=4),
                                            ],
                                            className="g-3",
                                        ),
                                        html.Hr(),
                                        html.H5("Recent rounds"),
                                        dash_table.DataTable(
                                            id="tournament-recent-table",
                                            page_size=10,
                                            sort_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "fontFamily": "system-ui",
                                                "fontSize": 13,
                                                "padding": "8px",
                                                "backgroundColor": "var(--card-bg)",
                                                "color": "var(--app-text)",
                                                "border": "1px solid var(--card-border)",
                                            },
                                            style_header={
                                                "fontWeight": "700",
                                                "backgroundColor": "var(--card-bg)",
                                                "color": "var(--app-text)",
                                                "border": "1px solid var(--card-border)",
                                            },
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Play a match",
                                    tab_id="tab-human",
                                    children=[
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Opponent strategy"),
                                                        dcc.Dropdown(
                                                            id="human-opponent",
                                                            options=[{"label": n, "value": n} for n in names],
                                                            value=names[0] if names else None,
                                                            clearable=False,
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Rounds"),
                                                        dbc.Input(id="human-rounds", type="number", value=10, min=1, step=1),
                                                    ],
                                                    md=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Seed"),
                                                        dbc.Input(id="human-seed", type="number", value=0, step=1),
                                                    ],
                                                    md=3,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                dbc.Button("New match", id="human-new", color="primary"),
                                                                dbc.Button("Cooperate", id="human-cooperate", color="success", disabled=True),
                                                                dbc.Button("Defect", id="human-defect", color="warning", disabled=True),
                                                                dbc.Button("Reset", id="human-reset", color="danger", outline=True),
                                                            ],
                                                            className="btn-row",
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Br(),
                                        html.Div(id="human-status", className="muted"),
                                        dbc.Row(
                                            [
                                                dbc.Col(dcc.Graph(id="human-score-graph", config=GRAPH_CONFIG), md=12),
                                            ],
                                            className="g-3",
                                        ),
                                        html.Hr(),
                                        dash_table.DataTable(
                                            id="human-events",
                                            page_size=10,
                                            sort_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "fontFamily": "system-ui",
                                                "fontSize": 13,
                                                "padding": "8px",
                                                "backgroundColor": "var(--card-bg)",
                                                "color": "var(--app-text)",
                                                "border": "1px solid var(--card-border)",
                                            },
                                            style_header={
                                                "fontWeight": "700",
                                                "backgroundColor": "var(--card-bg)",
                                                "color": "var(--app-text)",
                                                "border": "1px solid var(--card-border)",
                                            },
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Build a strategy",
                                    tab_id="tab-builder",
                                    children=[
                                        html.Br(),
                                        html.H5("Custom strategy builder"),
                                        html.P(
                                            "Create a strategy using simple rules and test it in the Tournament tab.",
                                            className="muted",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Strategy name"),
                                                        dbc.Input(id="custom-strategy-name", value="MyStrategy", type="text"),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Start move"),
                                                        dbc.RadioItems(
                                                            id="custom-start-move",
                                                            options=[
                                                                {"label": "Cooperate", "value": "cooperate"},
                                                                {"label": "Defect", "value": "defect"},
                                                            ],
                                                            value="cooperate",
                                                            inline=True,
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                            ],
                                            className="g-3",
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Rule toggles"),
                                                        dbc.Checklist(
                                                            id="custom-toggles",
                                                            options=[
                                                                {"label": "Tit-for-tat (mirror last opponent move)", "value": "tft"},
                                                                {"label": "Grudge (defect forever after any opponent defect)", "value": "grudge"},
                                                            ],
                                                            value=[],
                                                        ),
                                                    ],
                                                    md=12,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Defect-rate threshold (after minimum history)"),
                                                        dcc.Slider(
                                                            id="custom-defect-threshold",
                                                            min=0.0,
                                                            max=1.0,
                                                            step=0.05,
                                                            value=0.5,
                                                            tooltip={"placement": "bottom"},
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Minimum rounds before threshold applies"),
                                                        dcc.Slider(
                                                            id="custom-min-history",
                                                            min=0,
                                                            max=10,
                                                            step=1,
                                                            value=3,
                                                            tooltip={"placement": "bottom"},
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                            ],
                                            className="g-3",
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Endgame: always defect after turn (0 = disabled)"),
                                                        dcc.Slider(
                                                            id="custom-endgame-after",
                                                            min=0,
                                                            max=30,
                                                            step=1,
                                                            value=0,
                                                            tooltip={"placement": "bottom"},
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Noise: random flip probability"),
                                                        dcc.Slider(
                                                            id="custom-noise",
                                                            min=0.0,
                                                            max=0.2,
                                                            step=0.01,
                                                            value=0.0,
                                                            tooltip={"placement": "bottom"},
                                                        ),
                                                    ],
                                                    md=6,
                                                ),
                                            ],
                                            className="g-3",
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Button("Add strategy", id="custom-add", color="primary", className="me-2"),
                                                        dbc.Button("Clear custom strategies", id="custom-clear", color="danger", outline=True),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Br(),
                                        html.Div(id="custom-strategy-feedback", className="muted"),
                                        html.Hr(),
                                        html.H5("Custom strategies"),
                                        dash_table.DataTable(
                                            id="custom-strategy-table",
                                            page_size=10,
                                            sort_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "fontFamily": "system-ui",
                                                "fontSize": 13,
                                                "padding": "8px",
                                                "backgroundColor": "var(--card-bg)",
                                                "color": "var(--app-text)",
                                                "border": "1px solid var(--card-border)",
                                            },
                                            style_header={
                                                "fontWeight": "700",
                                                "backgroundColor": "var(--card-bg)",
                                                "color": "var(--app-text)",
                                                "border": "1px solid var(--card-border)",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                            active_tab="tab-tournament",
                        ),
                    ]
                ),
                className="glass-card",
            )
        ]
    )


# ----------------------------
# Routing
# ----------------------------


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname: str):
    if pathname == "/profiles":
        return profiles_page()
    if pathname == "/experiment":
        return experiment_page()
    if pathname == "/explore":
        # Backwards-compat deep link: redirect to Experiment.
        return html.Div([dcc.Location(id="redirect-explore", href="/experiment")])
    if pathname == "/donate":
        return donate_page()
    # default: about/overview
    return about_page()


# ----------------------------
# Profile callbacks
# ----------------------------


@callback(
    Output("profile-metadata", "children"),
    Output("profile-kpis", "children"),
    Output("profile-scorecard", "children"),
    Output("profile-vs-opponents", "figure"),
    Output("profile-round-behavior", "figure"),
    Output("profile-opponent-table", "columns"),
    Output("profile-opponent-table", "data"),
    Input("profile-strategy", "value"),
    Input("sim-settings", "data"),
)
def update_profile(strategy: str, sim_settings: dict):
    sim_settings = dict(sim_settings or {})
    rounds_per_match = int(sim_settings.get("rounds_per_match", 10) or 10)
    repetitions = int(sim_settings.get("repetitions", 10) or 10)
    seed = int(sim_settings.get("seed", 0) or 0)
    horizon_known = bool(sim_settings.get("horizon_known", True))
    df = get_results(rounds_per_match, repetitions, seed, horizon_known)
    persp = perspective_rows(df)

    if not strategy:
        empty_fig = px.scatter(title="No strategy selected")
        return "No strategy selected.", "", "", empty_fig, empty_fig, [], []

    profile = STRATEGY_PROFILES.get(strategy, {})

    meta = html.Div(
        [
            html.H4(strategy),
            html.P(profile.get("description", "No description provided yet.")),
            html.H6("Origin"),
            html.P(profile.get("origin", "Unknown / not documented.")),
            html.H6("Notes"),
            html.P(profile.get("notes", "—")),
        ]
    )

    s_rows = persp[persp["strategy"] == strategy].copy()
    if s_rows.empty:
        empty_fig = px.scatter(title="No data for selected strategy")
        return meta, "", "", empty_fig, empty_fig, [], []

    coop_rate = float((s_rows["move"] == "cooperate").mean())
    avg_points = float(s_rows["points"].mean())

    m = match_level(s_rows)
    win_rate = float((m["outcome"] == "win").mean()) if not m.empty else 0.0
    tie_rate = float((m["outcome"] == "tie").mean()) if not m.empty else 0.0

    kpis = dbc.Row(
        [
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg points / round"), html.H3(f"{avg_points:.3f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Match win rate"), html.H3(f"{win_rate*100:.1f}%")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Match tie rate"), html.H3(f"{tie_rate*100:.1f}%")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Cooperate rate"), html.H3(f"{coop_rate*100:.1f}%")]))),
        ],
        className="g-2",
    )

    # --- Scorecard (simple classification) ---
    tags = strategy_scorecard(strategy)

    primarily_coop = coop_rate >= (2.0 / 3.0)
    primarily_def = coop_rate <= (1.0 / 3.0)

    def _bubble(v: bool) -> html.Span:
        return html.Span(
            "✓" if v else "×",
            className=("score-bubble score-bubble-yes" if v else "score-bubble score-bubble-no"),
        )

    def _pill(label: str, v: bool) -> html.Div:
        return html.Div([html.Span(label, className="score-label"), _bubble(v)], className="score-pill")

    memory_label = str(tags.get("memory", "unknown"))
    scorecard = html.Div(
        [
            html.Div(
                [
                    html.H5("Scorecard", className="mb-0"),
                    dbc.Badge(f"Memory: {memory_label}", color="secondary", pill=True, className="ms-auto"),
                ],
                className="d-flex align-items-center justify-content-between mb-2",
            ),
            html.Div(
                [
                    _pill("Primarily cooperative", primarily_coop),
                    _pill("Primarily defective", primarily_def),
                    _pill("Deterministic", bool(tags.get("deterministic"))),
                    _pill("Stochastic", bool(tags.get("stochastic"))),
                    _pill("Reactive", bool(tags.get("reactive"))),
                    _pill("Time-based", bool(tags.get("time_based"))),
                ],
                className="scorecard-grid",
            ),
            html.Div("Some labels are heuristic / run-dependent.", className="muted mt-2"),
        ]
    )

    vs = (
        s_rows.groupby("opponent", as_index=False)
        .agg(
            avg_points_per_round=("points", "mean"),
            cooperate_rate=("move", lambda s: (s == "cooperate").mean()),
            rounds=("points", "size"),
        )
        .sort_values("avg_points_per_round", ascending=True)
    )

    vs_fig = px.bar(
        vs,
        x="avg_points_per_round",
        y="opponent",
        orientation="h",
        title=f"{strategy}: average points per round vs each opponent",
    )
    vs_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)

    by_round = (
        s_rows.groupby("round", as_index=False)
        .agg(cooperate_rate=("move", lambda s: float((s == "cooperate").mean())), avg_points=("points", lambda s: float(s.mean())))
    )
    by_round["round"] = by_round["round"] + 1

    # Correct visualization: cooperate_rate is [0,1], avg_points is on payoff scale.
    round_fig = make_subplots(specs=[[{"secondary_y": True}]])
    round_fig.add_trace(
        go.Scatter(x=by_round["round"], y=by_round["cooperate_rate"], name="Cooperate rate", mode="lines+markers"),
        secondary_y=False,
    )
    round_fig.add_trace(
        go.Scatter(x=by_round["round"], y=by_round["avg_points"], name="Avg points", mode="lines+markers"),
        secondary_y=True,
    )
    round_fig.update_layout(
        title=f"{strategy}: behavior by round (across all opponents)",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Metric",
    )
    round_fig.update_xaxes(title_text="Round")
    round_fig.update_yaxes(title_text="Cooperate rate", range=[0, 1], secondary_y=False)
    round_fig.update_yaxes(title_text="Avg points", secondary_y=True)

    columns = [{"name": c, "id": c} for c in vs.columns]
    data = vs.round(4).to_dict("records")

    return meta, kpis, scorecard, vs_fig, round_fig, columns, data


@callback(
    Output("compare-headtohead", "figure"),
    Output("compare-a-vs-opponents", "figure"),
    Output("compare-b-vs-opponents", "figure"),
    Output("compare-a-behavior", "figure"),
    Output("compare-b-behavior", "figure"),
    Input("compare-a", "value"),
    Input("compare-b", "value"),
    Input("sim-settings", "data"),
)
def compare_strategies(a: str, b: str, sim_settings: dict):
    sim_settings = dict(sim_settings or {})
    rounds_per_match = int(sim_settings.get("rounds_per_match", 10) or 10)
    repetitions = int(sim_settings.get("repetitions", 10) or 10)
    seed = int(sim_settings.get("seed", 0) or 0)
    horizon_known = bool(sim_settings.get("horizon_known", True))

    a = str(a or "")
    b = str(b or "")

    df = get_results(rounds_per_match, repetitions, seed, horizon_known)
    persp = perspective_rows(df)

    empty = px.scatter(title="Select two strategies to compare")
    empty.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    if not a or not b or a == b or persp.empty:
        return empty, empty, empty, empty, empty

    # --- Head-to-head (match-level) ---
    m = match_level(persp)
    ab = m[(m["strategy"] == a) & (m["opponent"] == b)].copy()
    ba = m[(m["strategy"] == b) & (m["opponent"] == a)].copy()

    def summarize_head(dfm: pd.DataFrame, label: str) -> dict:
        if dfm.empty:
            return {"pair": label, "avg_points": 0.0, "win_rate": 0.0, "loss_rate": 0.0, "tie_rate": 0.0}
        win = float((dfm["outcome"] == "win").mean())
        loss = float((dfm["outcome"] == "loss").mean())
        tie = float((dfm["outcome"] == "tie").mean())
        avg_pts = float((dfm["points"] / dfm["rounds"].where(dfm["rounds"] != 0, 1)).mean())
        return {"pair": label, "avg_points": avg_pts, "win_rate": win, "loss_rate": loss, "tie_rate": tie}

    head_rows = [
        summarize_head(ab, f"{a} vs {b}"),
        summarize_head(ba, f"{b} vs {a}"),
    ]
    head_df = pd.DataFrame(head_rows)

    head = make_subplots(rows=1, cols=2, subplot_titles=("Avg points / round", "Win / loss / tie rates"))
    head.add_trace(go.Bar(x=head_df["pair"], y=head_df["avg_points"], name="Avg points/round"), row=1, col=1)

    head.add_trace(go.Bar(x=head_df["pair"], y=head_df["win_rate"], name="Win rate"), row=1, col=2)
    head.add_trace(go.Bar(x=head_df["pair"], y=head_df["loss_rate"], name="Loss rate"), row=1, col=2)
    head.add_trace(go.Bar(x=head_df["pair"], y=head_df["tie_rate"], name="Tie rate"), row=1, col=2)

    head.update_layout(
        title=f"Head-to-head summary (seed={seed})",
        height=420,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        legend_orientation="h",
        legend_y=-0.15,
    )
    head.update_yaxes(title_text="Points/round", row=1, col=1)
    head.update_yaxes(title_text="Rate", range=[0, 1], row=1, col=2)

    # --- Vs opponents (side-by-side) ---
    def vs_opponents_fig(strategy: str) -> go.Figure:
        s_rows = persp[persp["strategy"] == strategy].copy()
        vs = (
            s_rows.groupby("opponent", as_index=False)
            .agg(avg_points_per_round=("points", "mean"), cooperate_rate=("move", lambda s: float((s == "cooperate").mean())))
            .sort_values("avg_points_per_round", ascending=True)
        )
        fig = px.bar(
            vs,
            x="avg_points_per_round",
            y="opponent",
            orientation="h",
            title=f"{strategy}: avg points/round vs opponents",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # --- Behavior by round (dual axis) ---
    def behavior_fig(strategy: str) -> go.Figure:
        s_rows = persp[persp["strategy"] == strategy].copy()
        by_round = (
            s_rows.groupby("round", as_index=False)
            .agg(cooperate_rate=("move", lambda s: float((s == "cooperate").mean())), avg_points=("points", lambda s: float(s.mean())))
        )
        by_round["round"] = by_round["round"] + 1
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=by_round["round"], y=by_round["cooperate_rate"], name="Cooperate rate", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Scatter(x=by_round["round"], y=by_round["avg_points"], name="Avg points", mode="lines+markers"), secondary_y=True)
        fig.update_layout(
            title=f"{strategy}: behavior by round",
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_orientation="h",
            legend_y=-0.18,
        )
        fig.update_xaxes(title_text="Round")
        fig.update_yaxes(title_text="Cooperate rate", range=[0, 1], secondary_y=False)
        fig.update_yaxes(title_text="Avg points", secondary_y=True)
        return fig

    return head, vs_opponents_fig(a), vs_opponents_fig(b), behavior_fig(a), behavior_fig(b)


@callback(
    Output("profile-download", "data"),
    Input("profile-export-csv", "n_clicks"),
    Input("profile-export-png", "n_clicks"),
    Input("profile-export-pdf", "n_clicks"),
    State("profile-strategy", "value"),
    State("sim-settings", "data"),
    prevent_initial_call=True,
)
def export_profile(_csv, _png, _pdf, strategy: str, sim_settings: dict):
    triggered = dash.ctx.triggered_id
    strategy = str(strategy or "strategy")
    sim_settings = dict(sim_settings or {})
    rounds_per_match = int(sim_settings.get("rounds_per_match", 10) or 10)
    repetitions = int(sim_settings.get("repetitions", 10) or 10)
    seed = int(sim_settings.get("seed", 0) or 0)
    horizon_known = bool(sim_settings.get("horizon_known", True))

    df = get_results(rounds_per_match, repetitions, seed, horizon_known)
    persp = perspective_rows(df)
    s_rows = persp[persp["strategy"] == strategy].copy()

    # Opponent breakdown (same as UI)
    vs = (
        s_rows.groupby("opponent", as_index=False)
        .agg(
            avg_points_per_round=("points", "mean"),
            cooperate_rate=("move", lambda s: float((s == "cooperate").mean())),
            rounds=("points", "size"),
        )
        .sort_values("avg_points_per_round", ascending=False)
    )

    if triggered == "profile-export-csv":
        return dcc.send_data_frame(vs.to_csv, filename=f"{strategy}_opponent_breakdown.csv", index=False)

    # Build a simple report figure (2 rows)
    report = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.18,
        subplot_titles=("Avg points/round vs opponent", "Behavior by round"),
        specs=[[{}], [{"secondary_y": True}]],
    )

    report.add_trace(
        go.Bar(x=vs["avg_points_per_round"], y=vs["opponent"], orientation="h", name="Avg points/round"),
        row=1,
        col=1,
    )

    by_round = (
        s_rows.groupby("round", as_index=False)
        .agg(
            cooperate_rate=("move", lambda s: float((s == "cooperate").mean())),
            avg_points=("points", lambda s: float(s.mean())),
        )
    )
    by_round["round"] = by_round["round"] + 1

    report.add_trace(
        go.Scatter(x=by_round["round"], y=by_round["cooperate_rate"], mode="lines+markers", name="Cooperate rate"),
        row=2,
        col=1,
        secondary_y=False,
    )
    report.add_trace(
        go.Scatter(x=by_round["round"], y=by_round["avg_points"], mode="lines+markers", name="Avg points"),
        row=2,
        col=1,
        secondary_y=True,
    )

    report.update_layout(
        title=f"Profile export — {strategy}",
        height=900,
        margin=dict(l=30, r=20, t=80, b=30),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend_title_text="",
    )
    report.update_yaxes(autorange="reversed", row=1, col=1)
    report.update_yaxes(range=[0, 1], title_text="Cooperate rate", row=2, col=1, secondary_y=False)
    report.update_yaxes(title_text="Avg points", row=2, col=1, secondary_y=True)
    report.update_xaxes(title_text="Avg points/round", row=1, col=1)
    report.update_xaxes(title_text="Round", row=2, col=1)

    if triggered in {"profile-export-png", "profile-export-pdf"}:
        fmt = "png" if triggered == "profile-export-png" else "pdf"
        try:
            img = pio.to_image(report, format=fmt, engine="kaleido")
        except Exception as e:
            # Provide a readable error to the user by downloading a text file.
            msg = f"Export failed: {e}\n\nTip: ensure 'kaleido' is installed in your environment."
            return dcc.send_string(msg, filename="export_error.txt")

        return dcc.send_bytes(lambda b: b.write(img), filename=f"{strategy}_profile_report.{fmt}")

    return dash.no_update


@callback(
    Output("explore-download", "data"),
    Input("explore-export-csv", "n_clicks"),
    Input("explore-export-png", "n_clicks"),
    Input("explore-export-pdf", "n_clicks"),
    State("sim-settings", "data"),
    prevent_initial_call=True,
)
def export_explore(_csv, _png, _pdf, sim_settings: dict):
    triggered = dash.ctx.triggered_id
    sim_settings = dict(sim_settings or {})
    rounds_per_match = int(sim_settings.get("rounds_per_match", 10) or 10)
    repetitions = int(sim_settings.get("repetitions", 10) or 10)
    seed = int(sim_settings.get("seed", 0) or 0)
    horizon_known = bool(sim_settings.get("horizon_known", True))

    df = get_results(rounds_per_match, repetitions, seed, horizon_known)
    summary = strategy_summary(df)

    if triggered == "explore-export-csv":
        return dcc.send_data_frame(summary.to_csv, filename="strategy_summary.csv", index=False)

    fig = px.bar(
        summary.sort_values("total_points", ascending=True),
        x="total_points",
        y="strategy",
        orientation="h",
        title="Total points by strategy (across all roles)",
    )
    fig.update_layout(height=600, margin=dict(l=30, r=20, t=80, b=30), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")

    if triggered in {"explore-export-png", "explore-export-pdf"}:
        fmt = "png" if triggered == "explore-export-png" else "pdf"
        try:
            img = pio.to_image(fig, format=fmt, engine="kaleido")
        except Exception as e:
            msg = f"Export failed: {e}\n\nTip: ensure 'kaleido' is installed in your environment."
            return dcc.send_string(msg, filename="export_error.txt")

        return dcc.send_bytes(lambda b: b.write(img), filename=f"total_points.{fmt}")

    return dash.no_update


# ----------------------------
# Experiment callbacks
# ----------------------------


@callback(
    Output("sim-settings", "data"),
    Input("tournament-rounds", "value"),
    Input("tournament-reps", "value"),
    Input("tournament-seed", "value"),
    Input("tournament-horizon-known", "value"),
    State("sim-settings", "data"),
)
def sync_sim_settings(rounds, reps, seed, horizon_known, current):
    current = dict(current or {})
    if rounds is not None:
        current["rounds_per_match"] = int(rounds)
    if reps is not None:
        current["repetitions"] = int(reps)
    if seed is not None:
        current["seed"] = int(seed)
    current["horizon_known"] = bool(horizon_known)
    return current


@callback(
    Output("tournament-strategies", "value"),
    Output("tournament-random-feedback", "children"),
    Input("tournament-random-10", "n_clicks"),
    Input("tournament-strategies", "value"),
    State("tournament-seed", "value"),
    prevent_initial_call=True,
)
def pick_random_strategies(n_clicks, current_value, seed):
    """
    - Clicking "Random 10" selects 10 strategies (seed-aware).
    - Manual selection is capped to 10 by trimming to the most recent 10.
    """
    triggered = dash.ctx.triggered_id

    names = list_strategy_names()
    if not names:
        return [], "No strategies available."

    # Cap manual selection to 10
    if triggered == "tournament-strategies":
        v = list(current_value or [])
        if len(v) <= 10:
            return v, ""
        trimmed = v[-10:]
        return trimmed, "Max 10 strategies — trimmed to the most recent 10 selections."

    # Random button
    k = min(10, len(names))
    base_seed = int(seed or 0)
    clicks = int(n_clicks or 0)
    rng = random.Random((base_seed + 1000003 * clicks) & 0xFFFFFFFF)
    chosen = rng.sample(names, k=k)
    return chosen, f"Selected {len(chosen)} random strategies."


@callback(
    Output("tournament-state", "data"),
    Output("tournament-interval", "disabled"),
    Output("tournament-status", "children"),
    Output("tournament-progress", "value"),
    Output("tournament-progress", "label"),
    Output("tournament-leaderboard", "figure"),
    Output("tournament-points-timeline", "figure"),
    Output("tournament-move-counts", "figure"),
    Output("tournament-wl-timeline", "figure"),
    Output("tournament-points-pie", "figure"),
    Output("tournament-summary-modal", "is_open"),
    Output("tournament-summary-body", "children"),
    Output("tournament-recent-table", "columns"),
    Output("tournament-recent-table", "data"),
    Input("tournament-start", "n_clicks"),
    Input("tournament-stop", "n_clicks"),
    Input("tournament-reset", "n_clicks"),
    Input("tournament-summary-close", "n_clicks"),
    Input("tournament-interval", "n_intervals"),
    State("tournament-strategies", "value"),
    State("tournament-rounds", "value"),
    State("tournament-reps", "value"),
    State("tournament-seed", "value"),
    State("custom-strategies", "data"),
    State("tournament-horizon-known", "value"),
    State("tournament-state", "data"),
)
def tournament_controller(_start, _stop, _reset, _close, _n, strategies, rounds, reps, seed, custom_strategies, horizon_known, state):

    def _summary_children(current_state: dict) -> html.Div:
        names = list(current_state.get("strategy_names", []))
        totals = current_state.get("totals", {}) or {}
        wins = current_state.get("match_wins", {}) or {}
        losses = current_state.get("match_losses", {}) or {}
        ties = current_state.get("match_ties", {}) or {}
        rounds_played = current_state.get("rounds_played", {}) or {}
        coop = current_state.get("cooperate", {}) or {}

        rows = []
        for s in names:
            rp = int(rounds_played.get(s, 0))
            c = int(coop.get(s, 0))
            coop_rate = (c / rp) if rp else 0.0
            w = int(wins.get(s, 0))
            l = int(losses.get(s, 0))
            t = int(ties.get(s, 0))
            matches = w + l + t
            win_pct = (w / matches) if matches else 0.0

            total_points = int(totals.get(s, 0))
            ppt = (total_points / rp) if rp else 0.0  # points per turn (turn == round played)
            ppr = (total_points / matches) if matches else 0.0  # avg points per match ("round" of opponents)
            rows.append(
                {
                    "strategy": s,
                    "total_points": total_points,
                    "wins": w,
                    "losses": l,
                    "ties": t,
                    "win_pct": win_pct,
                    "ppt": ppt,
                    "ppr": ppr,
                    "cooperate_rate": coop_rate,
                }
            )

        df = pd.DataFrame(rows).sort_values(["total_points", "wins"], ascending=False).reset_index(drop=True)
        winner = str(df.iloc[0]["strategy"]) if not df.empty else "—"
        top3 = df.head(3)["strategy"].tolist() if len(df) >= 3 else df["strategy"].tolist()

        # Highlights / tidbits
        most_coop = df.sort_values("cooperate_rate", ascending=False).head(1)
        least_coop = df.sort_values("cooperate_rate", ascending=True).head(1)
        most_wins = df.sort_values("wins", ascending=False).head(1)

        def _row_str(rdf, label, fmt):
            if rdf.empty:
                return html.Li([html.Strong(f"{label}: "), "—"])
            r = rdf.iloc[0]
            return html.Li([html.Strong(f"{label}: "), fmt(r)])

        table = dash_table.DataTable(
            columns=[
                {"name": "Strategy", "id": "strategy"},
                {"name": "Total points", "id": "total_points"},
                {"name": "Wins", "id": "wins"},
                {"name": "Losses", "id": "losses"},
                {"name": "Ties", "id": "ties"},
                {"name": "Win %", "id": "win_pct"},
                {"name": "PPT", "id": "ppt"},
                {"name": "PPR", "id": "ppr"},
                {"name": "Cooperation Rate", "id": "cooperate_rate"},
            ],
            data=df.round({"win_pct": 4, "ppt": 4, "ppr": 4, "cooperate_rate": 4}).to_dict("records"),
            sort_action="native",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={
                "fontFamily": "system-ui",
                "fontSize": 13,
                "padding": "8px",
                "backgroundColor": "var(--card-bg)",
                "color": "var(--app-text)",
                "border": "1px solid var(--card-border)",
            },
            style_header={
                "fontWeight": "700",
                "backgroundColor": "var(--card-bg)",
                "color": "var(--app-text)",
                "border": "1px solid var(--card-border)",
            },
        )

        return html.Div(
            [
                dbc.Alert(
                    [
                        html.Strong("Winner: "),
                        winner,
                        html.Span("  •  "),
                        html.Strong("Top 3: "),
                        ", ".join(top3),
                    ],
                    color="success",
                    className="mb-3",
                ),
                html.H5("Highlights"),
                html.Ul(
                    [
                        _row_str(most_wins, "Most match wins", lambda r: f"{r['strategy']} ({int(r['wins'])} wins)"),
                        _row_str(
                            most_coop,
                            "Most cooperative",
                            lambda r: f"{r['strategy']} ({float(r['cooperate_rate'])*100:.1f}%)",
                        ),
                        _row_str(
                            least_coop,
                            "Least cooperative",
                            lambda r: f"{r['strategy']} ({float(r['cooperate_rate'])*100:.1f}%)",
                        ),
                    ],
                    className="muted",
                ),
                html.Hr(),
                html.H5("Full breakdown"),
                table,
            ]
        )

    def _render(current_state: dict | None, interval_disabled: bool, status: str):
        if not current_state:
            empty = px.scatter(title="Start a tournament to see live results")
            empty.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return None, interval_disabled, status, 0, "0%", empty, empty, empty, empty, empty, False, [], [], []

        done = bool(current_state.get("done"))
        matches_done = int(current_state.get("matches_done", 0))
        total_matches = int(current_state.get("total_matches", 1)) or 1
        pct = int(round(100 * (matches_done / total_matches)))

        totals = current_state.get("totals", {})
        leaderboard = (
            pd.DataFrame([{"strategy": k, "total_points": v} for k, v in totals.items()])
            .sort_values("total_points", ascending=True)
            .reset_index(drop=True)
        )
        leaderboard_fig = px.bar(
            leaderboard,
            x="total_points",
            y="strategy",
            orientation="h",
            title="Live leaderboard (total points)",
        )
        leaderboard_fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Live counts: cooperate vs defect per strategy
        rp_map = current_state.get("rounds_played", {}) or {}
        coop_map = current_state.get("cooperate", {}) or {}
        order = leaderboard.sort_values("total_points", ascending=False)["strategy"].tolist() if not leaderboard.empty else list((current_state.get("strategy_names", []) or []))
        move_rows = []
        for s in order:
            rp = int(rp_map.get(s, 0))
            c = int(coop_map.get(s, 0))
            d = max(0, rp - c)
            move_rows.append({"strategy": s, "move": "Cooperate", "count": c})
            move_rows.append({"strategy": s, "move": "Defect", "count": d})
        moves_df = pd.DataFrame(move_rows)
        moves_fig = px.bar(
            moves_df,
            x="count",
            y="strategy",
            color="move",
            barmode="group",
            orientation="h",
            title="Live move counts (Cooperate vs Defect)",
        )
        moves_fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Move",
        )
        moves_fig.update_yaxes(autorange="reversed")

        # Build match-level timeline series (fallback to a single snapshot if timeline empty)
        timeline = list(current_state.get("timeline", []))
        if not timeline:
            timeline = [
                {
                    "matches_done": matches_done,
                    "totals": dict(current_state.get("totals", {})),
                    "match_wins": dict(current_state.get("match_wins", {})),
                    "match_losses": dict(current_state.get("match_losses", {})),
                    "match_ties": dict(current_state.get("match_ties", {})),
                }
            ]

        points_rows: list[dict] = []
        wl_rows: list[dict] = []
        for snap in timeline:
            md = int(snap.get("matches_done", 0))
            names = list(current_state.get("strategy_names", []) or [])

            # Support both snapshot formats:
            # - legacy: dicts keyed by strategy name
            # - compact: lists aligned to `strategy_names` order
            totals_snap = snap.get("totals", {}) or {}
            if isinstance(totals_snap, list):
                for s, v in zip(names, totals_snap):
                    points_rows.append({"matches_done": md, "strategy": s, "total_points": int(v)})
            else:
                for s, v in dict(totals_snap).items():
                    points_rows.append({"matches_done": md, "strategy": s, "total_points": int(v)})

            wins_snap = snap.get("match_wins", {}) or {}
            losses_snap = snap.get("match_losses", {}) or {}
            ties_snap = snap.get("match_ties", {}) or {}

            if isinstance(wins_snap, list) and isinstance(losses_snap, list) and isinstance(ties_snap, list):
                for idx, s in enumerate(names):
                    wl_rows.append(
                        {
                            "matches_done": md,
                            "strategy": s,
                            "wins": int(wins_snap[idx]) if idx < len(wins_snap) else 0,
                            "losses": int(losses_snap[idx]) if idx < len(losses_snap) else 0,
                            "ties": int(ties_snap[idx]) if idx < len(ties_snap) else 0,
                        }
                    )
            else:
                wins = dict(wins_snap) if isinstance(wins_snap, dict) else {}
                losses = dict(losses_snap) if isinstance(losses_snap, dict) else {}
                ties = dict(ties_snap) if isinstance(ties_snap, dict) else {}
                for s in set(list(wins.keys()) + list(losses.keys()) + list(ties.keys())):
                    wl_rows.append(
                        {
                            "matches_done": md,
                            "strategy": s,
                            "wins": int(wins.get(s, 0)),
                            "losses": int(losses.get(s, 0)),
                            "ties": int(ties.get(s, 0)),
                        }
                    )

        points_df = pd.DataFrame(points_rows)
        if not points_df.empty:
            points_df = points_df.sort_values(["matches_done", "strategy"]).reset_index(drop=True)
        points_timeline_fig = px.line(
            points_df,
            x="matches_done",
            y="total_points",
            color="strategy",
            title="Total points over time (matches completed)",
            markers=True,
        )
        points_timeline_fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Strategy",
        )

        wl_df = pd.DataFrame(wl_rows)
        wl_long = pd.DataFrame(columns=["matches_done", "strategy", "metric", "value"])
        if not wl_df.empty:
            wl_df = wl_df.sort_values(["matches_done", "strategy"]).reset_index(drop=True)
            wl_long = wl_df.melt(
                id_vars=["matches_done", "strategy"],
                value_vars=["wins", "losses", "ties"],
                var_name="metric",
                value_name="value",
            )
        wl_fig = px.line(
            wl_long,
            x="matches_done",
            y="value",
            color="strategy",
            facet_row="metric",
            title="Match outcomes over time (wins / losses / ties)",
            markers=True,
        )
        wl_fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Strategy",
        )
        wl_fig.for_each_annotation(lambda a: a.update(text=a.text.replace("metric=", "").title()))

        pie_df = pd.DataFrame([{"strategy": k, "total_points": int(v)} for k, v in totals.items()])
        points_pie_fig = px.pie(
            pie_df,
            names="strategy",
            values="total_points",
            title="Share of total points",
            hole=0.45,
        )
        points_pie_fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        recent_raw = list(current_state.get("recent", []))
        recent = recent_raw
        cols = []

        # Support compact recent format from game_logic:
        # [rep, i, j, round, move1, move2, points1, points2]
        if recent_raw and isinstance(recent_raw[0], (list, tuple)):
            names = list(current_state.get("strategy_names", []) or [])

            def _mv(x) -> str:
                return "cooperate" if int(x) == 0 else "defect"

            recent = []
            for r in recent_raw:
                try:
                    rep_i, i, j, rnd, m1, m2, p1, p2 = r
                    i = int(i)
                    j = int(j)
                    recent.append(
                        {
                            "rep": int(rep_i),
                            "strategy_1": names[i] if 0 <= i < len(names) else str(i),
                            "strategy_2": names[j] if 0 <= j < len(names) else str(j),
                            "round": int(rnd),
                            "move_1": _mv(m1),
                            "move_2": _mv(m2),
                            "points_1": int(p1),
                            "points_2": int(p2),
                        }
                    )
                except Exception:
                    # If something unexpected slips in, fall back to raw.
                    recent = recent_raw
                    break

        if recent and isinstance(recent[0], dict):
            cols = [{"name": c, "id": c} for c in (recent[0].keys() if recent else [])]

        label = "Done" if done else f"{pct}%"
        # If done, stop ticking automatically.
        if done:
            interval_disabled = True
            status = "Done."

        # Provide a richer status line when possible
        if current_state and not done:
            s1 = current_state.get("s1")
            s2 = current_state.get("s2")
            r = int(current_state.get("round", 0))
            rpm = int(current_state.get("rounds_per_match", 0))
            rep = int(current_state.get("rep", 0))
            reps_total = int(current_state.get("repetitions", 0))
            if bool(current_state.get("horizon_known", True)):
                status = f"{status} Match {matches_done}/{total_matches} — Rep {rep+1}/{reps_total} — {s1} vs {s2} (round {r}/{rpm})"
            else:
                status = f"{status} Match {matches_done}/{total_matches} — Rep {rep+1}/{reps_total} — {s1} vs {s2} (round {r})"

        # Modal: show summary once, when the tournament completes
        modal_open = False
        modal_body = dash.no_update
        if done and not bool(current_state.get("summary_shown")):
            current_state["summary_shown"] = True
            modal_open = True
            modal_body = _summary_children(current_state)

        return (
            current_state,
            interval_disabled,
            status,
            pct,
            label,
            leaderboard_fig,
            points_timeline_fig,
            moves_fig,
            wl_fig,
            points_pie_fig,
            modal_open,
            modal_body,
            cols,
            recent[::-1],
        )

    triggered = dash.ctx.triggered_id

    # Defaults: keep prior running state if we have it; otherwise idle.
    interval_disabled = True
    status = ""

    # If state exists and isn't done, assume it was running unless user stopped it.
    if state and not state.get("done"):
        # If interval is firing, it wasn't disabled.
        if triggered == "tournament-interval":
            interval_disabled = False

    if triggered == "tournament-start":
        try:
            chosen = list(strategies or [])
            if len(chosen) < 2:
                return _render(state, True, "Select at least 2 strategies to start.")
            # Hard cap for performance/clarity
            max_strategies = 10
            if len(chosen) > max_strategies:
                return _render(state, True, f"Select up to {max_strategies} strategies.")

            n = len(chosen)
            # Capture fewer timeline snapshots as strategy count increases to keep the
            # client/server state small (Dash stores send full JSON state each tick).
            if n <= 12:
                timeline_stride = 1
            elif n <= 20:
                timeline_stride = 2
            elif n <= 30:
                timeline_stride = 4
            else:
                timeline_stride = 6

            # Keep recent/timeline bounded to avoid huge payloads.
            # (Timeline snapshots are compact arrays aligned to strategy_names order.)
            recent_limit = 160 if n <= 20 else 120
            timeline_limit = 260 if n <= 20 else 200

            custom_map = {s["name"]: s["config"] for s in (custom_strategies or []) if isinstance(s, dict) and "name" in s and "config" in s}
            state = init_tournament_state(
                strategy_names=chosen,
                rounds_per_match=int(rounds or 10),
                repetitions=int(reps or 10),
                seed=int(seed or 0),
                horizon_known=bool(horizon_known),
                recent_limit=recent_limit,
                timeline_limit=timeline_limit,
                timeline_stride=timeline_stride,
                custom_strategies=custom_map,
            )
            # Track when we last rebuilt heavy figures/tables (for throttling).
            state["_last_render_matches_done"] = 0
        except Exception as e:
            return _render(state, True, f"Cannot start: {e}")
        interval_disabled = False
        status = "Running…"

    elif triggered == "tournament-stop":
        interval_disabled = True
        status = "Stopped."

    elif triggered == "tournament-reset":
        # close modal + clear state
        empty = px.scatter(title="Start a tournament to see live results")
        empty.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return None, True, "Reset.", 0, "0%", empty, empty, empty, empty, empty, False, [], [], []

    elif triggered == "tournament-summary-close":
        # Close the modal (keep state as-is)
        rendered = _render(state, True, status)
        # Force modal closed, keep body as-is
        return (*rendered[:10], False, dash.no_update, *rendered[12:])

    # Tick simulation if we're running and have state.
    if triggered == "tournament-interval" and state and not state.get("done"):
        # Step the simulation aggressively, but avoid re-rendering heavy figures/tables every tick.
        state = step_tournament(state, max_rounds=1400)
        interval_disabled = False

        matches_done = int(state.get("matches_done", 0))
        total_matches = int(state.get("total_matches", 1)) or 1
        pct = int(round(100 * (matches_done / total_matches)))
        label = "Done" if bool(state.get("done")) else f"{pct}%"

        last_render = int(state.get("_last_render_matches_done", 0) or 0)
        # Aim for ~40 chart refreshes max over a full run (min 2 matches between renders).
        render_delta = max(2, total_matches // 40)

        status = f"Running… Match {matches_done}/{total_matches} (charts update every ~{render_delta} matches)"

        # Force a render when:
        # - we crossed the delta threshold, or
        # - the tournament just finished (so the user sees final charts + summary modal)
        if (matches_done - last_render) < render_delta and not bool(state.get("done")):
            return (
                state,
                interval_disabled,
                status,
                pct,
                label,
                dash.no_update,  # leaderboard
                dash.no_update,  # points timeline
                dash.no_update,  # move counts
                dash.no_update,  # wl timeline
                dash.no_update,  # pie
                False,  # modal open (keep closed while throttling)
                dash.no_update,  # modal body
                dash.no_update,  # table cols
                dash.no_update,  # table data
            )

        state["_last_render_matches_done"] = matches_done

    return _render(state, interval_disabled, status)


@callback(
    Output("human-match-state", "data"),
    Output("human-cooperate", "disabled"),
    Output("human-defect", "disabled"),
    Output("human-status", "children"),
    Output("human-score-graph", "figure"),
    Output("human-events", "columns"),
    Output("human-events", "data"),
    Input("human-new", "n_clicks"),
    Input("human-cooperate", "n_clicks"),
    Input("human-defect", "n_clicks"),
    Input("human-reset", "n_clicks"),
    State("human-opponent", "value"),
    State("human-rounds", "value"),
    State("human-seed", "value"),
    State("sim-settings", "data"),
    State("custom-strategies", "data"),
    State("human-match-state", "data"),
    prevent_initial_call=True,
)
def play_human(new, coop, defect, reset, opponent, rounds, seed, sim_settings, custom_strategies, state):
    triggered = dash.ctx.triggered_id

    if triggered == "human-reset":
        empty = px.scatter(title="Start a new match to play")
        return None, True, True, "Reset.", empty, [], []

    if triggered == "human-new":
        try:
            horizon_known = bool((sim_settings or {}).get("horizon_known", True))
            custom_map = {s["name"]: s["config"] for s in (custom_strategies or []) if isinstance(s, dict) and "name" in s and "config" in s}
            state = init_human_match_state(
                opponent=str(opponent),
                rounds=int(rounds or 10),
                seed=int(seed or 0),
                horizon_known=bool(horizon_known),
                custom_strategies=custom_map,
            )
        except Exception as e:
            empty = px.scatter(title="Cannot start match")
            return None, True, True, f"Cannot start: {e}", empty, [], []

    if triggered in {"human-cooperate", "human-defect"} and state and not state.get("done"):
        move = "cooperate" if triggered == "human-cooperate" else "defect"
        state = step_human_match(state, human_move=move)

    # build UI from state
    if not state:
        empty = px.scatter(title="Start a new match to play")
        return None, True, True, "", empty, [], []

    events = list(state.get("events", []))
    human_points = int(state.get("human_points", 0))
    opp_points = int(state.get("opponent_points", 0))
    r = int(state.get("round", 0))
    total_r = int(state.get("rounds", 0))
    done = bool(state.get("done"))

    if bool(state.get("horizon_known", True)):
        status = f"Round {r}/{total_r} — You: {human_points} | {state.get('opponent')}: {opp_points}"
    else:
        status = f"Round {r} — You: {human_points} | {state.get('opponent')}: {opp_points}"
    if done:
        status += " — Finished."

    # cumulative score chart
    cum_h = []
    cum_o = []
    th = 0
    to = 0
    for e in events:
        th += int(e["human_points"])
        to += int(e["opponent_points"])
        cum_h.append({"round": int(e["round"]), "player": "You", "score": th})
        cum_o.append({"round": int(e["round"]), "player": str(state.get("opponent")), "score": to})

    # Ensure the chart renders even before the first move.
    if not events:
        score_df = pd.DataFrame(
            [
                {"round": 0, "player": "You", "score": 0},
                {"round": 0, "player": str(state.get("opponent")), "score": 0},
            ]
        )
    else:
        score_df = pd.DataFrame(cum_h + cum_o)
    fig = px.line(score_df, x="round", y="score", color="player", title="Cumulative score")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))

    cols = [{"name": c, "id": c} for c in (events[0].keys() if events else [])]
    disabled = done
    return state, disabled, disabled, status, fig, cols, events[::-1]


@callback(
    Output("custom-strategies", "data"),
    Output("custom-strategy-feedback", "children"),
    Output("custom-strategy-table", "columns"),
    Output("custom-strategy-table", "data"),
    Output("tournament-strategies", "options"),
    Output("human-opponent", "options"),
    Input("custom-add", "n_clicks"),
    Input("custom-clear", "n_clicks"),
    State("custom-strategy-name", "value"),
    State("custom-start-move", "value"),
    State("custom-toggles", "value"),
    State("custom-defect-threshold", "value"),
    State("custom-min-history", "value"),
    State("custom-endgame-after", "value"),
    State("custom-noise", "value"),
    State("custom-strategies", "data"),
    prevent_initial_call=True,
)
def manage_custom_strategies(
    _add,
    _clear,
    name,
    start_move,
    toggles,
    defect_threshold,
    min_history,
    endgame_after,
    noise,
    custom_strategies,
):
    builtins = list_strategy_names()
    custom_list = list(custom_strategies or [])

    def _options():
        opts = [{"label": n, "value": n} for n in builtins]
        for s in custom_list:
            if isinstance(s, dict) and s.get("name"):
                opts.append({"label": f"{s['name']} (custom)", "value": s["name"]})
        return opts

    def _table():
        rows = []
        for s in custom_list:
            if not isinstance(s, dict):
                continue
            cfg = s.get("config", {}) or {}
            rows.append(
                {
                    "name": s.get("name"),
                    "start_move": cfg.get("start_move"),
                    "use_tft": bool(cfg.get("use_tft")),
                    "use_grudge": bool(cfg.get("use_grudge")),
                    "defect_rate_threshold": cfg.get("defect_rate_threshold"),
                    "min_history": cfg.get("min_history"),
                    "endgame_after_turn": cfg.get("endgame_after_turn"),
                    "noise": cfg.get("noise"),
                }
            )
        cols = [{"name": c, "id": c} for c in (rows[0].keys() if rows else ["name", "start_move", "use_tft", "use_grudge", "defect_rate_threshold", "min_history", "endgame_after_turn", "noise"])]
        return cols, rows

    triggered = dash.ctx.triggered_id

    if triggered == "custom-clear":
        custom_list = []
        cols, rows = _table()
        return custom_list, "Cleared custom strategies.", cols, rows, _options(), _options()

    if triggered == "custom-add":
        nm = str(name or "").strip()
        if not nm:
            cols, rows = _table()
            return custom_list, "Name is required.", cols, rows, _options(), _options()

        if nm in builtins or any(isinstance(s, dict) and s.get("name") == nm for s in custom_list):
            cols, rows = _table()
            return custom_list, f"Strategy name '{nm}' already exists. Choose a different name.", cols, rows, _options(), _options()

        toggles = set(toggles or [])
        cfg = {
            "start_move": "cooperate" if start_move == "cooperate" else "defect",
            "use_tft": ("tft" in toggles),
            "use_grudge": ("grudge" in toggles),
            "defect_rate_threshold": float(defect_threshold if defect_threshold is not None else 1.0),
            "min_history": int(min_history or 0),
            "endgame_after_turn": int(endgame_after or 0),
            "noise": float(noise or 0.0),
        }

        custom_list.append({"name": nm, "config": cfg})
        cols, rows = _table()
        return custom_list, f"Added '{nm}'. You can now select it in Tournament / Play a match.", cols, rows, _options(), _options()

    cols, rows = _table()
    return custom_list, "", cols, rows, _options(), _options()


if __name__ == "__main__":
    app.run(debug=True)
