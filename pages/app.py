from __future__ import annotations

from functools import lru_cache

import dash
from dash import Input, Output, State, callback, dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from game_logic import list_strategy_names, simulate_tournament, strategy_summary


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
}


# ----------------------------
# Data helpers
# ----------------------------


@lru_cache(maxsize=16)
def get_results(rounds_per_match: int, repetitions: int, seed: int) -> pd.DataFrame:
    return simulate_tournament(rounds_per_match=rounds_per_match, repetitions=repetitions, seed=seed)


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
# App + layout
# ----------------------------


app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])


def controls_panel() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(html.Div([html.Strong("Simulation settings")])),
            dbc.CardBody(
                [
                    dbc.Label("Rounds per match"),
                    dcc.Slider(id="rounds-per-match", min=5, max=50, step=5, value=10, marks=None, tooltip={"placement": "bottom"}),
                    dbc.FormText("Higher = more stable results, slower updates."),
                    html.Hr(),
                    dbc.Label("Repetitions"),
                    dcc.Slider(id="repetitions", min=5, max=100, step=5, value=30, marks=None, tooltip={"placement": "bottom"}),
                    dbc.FormText("How many times each strategy-pair match is repeated."),
                    html.Hr(),
                    dbc.Label("Random seed"),
                    dbc.Input(id="seed", type="number", value=0, step=1),
                ]
            ),
        ],
        className="shadow-sm",
    )


def navbar() -> dbc.Nav:
    return dbc.Nav(
        [
            dbc.NavLink("Overview", href="/", active="exact"),
            dbc.NavLink("Profile Overview", href="/profiles", active="exact"),
        ],
        pills=True,
    )


app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.H2("Prisoner’s Dilemma — Analytics Dashboard"), md=8),
                dbc.Col(navbar(), md=4, className="d-flex justify-content-end align-items-center"),
            ],
            align="center",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls_panel(), md=3),
                dbc.Col(html.Div(id="page-content"), md=9),
            ],
            className="g-3",
        ),
        html.Br(),
    ],
    fluid=True,
)


# ----------------------------
# Pages
# ----------------------------


def overview_page(rounds_per_match: int, repetitions: int, seed: int) -> html.Div:
    df = get_results(rounds_per_match, repetitions, seed)
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
        style_cell={"fontFamily": "system-ui", "fontSize": 14, "padding": "8px"},
        style_header={"fontWeight": "700"},
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig), md=12),
                ]
            ),
            html.Hr(),
            html.H4("Summary table"),
            table,
        ]
    )


def profiles_page() -> html.Div:
    names = list_strategy_names()
    return html.Div(
        [
            html.H3("Profile Overview"),
            html.P("Pick a strategy to see what it does and how it performs in the current simulation settings."),
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
                ],
                className="g-2",
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(html.Div(id="profile-metadata"))), md=6),
                    dbc.Col(dbc.Card(dbc.CardBody(html.Div(id="profile-kpis"))), md=6),
                ],
                className="g-3",
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="profile-vs-opponents"), md=12),
                ],
                className="g-3",
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="profile-round-behavior"), md=12),
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
                style_cell={"fontFamily": "system-ui", "fontSize": 14, "padding": "8px"},
                style_header={"fontWeight": "700"},
            ),
        ]
    )


# ----------------------------
# Routing
# ----------------------------


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("rounds-per-match", "value"),
    Input("repetitions", "value"),
    Input("seed", "value"),
)
def display_page(pathname: str, rounds_per_match: int, repetitions: int, seed: int):
    if pathname == "/profiles":
        return profiles_page()
    # default: overview
    return overview_page(rounds_per_match, repetitions, int(seed or 0))


# ----------------------------
# Profile callbacks
# ----------------------------


@callback(
    Output("profile-metadata", "children"),
    Output("profile-kpis", "children"),
    Output("profile-vs-opponents", "figure"),
    Output("profile-round-behavior", "figure"),
    Output("profile-opponent-table", "columns"),
    Output("profile-opponent-table", "data"),
    Input("profile-strategy", "value"),
    Input("rounds-per-match", "value"),
    Input("repetitions", "value"),
    Input("seed", "value"),
)
def update_profile(strategy: str, rounds_per_match: int, repetitions: int, seed: int):
    seed = int(seed or 0)
    df = get_results(rounds_per_match, repetitions, seed)
    persp = perspective_rows(df)

    if not strategy:
        empty_fig = px.scatter(title="No strategy selected")
        return "No strategy selected.", "", empty_fig, empty_fig, [], []

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
        return meta, "", empty_fig, empty_fig, [], []

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
        .agg(cooperate_rate=("move", lambda s: (s == "cooperate").mean()), avg_points=("points", "mean"))
    )
    by_round["round"] = by_round["round"] + 1

    round_fig = px.line(
        by_round,
        x="round",
        y=["cooperate_rate", "avg_points"],
        title=f"{strategy}: behavior by round (across all opponents)",
    )
    round_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
    round_fig.update_yaxes(range=[0, 1], title_text="Rate / points (see legend)")

    columns = [{"name": c, "id": c} for c in vs.columns]
    data = vs.round(4).to_dict("records")

    return meta, kpis, vs_fig, round_fig, columns, data


if __name__ == "__main__":
    app.run(debug=True)
