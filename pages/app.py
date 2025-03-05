# Import necessary Dash components
import dash
from dash import dcc, html, callback, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc  # For better styling
from game_logic import game

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Define the layout of the app, which includes a link to switch between pages
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dbc.Nav([
            dbc.NavLink("Home", href="/", active="exact"),
            dbc.NavLink("Match History", href="/match_history", active="exact"),
            dbc.NavLink("Detailed Round Visualization", href="/rounds", active="exact")
        ], className="nav nav-pills"),
    ]),
    html.Div(id='page-content')
])

# Home page layout
def serve_home_page():
    return html.Div([
        html.H1("Game Theory Simulation Results"),
        html.H2("Overview of Strategies and Matchups"),
        # Include any other overview visualizations or elements here
    ])

def rounds_page():
    return html.Div([
        html.H1("Game Theory Simulation Results"),
        html.H2("Overview of Strategies and Matchups"),
        # Include any other overview visualizations or elements here
    ])

def match_history_page():
    unique_strategies = game.results['Strategy 1'].unique()
    if unique_strategies.size > 0:
        default_strategy = unique_strategies[0]
    else:
        default_strategy = None  # Handle the case where no strategies are available
    
    return html.Div([
        html.H1("Detailed Match Analysis"),
        dcc.Dropdown(
            id='strategy-selector',
            options=[{'label': s, 'value': s} for s in sorted(unique_strategies)],
            value=default_strategy
        ),
        dcc.Dropdown(
            id='opponent-selector',
            value=None  # This will be populated based on the first selector
        ),
        dcc.Graph(id='match-history-graph'),
        html.Div(id='additional-metrics')
    ])


def set_opponent_options(selected_strategy):
    opponents = game.results[game.results['Strategy 1'] == selected_strategy]['Strategy 2'].unique()
    return [{'label': opponent, 'value': opponent} for opponent in sorted(opponents)]


# Generate the detailed match history graph and additional metrics
@app.callback(
    [Output('match-history-graph', 'figure'),
    Output('additional-metrics', 'children')],
    [Input('strategy-selector', 'value'),
    Input('opponent-selector', 'value')])
def update_history_graph(selected_strategy, selected_opponent):
    filtered_data = game.results[(game.results['Strategy 1'] == selected_strategy) & (game.results['Strategy 2'] == selected_opponent)]
    # Graph code as defined previously
    figure = create_match_history_figure(filtered_data)

    # Calculate additional metrics like total points, win percentage, longest streaks
    total_points = filtered_data['Result 1'].sum()
    win_percentage = (filtered_data['Result 1'] > filtered_data['Result 2']).mean() * 100
    # Assume you have functions to calculate longest win and loss streaks
    longest_win_streak = calculate_longest_streak(filtered_data['Result 1'], 'win')
    longest_loss_streak = calculate_longest_streak(filtered_data['Result 1'], 'loss')

    metrics = html.Div([
        html.P(f"Total Points: {total_points}"),
        html.P(f"Percentage of Rounds Won: {win_percentage}%"),
        html.P(f"Longest Win Streak: {longest_win_streak} rounds"),
        html.P(f"Longest Loss Streak: {longest_loss_streak} rounds"),
    ])
    return figure, metrics

# Placeholder function for creating the figure
def create_match_history_figure(filtered_data):
    return go.Figure()  # Replace with actual plotting logic

# Placeholder functions for streak calculations
def calculate_longest_streak(data, result_type):
    return 0  # Replace with actual streak calculation logic


# Update the page content based on the URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/match_history':
        return match_history_page()
    elif pathname == '/':
        return serve_home_page()
    elif pathname == 'rounds':
        return rounds_page()


if __name__ == '__main__':
    app.run_server(debug=True)
