# game_logic.py

import pandas as pd
import random
import plotly.graph_objects as go

class Game:
    def __init__(self):
        self.strategies = []
        self.round_results = []

    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    def play_game(self, strategy1, strategy2):
        history1, history2 = [], []
        for _ in range(10):  # Play 10 rounds
            move1 = strategy1.play(history2)
            move2 = strategy2.play(history1)
            history1.append(move1)
            history2.append(move2)

            if move1 == 'cooperate' and move2 == 'cooperate':
                result1, result2 = 3, 3
            elif move1 == 'defect' and move2 == 'defect':
                result1, result2 = 1, 1
            elif move1 == 'cooperate' and move2 == 'defect':
                result1, result2 = 0, 5
            elif move1 == 'defect' and move2 == 'cooperate':
                result1, result2 = 5, 0
            self.round_results.append([strategy1.name, strategy2.name, result1, result2, move1, move2])

    def play_all(self):
        for _ in range(100):  # Repeat 100 times
            for i in range(len(self.strategies)):
                for j in range(i + 1, len(self.strategies)):
                    self.play_game(self.strategies[i], self.strategies[j])
        self.results = pd.DataFrame(self.round_results, columns=['Strategy 1', 'Strategy 2', 'Result 1', 'Result 2', 'Move 1', 'Move 2'])

    def save_results(self, filename):
        self.results.to_csv(filename, index=False)
        print("Results saved to", filename)

# Calculate the necessary statistics after playing all games
def summarize_results(results_df):
    results_df['Total Points 1'] = results_df['Result 1']
    results_df['Total Points 2'] = results_df['Result 2']
    
    # Summing up results
    total_points = results_df.groupby('Strategy 1')['Total Points 1'].sum().add(
        results_df.groupby('Strategy 2')['Total Points 2'].sum(), fill_value=0
    ).reset_index()
    total_points.columns = ['Strategy', 'Total Points']
    
    # Biggest wins
    biggest_wins = results_df.groupby('Strategy 1')['Result 1'].max().combine(
        results_df.groupby('Strategy 2')['Result 2'].max(), max, fill_value=0
    ).reset_index()
    biggest_wins.columns = ['Strategy', 'Biggest Win']
    
    # Biggest losses (assuming losses are where the other player wins significantly)
    biggest_losses = results_df.groupby('Strategy 1')['Result 2'].max().combine(
        results_df.groupby('Strategy 2')['Result 1'].max(), max, fill_value=0
    ).reset_index()
    biggest_losses.columns = ['Strategy', 'Biggest Loss']
    
    # Merging the dataframes
    summary_df = pd.merge(total_points, biggest_wins, on='Strategy')
    summary_df = pd.merge(summary_df, biggest_losses, on='Strategy')
    
    return summary_df

    # Example usage after game.play_all():
    

def create_match_history_figure(filtered_data):
    colors = filtered_data.apply(lambda row: 'green' if row['Move 1'] == 'cooperate' and row['Move 2'] == 'cooperate' else
                                ('yellow' if (row['Move 1'] == 'cooperate' and row['Move 2'] == 'defect') or (row['Move 1'] == 'defect' and row['Move 2'] == 'cooperate') else 'red'), axis=1)

    figure = go.Figure(
        data=go.Scatter(
            x=filtered_data.index,
            y=[1] * len(filtered_data),  # Dummy Y to keep dots aligned
            mode='markers',
            marker=dict(color=colors, size=10)
        ),
        layout=go.Layout(
            title='Match Results History',
            xaxis_title='Round',
            yaxis=dict(showticklabels=False),  # Hide Y-axis labels
            showlegend=False
        )
    )
    return figure

# Placeholder functions for streak calculations
def calculate_longest_streak(data, result_type):
    # Implement the logic to calculate the longest streaks
    return 0  # Replace with actual streak calculation logic

class Strategy:
    def __init__(self, name):
        self.name = name

    def play(self, history):
        pass  # This should be overridden by subclasses

# Redefine strategies using history for decisions
class MrNiceGuy(Strategy):
    def play(self, history):
        return 'cooperate'

class BadCop(Strategy):
    def play(self, history):
        return 'defect'

class TitForTat(Strategy):
    def play(self, history):
        if not history:
            return 'cooperate'
        return history[-1]  # Mimic the last move of the opponent

class ImSoRandom(Strategy):
    def play(self, history):
        return 'cooperate' if random.random() < 0.5 else 'defect'

class CalculatedDefector(Strategy):
    def play(self, history):
        if history.count('defect') > len(history) * 0.25:
            return 'defect'
        return 'cooperate'

class HoldingAGrudge(Strategy):
    def play(self, history):
        if 'defect' in history:
            return 'defect'
        return 'cooperate'

class ForgiveButDontForget(Strategy):
    def play(self, history):
        if history.count('defect') > len(history) * 0.5:
            return 'defect'
        return 'cooperate'

class BadAlternator(Strategy):
    def __init__(self, name):
        super().__init__(name)
        self.turn = 0

    def play(self, history):
        self.turn += 1
        return 'cooperate' if self.turn % 2 == 1 else 'defect'

class RitualDefection(Strategy):
    def __init__(self, name):
        super().__init__(name)
        self.turn = 0

    def play(self, history):
        self.turn += 1
        return 'defect' if self.turn % 5 == 0 else 'cooperate'
    
class TripleThreat(Strategy):
    def __init__(self, name):
        super().__init__(name)
        self.turn = 0

    def play(self, history):
        self.turn += 1
        cycle_position = self.turn % 6
        return 'defect' if 3 <= cycle_position < 6 else 'cooperate'
    
# Add strategies and initialize game
strategies = [MrNiceGuy("MrNiceGuy"), BadCop("BadCop"), TitForTat("TitForTat"), ImSoRandom("ImSoRandom"), CalculatedDefector("CalculatedDefector"), HoldingAGrudge("HoldingAGrudge"), ForgiveButDontForget("ForgiveButDontForget"), BadAlternator("BadAlternator"), RitualDefection("RitualDefection"), TripleThreat("TripleThreat")]

game = Game()
game.play_all()
game.save_results('results.csv')
summary_df = summarize_results(game.results)