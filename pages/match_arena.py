"""Serializable round frames for the interactive head-to-head Match Arena."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

VALID_MOVES = {"cooperate", "defect"}


def _outcome(strategy_move: str, opponent_move: str) -> str:
    if strategy_move == opponent_move == "cooperate":
        return "Mutual cooperation"
    if strategy_move == opponent_move == "defect":
        return "Mutual defection"
    if strategy_move == "defect":
        return "The selected strategy exploited its opponent"
    return "The opponent exploited the selected strategy"


def build_arena_frames(
    rows: Sequence[Mapping[str, object]],
    strategy: str,
    opponent: str,
) -> list[dict[str, object]]:
    """Normalize replay rows into JSON-safe frames for a round selector or player."""
    frames: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        strategy_move = str(row.get("strategy_move", ""))
        opponent_move = str(row.get("opponent_move", ""))
        strategy_intended = str(row.get("strategy_intended", strategy_move))
        opponent_intended = str(row.get("opponent_intended", opponent_move))
        if {strategy_move, opponent_move, strategy_intended, opponent_intended} - VALID_MOVES:
            raise ValueError("arena moves must be cooperate or defect")
        round_number = int(row.get("round", index))
        strategy_points = int(row.get("strategy_points", 0))
        opponent_points = int(row.get("opponent_points", 0))
        strategy_total = int(row.get("cumulative_strategy", strategy_points))
        opponent_total = int(row.get("cumulative_opponent", opponent_points))
        outcome = _outcome(strategy_move, opponent_move)
        strategy_error = strategy_intended != strategy_move
        opponent_error = opponent_intended != opponent_move
        error_copy = []
        if strategy_error:
            error_copy.append(f"{strategy}'s intended move was flipped during execution")
        if opponent_error:
            error_copy.append(f"{opponent}'s intended move was flipped during execution")
        statement = (
            f"Round {round_number}: {strategy} played {strategy_move} and earned {strategy_points} points; "
            f"{opponent} played {opponent_move} and earned {opponent_points} points. {outcome}. "
            f"The cumulative score is {strategy_total} to {opponent_total}."
        )
        if error_copy:
            statement += " " + "; ".join(error_copy) + "."
        frames.append(
            {
                "round": round_number,
                "strategy": str(strategy),
                "opponent": str(opponent),
                "strategy_move": strategy_move,
                "opponent_move": opponent_move,
                "strategy_intended": strategy_intended,
                "opponent_intended": opponent_intended,
                "strategy_error": strategy_error,
                "opponent_error": opponent_error,
                "strategy_points": strategy_points,
                "opponent_points": opponent_points,
                "strategy_total": strategy_total,
                "opponent_total": opponent_total,
                "outcome": outcome,
                "statement": statement,
            }
        )
    return frames


def select_arena_frame(frames: object, round_number: int | None) -> dict[str, object] | None:
    """Select and clamp a one-based arena frame from JSON-safe stored state."""
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
        return None
    normalized = [frame for frame in frames if isinstance(frame, Mapping)]
    if not normalized:
        return None
    index = max(0, min(int(round_number or 1) - 1, len(normalized) - 1))
    return dict(normalized[index])
