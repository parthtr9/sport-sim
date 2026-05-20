from __future__ import annotations

from typing import Any

from data import fetch_past_news
from model import get_actual_outcome
from simulation import generate_agents


def reward_from_probability(pred_prob: float, actual: int) -> float:
    return 1 - abs(pred_prob - actual)


def run_backtest(days: int = 10, headlines_per_day: int = 10) -> dict[str, Any]:
    headlines = fetch_past_news(days=days, headlines_per_day=headlines_per_day)
    results: list[dict[str, Any]] = []

    for headline in headlines:
        simulation_result = generate_agents(event_text=headline)
        pred_prob = float(simulation_result["model_probability"])
        predicted = 1 if pred_prob > 0.5 else 0
        actual = get_actual_outcome(headline)
        reward = reward_from_probability(pred_prob, actual)
        results.append(
            {
                "event": headline,
                "pred_prob": pred_prob,
                "prediction": predicted,
                "actual": actual,
                "reward": reward,
            }
        )

    total_events = len(results)
    correct_predictions = sum(1 for row in results if row["prediction"] == row["actual"])
    accuracy = (correct_predictions / total_events) if total_events else 0.0
    avg_reward = (
        sum(row["reward"] for row in results) / total_events
        if total_events else 0.0
    )

    return {
        "results": results,
        "accuracy": accuracy,
        "avg_reward": avg_reward,
        "total_events": total_events,
        "days": days,
        "headlines_per_day": headlines_per_day,
    }
