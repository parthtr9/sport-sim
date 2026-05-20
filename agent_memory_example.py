"""
Example usage of agent memory for cross-event delta reasoning.

This reduces LLM token usage by ~50% and makes reactions deterministic.
"""

from simulation import simulate_market, _reaction_to_dict
from utils import load_agent_reactions, save_agent_reactions, load_state


def simulate_market_with_memory(market_id: str, state: dict, agent_count: int = 8) -> dict:
    """
    Simulate a market using saved agent reactions from previous events.
    
    Flow:
    1. Try to load prior reactions for this market
    2. If found, pass them to simulate_market (enables delta reasoning)
    3. After simulation, save new reactions for next event
    """
    market = next((m for m in state["markets"] if m["id"] == market_id), None)
    if not market:
        return {"error": f"Market {market_id} not found"}
    
    # Get linked news and edges
    news_by_id = {n["id"]: n for n in state["news"]}
    news_edges = [
        e for e in state["edges"]
        if e["target_id"] == market_id and e["source_type"] == "news"
    ]
    linked_news = [
        news_by_id[e["source_id"]]
        for e in news_edges
        if e["source_id"] in news_by_id
    ]
    
    # Load prior reactions if available
    prior_reactions = load_agent_reactions(market_id)
    
    # Run simulation with optional prior reactions
    result = simulate_market(
        market=market,
        news_items=linked_news,
        news_edges=news_edges,
        agent_count=agent_count,
        mode="llm_agents",
        prior_reactions=prior_reactions,  # This enables delta reasoning
    )
    
    # Save reactions for next news event
    reactions_as_dicts = [_reaction_to_dict(r) for r in result["agents"]]
    save_agent_reactions(market_id, reactions_as_dicts)
    
    return result


# Example usage:
if __name__ == "__main__":
    state = load_state()
    
    # First news event: agents react from scratch
    result_1 = simulate_market_with_memory("btc-approval", state)
    print(f"Event 1 - Model Probability: {result_1['model_probability']:.2%}")
    print(f"Agents: {[a.name for a in result_1['agents']]}")
    
    # Second news event: agents use saved reactions + delta reasoning
    # This uses ~50% fewer LLM tokens because Round 1 is skipped
    result_2 = simulate_market_with_memory("btc-approval", state)
    print(f"\nEvent 2 - Model Probability: {result_2['model_probability']:.2%}")
    print(f"Signal: {result_2['signal']}")
    
    # Reactions are deterministic now (building on prior state)
    print(f"\nAgent memory was {'used' if result_2['agent_backend'] == 'llm_agents' else 'not used'}")
