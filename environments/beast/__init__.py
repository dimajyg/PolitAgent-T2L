"""
Beast strategic wealth game environment.

In this game, players engage in negotiations, bargaining, and voting rounds
to accumulate the most wealth. The environment uses LangChain for all prompt
templating and LLM interactions.
"""

from environments.beast.game import BeastGame
from environments.beast.agents.base_agent import BeastAgent

__all__ = ["BeastGame", "BeastAgent"]
