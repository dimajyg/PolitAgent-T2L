"""
Beast strategic wealth game environment.

In this game, players engage in negotiations, bargaining, and voting rounds
to accumulate the most wealth. The environment uses LangChain for all prompt
templating and LLM interactions.
"""

from .game import run_beast_game, run_enhanced_beast_game, EnhancedBeastGame, BeastGame
from .agents.base_agent import BeastAgent, SecretRole, ChallengeType, TrustLevel

__all__ = [
    "run_beast_game", 
    "run_enhanced_beast_game",
    "EnhancedBeastGame",
    "BeastGame",
    "BeastAgent", 
    "SecretRole", 
    "ChallengeType", 
    "TrustLevel"
]
