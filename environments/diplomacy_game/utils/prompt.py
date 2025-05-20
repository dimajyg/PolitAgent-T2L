import json
from typing import Dict, Any, List

# Role-specific prompts
def get_diplomacy_role_prompt(power_name: str) -> str:
    """Prompt to set the role of a Diplomacy power."""
    return f"""You are playing the game of Diplomacy as {power_name}. 
    
Your goal is to strategically negotiate and make decisions to increase your influence and win the game. You will interact with other powers (Austria, England, France, Germany, Italy, Russia, Turkey) through natural language negotiation.

Focus on diplomacy, strategy, and achieving your objectives as {power_name}. Be cunning but also willing to form alliances when necessary. Always think a few steps ahead and consider the motivations of other players.

Remember that in Diplomacy, trust is a valuable currency, but deception can sometimes be necessary for victory."""

def get_negotiation_prompt(current_power_name: str, opponent_power_name: str, game_state: Dict[str, Any]) -> str:
    """Prompt to guide negotiation between two powers."""
    game_state_str = json.dumps(game_state, indent=2)
    
    return f"""You are {current_power_name} negotiating with {opponent_power_name}.

Current game state:
{game_state_str}

Consider your current position, your goals, and the game state when crafting your negotiation messages. 
Think about what you want to achieve from this negotiation with {opponent_power_name}.

You may want to:
- Propose alliances or non-aggression pacts
- Request or offer support for specific moves
- Share (true or false) intelligence about other powers
- Make territorial agreements

Be strategic in your communication. Remember that in Diplomacy, words are as important as actions."""

def get_strategic_decision_prompt(power_name: str, game_state: Dict[str, Any]) -> str:
    """Prompt to guide strategic decision making."""
    game_state_str = json.dumps(game_state, indent=2)
    
    return f"""You are {power_name}.

Current game state:
{game_state_str}

Based on the current situation and your past negotiations, decide on your strategic direction for this turn.

Consider:
- Your current position and unit placement
- Potential alliances and threats
- Opportunities for expansion
- Areas that need defense
- The overall balance of power

Formulate a coherent strategy that will guide your specific orders."""

def get_orders_prompt(power_name: str, game_state: Dict[str, Any]) -> str:
    """Prompt to guide order generation."""
    game_state_str = json.dumps(game_state, indent=2)
    
    return f"""You are {power_name}.

Current game state:
{game_state_str}

Based on your strategic assessment and the current game state, formulate your orders for this turn.

Your orders should:
- Advance your strategic goals
- Be valid according to Diplomacy rules
- Take into account expected moves from allies and enemies

IMPORTANT: You MUST provide your final orders as a JSON list of strings, using proper Diplomacy notation.
For example: ["A VIE-TYR", "F TRI-ADR", "A BUD S A VIE-TYR"]"""

def get_game_status_prompt(game_state: Dict[str, Any]) -> str:
    """Prompt to summarize the game status for all players."""
    game_state_str = json.dumps(game_state, indent=4)
    
    return f"""Current Game Status:

{game_state_str}

Use this information to inform your negotiation and strategic decisions. Pay attention to:
- The current phase and year
- Unit positions for all powers
- Supply center ownership
- Potential opportunities and threats"""