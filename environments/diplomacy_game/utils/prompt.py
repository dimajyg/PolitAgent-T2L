import json
from typing import Dict, Any

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

IMPORTANT: Keep your negotiation message BRIEF and CONCISE (maximum 3-4 sentences). 
Focus on one clear proposal or response.

Consider your goals and be strategic. You may:
- Propose alliances or non-aggression pacts
- Request or offer support for specific moves
- Share intelligence about other powers
- Make territorial agreements

Be direct and to the point. This is a benchmark environment - avoid long diplomatic speeches."""

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
    
    return f"""You are {power_name}.

CRITICAL: You can ONLY give orders to YOUR OWN units. Do not try to command units belonging to other powers.

Based on your strategic assessment, formulate orders for THIS TURN only.

IMPORTANT RULES:
1. You MUST give exactly ONE order per unit you control
2. Use the EXACT unit format shown in "YOUR UNITS" section 
3. Return ONLY a valid JSON array of strings
4. Each order must start with the exact unit designation (like "A VIE" or "F LON")

Common order types:
- Hold: "A VIE" (Army in Vienna holds position)
- Movement: "A VIE-TYR" (Army from Vienna to Tyrolia) 
- Support: "A BUD S A VIE-TYR" (Army in Budapest supports Vienna to Tyrolia)

REQUIRED FORMAT EXAMPLE: ["A VIE", "F TRI", "A BUD"]

Your orders:"""

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