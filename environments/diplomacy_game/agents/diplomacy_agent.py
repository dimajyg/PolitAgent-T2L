from typing import List, Dict, Any, Optional
from langchain_core.language_models.base import BaseLanguageModel
from environments.diplomacy_game.utils.utils import create_message
from environments.diplomacy_game.utils.prompt import get_diplomacy_role_prompt, get_negotiation_prompt, get_strategic_decision_prompt, get_orders_prompt
import json

class DiplomacyAgent:
    def __init__(self, model: BaseLanguageModel, power_name: str, powers: List[str], diplomacy_game) -> None:
        self.model = model
        self.power_name = power_name
        self.powers = powers
        self.diplomacy_game = diplomacy_game
        self.role_prompt = self.get_role_prompt()
        self.private_history = []

        role_message = create_message("system", self.role_prompt)
        self.private_history.append(role_message)

    def get_role_description(self) -> str:
        return f"A player representing {self.power_name} in the game of Diplomacy."

    def get_role_prompt(self) -> str:
        return get_diplomacy_role_prompt(self.power_name)

    def negotiate(self, opponent_power_name: str, game_state: Dict[str, Any]) -> str:
        """Initiate or continue negotiation with another power."""
        messages = self.private_history.copy()
        negotiation_context = get_negotiation_prompt(self.power_name, opponent_power_name, game_state)
        messages.append(create_message("user", negotiation_context))
        messages.append(create_message("user", 'Respond with a message to start or continue negotiation. Consider your strategic goals and the current game state.'))
        
        response = self.model.invoke(messages).content
        self.private_history.append(create_message("assistant", response))
        return response

    def respond_to_negotiation(self, opponent_power_name: str, message_from_opponent: str, game_state: Dict[str, Any]) -> str:
        """Respond to a negotiation message from another power."""
        messages = self.private_history.copy()
        messages.append(create_message("user", f"Message from {opponent_power_name}: {message_from_opponent}"))
        negotiation_context = get_negotiation_prompt(self.power_name, opponent_power_name, game_state)
        messages.append(create_message("user", f"Respond to the message from {opponent_power_name}. {negotiation_context}"))
        messages.append(create_message("user", 'Respond to the message and consider your next move in negotiation.'))
        
        response = self.model.invoke(messages).content
        self.private_history.append(create_message("assistant", response))
        return response

    def make_strategic_decision(self, game_state: Dict[str, Any]) -> str:
        """Make a strategic decision based on the game state and negotiations."""
        messages = self.private_history.copy()

        # Get more detailed game state info from diplomacy_game
        current_phase = self.diplomacy_game.phase
        current_year = self.diplomacy_game.year
        unit_positions = {power: [str(unit) for unit in self.diplomacy_game.get_units(power)] for power in self.powers}

        # Enhanced strategic decision prompt with diplomacy_game info
        strategic_decision_context = get_strategic_decision_prompt(self.power_name, game_state)
        strategic_decision_context += f"\n\n Current Phase (Diplomacy Lib): {current_phase}"
        strategic_decision_context += f"\n Current Year (Diplomacy Lib): {current_year}"
        strategic_decision_context += f"\n Unit Positions (Diplomacy Lib): {unit_positions}"

        messages.append(create_message("user", strategic_decision_context))
        messages.append(create_message("user", "Based on the game state, your negotiations, and the current game situation from the diplomacy library, what is your strategic decision for this turn? (e.g., focus on alliance with X, pressure Y, etc.)"))
        messages.append(create_message("user", 'State your strategic decision briefly.'))
        
        response = self.model.invoke(messages).content
        return response

    def get_orders(self, game_state: Dict[str, Any]) -> List[str]:
        """Generate game orders based on strategic decisions."""
        messages = self.private_history.copy()
        strategic_decision = self.make_strategic_decision(game_state)

        # Get possible orders from diplomacy_game
        possible_orders_dict = self.diplomacy_game.get_all_possible_orders()
        possible_orders_for_power = possible_orders_dict.get(self.power_name, {})
        possible_orders_str = {loc: [str(order) for order in orders] for loc, orders in possible_orders_for_power.items()}

        orders_context = get_orders_prompt(self.power_name, game_state)
        orders_context += f"\n\n Strategic Decision: {strategic_decision}"
        orders_context += f"\n Possible Orders (Diplomacy Lib - for reference): {possible_orders_str}"

        messages.append(create_message("user", orders_context))
        messages.append(create_message("user", f"Based on your strategic decision: {strategic_decision}, formulate your orders for this turn, considering the possible orders from the diplomacy library."))
        messages.append(create_message("user", 'Provide your orders as a JSON list of strings.'))

        response = self.model.invoke(messages).content
        try:
            orders = json.loads(response)
            # Order Validation
            validated_orders = []
            for order_str in orders:
                is_valid = False
                for location_orders in possible_orders_for_power.values():
                    for possible_order in location_orders:
                        if str(possible_order).lower() == order_str.lower():
                            is_valid = True
                            break
                    if is_valid:
                        break
                if is_valid:
                    validated_orders.append(order_str)
                else:
                    print(f"Warning: Invalid order '{order_str}' generated by {self.power_name}, skipping.")

            return validated_orders

        except json.JSONDecodeError:
            print(f"Error decoding orders from {self.power_name}: {response}")
            return ["Hold"]
        
    def chat(self, message: str) -> str:
        """General chat function for the agent."""
        messages = self.private_history.copy()
        messages.append(create_message("user", message))
        response = self.model.invoke(messages).content
        self.private_history.append(create_message("assistant", response))
        return response