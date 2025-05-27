from typing import List, Dict, Any
from langchain_core.language_models.base import BaseLanguageModel
from environments.diplomacy_game.utils.utils import create_message, estimate_tokens
from environments.diplomacy_game.utils.prompt import get_diplomacy_role_prompt, get_negotiation_prompt, get_strategic_decision_prompt, get_orders_prompt
import json
import logging

logger = logging.getLogger(__name__)

class DiplomacyAgent:
    def __init__(self, model: BaseLanguageModel, power_name: str, powers: List[str], diplomacy_game) -> None:
        self.model = model
        self.power_name = power_name
        self.powers = powers
        self.diplomacy_game = diplomacy_game
        self.role_prompt = self.get_role_prompt()
        self.private_history = []
        self.max_history_length = 10
        self.max_token_count = 8000

        role_message = create_message("system", self.role_prompt)
        self.private_history.append(role_message)

    def get_role_description(self) -> str:
        return f"A player representing {self.power_name} in the game of Diplomacy."

    def get_role_prompt(self) -> str:
        return get_diplomacy_role_prompt(self.power_name)

    def _manage_history(self) -> None:
        """Manages the size of the message history to prevent exceeding the context limit."""
        if len(self.private_history) <= 2:
            return
            
        total_tokens = sum(estimate_tokens(msg["content"]) for msg in self.private_history)
        
        if total_tokens > self.max_token_count or len(self.private_history) > self.max_history_length:
            system_message = self.private_history[0]
            
            if total_tokens > self.max_token_count * 1.2:
                if len(self.private_history) > 2:
                    self.private_history = [system_message] + self.private_history[-2:]
                    logger.warning(f"{self.power_name}: История сообщений сильно усечена до {len(self.private_history)} сообщений")
                    return
            
            messages_to_keep = min(5, self.max_history_length // 2)
            self.private_history = [system_message] + self.private_history[-messages_to_keep:]
            logger.info(f"{self.power_name}: История сообщений усечена до {len(self.private_history)} сообщений")
        
    def negotiate(self, opponent_power_name: str, game_state: Dict[str, Any]) -> str:
        """Initiate or continue negotiation with another power."""
        self._manage_history()
        messages = self.private_history.copy()
        negotiation_context = get_negotiation_prompt(self.power_name, opponent_power_name, game_state)
        messages.append(create_message("user", negotiation_context))
        messages.append(create_message("user", 'Respond with a message to start or continue negotiation. Consider your strategic goals and the current game state.'))
        
        try:
            response = self.model.invoke(messages).content
        except Exception as e:
            logger.warning(f"{self.power_name}: Error in negotiation with {opponent_power_name}: {e}, using fallback response")
            response = f"I propose we maintain a neutral stance and focus on mutual security interests."
        
        self.private_history.append(create_message("assistant", response))
        return response

    def respond_to_negotiation(self, opponent_power_name: str, message_from_opponent: str, game_state: Dict[str, Any]) -> str:
        """Respond to a negotiation message from another power."""
        self._manage_history()
        messages = self.private_history.copy()
        messages.append(create_message("user", f"Message from {opponent_power_name}: {message_from_opponent}"))
        negotiation_context = get_negotiation_prompt(self.power_name, opponent_power_name, game_state)
        messages.append(create_message("user", f"Respond to the message from {opponent_power_name}. {negotiation_context}"))
        messages.append(create_message("user", 'Respond to the message and consider your next move in negotiation.'))
        
        try:
            response = self.model.invoke(messages).content
        except Exception as e:
            logger.warning(f"{self.power_name}: Error in respond_to_negotiation: {e}, using fallback response")
            response = f"I understand your position. Let's consider mutual cooperation for our shared interests."
        
        self.private_history.append(create_message("assistant", response))
        return response

    def make_strategic_decision(self, game_state: Dict[str, Any]) -> str:
        """Make a strategic decision based on the game state and negotiations."""
        self._manage_history()
        messages = self.private_history.copy()

        current_phase = game_state.get("phase", "Spring")
        current_year = game_state.get("year", 1901)
        unit_positions = {power: [str(unit) for unit in self.diplomacy_game.get_units(power)] for power in self.powers}

        strategic_decision_context = get_strategic_decision_prompt(self.power_name, game_state)
        strategic_decision_context += f"\n\n Current Phase (Diplomacy Lib): {current_phase}"
        strategic_decision_context += f"\n Current Year (Diplomacy Lib): {current_year}"
        strategic_decision_context += f"\n Unit Positions (Diplomacy Lib): {unit_positions}"

        messages.append(create_message("user", strategic_decision_context))
        messages.append(create_message("user", "Based on the game state, your negotiations, and the current game situation from the diplomacy library, what is your strategic decision for this turn? (e.g., focus on alliance with X, pressure Y, etc.)"))
        messages.append(create_message("user", 'State your strategic decision briefly.'))
        
        try:
            response = self.model.invoke(messages).content
        except Exception as e:
            logger.warning(f"{self.power_name}: Error in strategic decision: {e}, using fallback response")
            response = f"Focus on defensive positioning and maintaining current alliances."
        
        return response

    def get_orders(self, game_state: Dict[str, Any]) -> List[str]:
        """Get orders for the current phase."""
        self._manage_history()
        current_year = game_state.get("year", 1901)
        
        messages = self.private_history.copy()
        strategic_decision = self.make_strategic_decision(game_state)

        my_units = self.diplomacy_game.get_units(self.power_name)
        my_units_info = []
        
        possible_orders_dict = self.diplomacy_game.get_all_possible_orders()
        possible_orders_for_power = possible_orders_dict.get(self.power_name, {})
        
        for unit in my_units:
            unit_str = str(unit)
            unit_parts = unit_str.split()
            if len(unit_parts) >= 2:
                unit_type = unit_parts[0]
                unit_location = unit_parts[1]
                
                unit_orders = possible_orders_for_power.get(unit_location, [])
                unit_orders_str = [str(order) for order in unit_orders]
                
                my_units_info.append({
                    "unit": f"{unit_type} {unit_location}",
                    "possible_orders": unit_orders_str[:5]
                })

        orders_context = get_orders_prompt(self.power_name, game_state)
        orders_context += f"\n\n Strategic Decision: {strategic_decision}"
        orders_context += f"\n\n YOUR UNITS AND THEIR POSSIBLE ORDERS:"
        for unit_info in my_units_info:
            orders_context += f"\n - {unit_info['unit']}: {unit_info['possible_orders']}"
        orders_context += f"\n\n IMPORTANT: You can ONLY give orders to these {len(my_units_info)} units above!"

        messages.append(create_message("user", orders_context))
        messages.append(create_message("user", f"Based on your strategic decision: {strategic_decision}, formulate your orders for this turn, considering the possible orders from the diplomacy library."))
        messages.append(create_message("user", 'Provide your orders as a JSON list of strings.'))

        try:
            response = self.model.invoke(messages).content
        except Exception as e:
            logger.warning(f"{self.power_name}: Error in get_orders: {e}, using fallback orders")
            response = '["Hold"]'
        
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "").strip()
            
            orders = json.loads(cleaned_response)
            validated_orders = []
            units_with_orders = set()
            
            my_units_for_validation = self.diplomacy_game.get_units(self.power_name)
            my_unit_locations = [str(unit).split()[1] for unit in my_units_for_validation]
            
            for order_str in orders:
                order_str = order_str.strip()
                if not order_str:
                    continue

                order_parts = order_str.split()
                if len(order_parts) >= 2:
                    unit_location = order_parts[1].split('-')[0].split(' ')[0]
                    
                    if unit_location in my_unit_locations and unit_location not in units_with_orders:
                        is_valid = False
                        for location, location_orders in possible_orders_for_power.items():
                            for possible_order in location_orders:
                                if str(possible_order).lower().strip() == order_str.lower().strip():
                                    is_valid = True
                                    break
                            if is_valid:
                                break
                        
                        if is_valid:
                            validated_orders.append(order_str)
                            units_with_orders.add(unit_location)
                        else:
                            logger.warning(f"Invalid order '{order_str}' for {self.power_name}, using hold")
                            unit_type = order_parts[0] if len(order_parts) > 0 else "A"
                            hold_order = f"{unit_type} {unit_location}"
                            validated_orders.append(hold_order)
                            units_with_orders.add(unit_location)
                    elif unit_location in units_with_orders:
                        logger.warning(f"Unit {unit_location} already has an order, skipping '{order_str}'")
                    else:
                        logger.warning(f"Order '{order_str}' for {self.power_name} involves unit not owned by this power")
                else:
                    logger.warning(f"Malformed order '{order_str}' for {self.power_name}")
            
            for unit in my_units_for_validation:
                unit_str = str(unit)
                unit_parts = unit_str.split()
                if len(unit_parts) >= 2:
                    unit_location = unit_parts[1]
                    if unit_location not in units_with_orders:
                        hold_order = f"{unit_parts[0]} {unit_location}"
                        validated_orders.append(hold_order)
                        logger.info(f"Added hold order for {self.power_name} unit {unit_location}: {hold_order}")

            return validated_orders

        except json.JSONDecodeError:
            logger.error(f"Error decoding orders from {self.power_name}: {response}")
            print(f"Error decoding orders from {self.power_name}: {response}")
            return ["Hold"]
        
    def chat(self, message: str) -> str:
        """General chat function for the agent."""
        self._manage_history()
        messages = self.private_history.copy()
        messages.append(create_message("user", message))
        try:
            response = self.model.invoke(messages).content
        except Exception as e:
            logger.warning(f"{self.power_name}: Error in respond_to_negotiation: {e}, using fallback response")
            response = f"I understand your position. Let's consider mutual cooperation for our shared interests."
        
        self.private_history.append(create_message("assistant", response))
        return response