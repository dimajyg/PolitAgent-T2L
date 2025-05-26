from llm.game import Game
from environments.diplomacy_game.agents.diplomacy_agent import DiplomacyAgent
from environments.diplomacy_game.utils.utils import create_message
from environments.diplomacy_game.utils.prompt import get_game_status_prompt
import random
import json
import logging
import time
from typing import Dict, List, Any, Optional
import os
from diplomacy import Game as DiplomacyLibGame
from diplomacy.utils.export import to_saved_game_format

logger = logging.getLogger(__name__)

class DiplomacyGame(Game):
    """
    Diplomacy game implementation with LangChain models support.
    """
    
    def __init__(self, args, model) -> None:
        super().__init__(args)
        self.model = model
        self.powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        self.agents = []
        self.game_round = 0
        self.max_rounds = args.max_rounds if hasattr(args, 'max_rounds') and args.max_rounds else 10
        self.game_state = {}
        self.diplomacy_game = None
        self.log_file_path = 'diplomacy_game_log.txt'
        
        # Add debug flag
        self.debug = args.debug if hasattr(args, 'debug') else False

    def init_game(self) -> str:
        """Initialize the Diplomacy game using the 'diplomacy' library."""
        # Record start time for benchmarking
        start_time = time.time()
        
        power_names = self.powers
        random.shuffle(power_names)

        # Initialize the diplomacy library game
        self.diplomacy_game = DiplomacyLibGame(map_name="standard")

        # Initialize agents with the LangChain model
        for power_name in power_names:
            agent = DiplomacyAgent(self.model, power_name, power_names, self.diplomacy_game)
            self.agents.append(agent)

        # Initialize units using set_units
        initial_units = {
            "AUSTRIA": ["A VIE", "A BUD", "F TRI"],
            "ENGLAND": ["F LON", "F EDI", "A LVP"],
            "FRANCE": ["A PAR", "A MAR", "F BRE"],
            "GERMANY": ["A BER", "A MUN", "F KIE"],
            "ITALY": ["A ROM", "A VEN", "F NAP"],
            "RUSSIA": ["A MOS", "A WAR", "F SEV", "F STP/SC"],
            "TURKEY": ["A CON", "A SMY", "F ANK"],
        }
        for power, unit_strings in initial_units.items():
            self.diplomacy_game.set_units(power, unit_strings, reset=True)

        # Initialize supply centers
        initial_centers = {
            "AUSTRIA": ["VIE", "BUD", "TRI"],
            "ENGLAND": ["LON", "LVP", "EDI"],
            "FRANCE": ["PAR", "MAR", "BRE"],
            "GERMANY": ["BER", "MUN", "KIE"],
            "ITALY": ["ROM", "VEN", "NAP"],
            "RUSSIA": ["MOS", "WAR", "SEV", "STP"],
            "TURKEY": ["CON", "SMY", "ANK"],
        }
        for power, center_strings in initial_centers.items():
            self.diplomacy_game.set_centers(power, center_strings, reset=True)

        self._update_game_state_from_lib()

        # Log initialization time
        init_time = time.time() - start_time
        
        settings = "Diplomacy Game Initialized (using 'diplomacy' library):\n"
        settings += f"Initialization time: {init_time:.2f} seconds\n"
        for agent in self.agents:
            settings += f"{agent.power_name}: Agent created\n"
        settings += f"Game will run for max {self.max_rounds} rounds\n"
        
        if self.debug:
            logger.info(settings)
            
        return settings

    def handle_negotiation_phase(self, log_file) -> None:
        """Obsolete method, use _handle_negotiation_phase_with_metrics instead."""
        self._handle_negotiation_phase_with_metrics(log_file)

    def handle_action_phase(self, log_file) -> None:
        """Obsolete method, use _handle_action_phase_with_metrics instead."""
        self._handle_action_phase_with_metrics(log_file)

    def _update_game_state_from_lib(self) -> None:
        """Updates self.game_state from the diplomacy_game library instance."""
        # Split phase string to get phase and year
        phase_parts = str(self.diplomacy_game.phase).split()
        self.game_state["phase"] = phase_parts[0]  # Spring, Fall, etc.
        self.game_state["year"] = int(phase_parts[1]) if len(phase_parts) > 1 else 1901

        # Update supply centers - используем более компактное представление
        centers_by_power = self.diplomacy_game.get_centers()
        self.game_state["supply_centers"] = {}
        for power_name, centers in centers_by_power.items():
            self.game_state["supply_centers"][power_name] = centers

        # Update unit positions - используем более компактное представление
        self.game_state["units"] = {}
        for power_name in self.powers:
            units = [str(unit) for unit in self.diplomacy_game.get_units(power_name)]
            if units:  # Добавляем только если есть юниты
                self.game_state["units"][power_name] = units

    def update_game_state(self, orders: Dict[str, List[str]], log_file) -> None:
        """Update game state based on the diplomacy library's state."""
        self.log_message(log_file, "\nGame State Update:")
        self._update_game_state_from_lib()
        if self.debug:
            self.log_message(log_file, f"Current Game State: {json.dumps(self.game_state, indent=4)}")
        else:
            self.log_message(log_file, "Game state updated (state details omitted in non-debug mode)")

    def check_game_end(self) -> bool:
        """Check if the game has ended."""
        if self.game_round >= self.max_rounds:
            return True
            
        # Check if any power has 18 or more supply centers (victory condition)
        centers_by_power = self.diplomacy_game.get_centers()
        for power, centers in centers_by_power.items():
            if len(centers) >= 18:
                return True
                
        return False

    def log_message(self, log_file, message: str) -> None:
        """Logs a message to the specified log file."""
        if log_file:
            log_file.write(message + "\n")
        if self.debug:
            print(message)

    def game_loop(self, log_file=None) -> Dict[str, Any]:
        """Main game loop processing rounds until end conditions are met."""
        if log_file is None:
            log_file = open(self.log_file_path, 'w')
            
        # Track game start time
        game_start_time = time.time()
        
        # Добавляем данные для метрик
        rounds_data = []

        try:
            while not self.check_game_end():
                self.game_round += 1
                round_start_time = time.time()
                
                self.log_message(log_file, f"\nRound {self.game_round} begins, Year: {self.game_state.get('year', 'Unknown')}, Phase: {self.game_state.get('phase', 'Initial')}")
                
                # Сохраняем состояние перед ходом
                territories_before = self._get_territories_state()

                # Game status update for all agents
                game_status = get_game_status_prompt(self.game_state)
                status_message = create_message("user", game_status)
                for agent in self.agents:
                    agent.private_history.append(status_message)
                self.log_message(log_file, f"\nCurrent Game Status shared with all agents")
                
                # Сохраняем переговоры для метрик
                negotiations = self._handle_negotiation_phase_with_metrics(log_file)
                
                # Handle action phase
                self.game_state["phase"] = "Action"
                orders = self._handle_action_phase_with_metrics(log_file)
                
                # Сохраняем состояние после хода
                territories_after = self._get_territories_state()
                
                # Анализируем атаки для метрик
                attacks_received = self._analyze_attacks(orders)
                
                # Собираем данные раунда для метрик
                round_data = {
                    "round": self.game_round,
                    "year": self.game_state.get("year", 0),
                    "phase": self.game_state.get("phase", ""),
                    "territories_before": territories_before,
                    "territories_after": territories_after,
                    "negotiations": negotiations,
                    "orders": orders,
                    "attacks_received": attacks_received
                }
                rounds_data.append(round_data)
                
                round_time = time.time() - round_start_time
                self.log_message(log_file, f"Round {self.game_round} completed in {round_time:.2f} seconds")

                if self.check_game_end():
                    break

            # Calculate game play time
            game_time = time.time() - game_start_time
            
            # Get final results
            results = self.get_game_results()
            results["game_time"] = game_time
            results["rounds_data"] = rounds_data  # Добавляем данные раундов для метрик
            
            self.log_message(log_file, f"\nGame ended after {self.game_round} rounds in {game_time:.2f} seconds")
            self.log_message(log_file, f"Final state: {json.dumps(results, indent=2)}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error in game loop: {str(e)}"
            self.log_message(log_file, error_msg)
            logger.error(error_msg)
            return {"error": error_msg, "rounds_played": self.game_round}
        finally:
            if log_file:
                log_file.close()

    def get_game_results(self) -> Dict[str, Any]:
        """Determine and return game results."""
        # Count supply centers for each power
        supply_centers = {}
        for power in self.powers:
            centers = self.diplomacy_game.get_centers(power)
            supply_centers[power] = len(centers)
            
        # Determine winner if any
        winner = None
        for power, count in supply_centers.items():
            if count >= 18:
                winner = power
        
        # Get strategic decisions from each power
        strategic_decisions = {}
        for agent in self.agents:
            try:
                strategic_decisions[agent.power_name] = agent.make_strategic_decision(self.game_state)
            except Exception as e:
                strategic_decisions[agent.power_name] = f"Error getting strategic decision: {str(e)}"
        
        results = {
            "rounds_played": self.game_round,
            "supply_centers": supply_centers,
            "winner": winner,
            "strategic_decisions": strategic_decisions,
            "final_year": self.game_state.get("year")
        }
        return results

    def _handle_negotiation_phase_with_metrics(self, log_file) -> Dict[str, Any]:
        """Orchestrate negotiation between powers and collect data for metrics."""
        self.log_message(log_file, "\nNegotiation Phase Begins")
        powers_to_negotiate = [agent.power_name for agent in self.agents]
        negotiations = {}  # Структура для хранения переговоров
        
        # ИСПРАВЛЕНИЕ: Ограничиваем количество переговорных пар для бенчмарка
        max_negotiation_pairs = 5  # Максимум 5 пар вместо всех 21
        
        negotiation_pairs = []
        for i in range(len(powers_to_negotiate)):
            for j in range(i + 1, len(powers_to_negotiate)):
                negotiation_pairs.append((powers_to_negotiate[i], powers_to_negotiate[j]))
        
        # Ограничиваем количество пар для бенчмарка
        if len(negotiation_pairs) > max_negotiation_pairs:
            negotiation_pairs = random.sample(negotiation_pairs, max_negotiation_pairs)
            self.log_message(log_file, f"Limiting negotiations to {max_negotiation_pairs} pairs for benchmark efficiency")
        
        for power1_name, power2_name in negotiation_pairs:
            agent1 = None
            agent2 = None
            for agent in self.agents:
                if agent.power_name == power1_name:
                    agent1 = agent
                if agent.power_name == power2_name:
                    agent2 = agent
            if not agent1 or not agent2:
                continue
            
            if power1_name not in negotiations:
                negotiations[power1_name] = {}
            if power2_name not in negotiations:
                negotiations[power2_name] = {}
            
            self.log_message(log_file, f"\nNegotiation between {power1_name} and {power2_name}:")
            
            # ИСПРАВЛЕНИЕ: Ограничиваем переговоры до 1 раунда вместо 3 для бенчмарка
            max_negotiation_turns = 1  # Сокращаем с 3 до 1 раунда
            for turn in range(max_negotiation_turns):
                self.log_message(log_file, f"\nNegotiation Turn {turn + 1} between {power1_name} and {power2_name}:")
                
                # Power 1 initiates/responds
                message_p1_to_p2 = agent1.negotiate(power2_name, self.game_state)
                self.log_message(log_file, f"{power1_name} to {power2_name}: {message_p1_to_p2}")
                agent2.private_history.append(create_message('assistant', f"Negotiation message from {power1_name} (Turn {turn + 1}): {message_p1_to_p2}"))
                
                # Сохраняем для метрик
                if power2_name not in negotiations[power1_name]:
                    negotiations[power1_name][power2_name] = {}
                negotiations[power1_name][power2_name][str(turn)] = message_p1_to_p2
                
                # Power 2 responds/initiates
                message_p2_to_p1 = agent2.negotiate(power1_name, self.game_state)
                self.log_message(log_file, f"{power2_name} to {power1_name}: {message_p2_to_p1}")
                agent1.private_history.append(create_message('assistant', f"Negotiation message from {power2_name} (Turn {turn + 1}): {message_p2_to_p1}"))
                
                # Сохраняем для метрик
                if power1_name not in negotiations[power2_name]:
                    negotiations[power2_name][power1_name] = {}
                negotiations[power2_name][power1_name][str(turn)] = message_p2_to_p1
        
        return negotiations

    def _handle_action_phase_with_metrics(self, log_file) -> Dict[str, List[str]]:
        """Collect and process actions from agents and submit to the diplomacy game."""
        self.log_message(log_file, "\nAction Phase Begins")
        orders = {}
        
        for agent in self.agents:
            power_name = agent.power_name
            try:
                power_orders = agent.get_orders(self.game_state)
                
                # Проверяем, что приказы не пустые
                if not power_orders:
                    # Создаем hold приказы для всех юнитов
                    my_units = self.diplomacy_game.get_units(power_name)
                    power_orders = []
                    for unit in my_units:
                        unit_str = str(unit)
                        unit_parts = unit_str.split()
                        if len(unit_parts) >= 2:
                            hold_order = f"{unit_parts[0]} {unit_parts[1]}"
                            power_orders.append(hold_order)
                    self.log_message(log_file, f"Generated fallback hold orders for {power_name}: {power_orders}")
                
                orders[power_name] = power_orders
                self.log_message(log_file, f"{power_name} orders: {power_orders}")
                
                # Submit orders to the diplomacy library game
                if power_orders:
                    try:
                        self.diplomacy_game.set_orders(power_name, power_orders)
                        self.log_message(log_file, f"Successfully set orders for {power_name}")
                    except Exception as e:
                        self.log_message(log_file, f"Error setting orders for {power_name}: {e}")
                        # Пытаемся установить пустой список приказов
                        try:
                            self.diplomacy_game.set_orders(power_name, [])
                            self.log_message(log_file, f"Set empty orders for {power_name} as fallback")
                        except Exception as e2:
                            self.log_message(log_file, f"Failed to set even empty orders for {power_name}: {e2}")
                else:
                    self.log_message(log_file, f"No orders to set for {power_name}")
                    
            except Exception as e:
                self.log_message(log_file, f"Error getting orders from {power_name}: {e}")
                orders[power_name] = []
        
        # Process the turn in the diplomacy library
        try:
            self.log_message(log_file, "Processing turn...")
            self.diplomacy_game.process()
            self.log_message(log_file, "Turn processed successfully")
        except Exception as e:
            self.log_message(log_file, f"Error processing turn: {e}")
            # Не прерываем игру, продолжаем
        
        # Update the game state
        try:
            self.update_game_state(orders, log_file)
            self.game_state["phase"] = "Negotiation"  # Reset phase for next round
        except Exception as e:
            self.log_message(log_file, f"Error updating game state: {e}")
            # Принудительно обновляем состояние
            self._update_game_state_from_lib()
        
        return orders

    def _get_territories_state(self) -> Dict[str, List[str]]:
        """Get the current state of territories for all powers."""
        territories = {}
        # Получаем данные из game_state вместо diplomacy_game напрямую
        centers_by_power = self.game_state.get("supply_centers", {})
        units_by_power = self.game_state.get("units", {})
        
        for power in self.powers:
            territories[power] = []
            
            # Добавляем контролируемые центры снабжения
            if power in centers_by_power:
                territories[power].extend(centers_by_power[power])
                
            # Добавляем территории с юнитами
            if power in units_by_power:
                for unit_str in units_by_power[power]:
                    # Извлекаем территорию из строки юнита (формат "F LON", "A PAR" и т.д.)
                    parts = unit_str.split()
                    if len(parts) > 1:
                        territory = parts[-1]
                        if territory not in territories[power]:
                            territories[power].append(territory)
                            
        return territories

    def _analyze_attacks(self, orders: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Analyze which territories are being attacked by whom."""
        attacks_received = {power: [] for power in self.powers}
        
        # Получаем текущий state территорий
        territories_state = self._get_territories_state()
        
        # Для каждой державы анализируем приказы
        for attacker, attacker_orders in orders.items():
            for order in attacker_orders:
                # Ищем приказы на движение (они обычно содержат "-")
                if " - " in order.upper() or " TO " in order.upper():
                    # Извлекаем целевую территорию
                    parts = order.split()
                    if len(parts) >= 3:
                        target_territory = parts[-1]
                        
                        # Определяем, какой державе принадлежит территория
                        for defender, territories in territories_state.items():
                            if target_territory in territories:
                                attacks_received[defender].append({
                                    "from": attacker,
                                    "target": target_territory,
                                    "order": order
                                })
                                break
        
        return attacks_received