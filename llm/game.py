from abc import ABC, abstractmethod
import json
import os
from typing import List, Dict, Any, Optional
from llm.agent import BaseAgent

class Game(ABC):
    def __init__(self, args):
        self.args = args
        self.debug = args.debug if hasattr(args, 'debug') else False
        self.players = []
        self.agents = []
        self.name2agent = {}
        self.public_messages = []
        self.game_round = 0

    @abstractmethod
    def init_game(self):
        """Initialize game state, agents, and settings"""
        pass

    @abstractmethod
    def game_loop(self):
        """Main game loop implementation"""
        pass

    def update_history(self, message, sender_name):
        """Update message history for all agents except the sender"""
        for agent in self.agents:
            if agent.player_name != sender_name:
                agent.private_history.append(message)
        self.public_messages.append(message)

    def log_message(self, file, message, cot=None):
        """Log message and chain of thought to file"""
        file.write(message + "\n")
        if cot:
            file.write(json.dumps(cot) + "\n")
        if self.debug:
            print(message)
            if cot:
                print(json.dumps(cot))
            print()

    def create_log_dir(self, base_dir):
        """Create log directory if it doesn't exist"""
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        return base_dir

    def run(self, log_file):
        """Run a complete game session"""
        try:
            self.init_game()
            return self.game_loop(log_file)
        except Exception as e:
            if self.debug:
                print(f"Error: {str(e)}")
            return {"error": str(e)}

class BaseGame:
    """
    Base class for games with LangChain agents.

    Args:
        agents (List[BaseAgent]): List of agents.
        state (Optional[Dict[str, Any]]): Initial game state.

    Attributes:
        agents (List[BaseAgent]): Agents.
        state (Dict[str, Any]): Current game state.
        history (List[Dict[str, Any]]): Game history.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agents = agents
        self.state = state or {}
        self.history: List[Dict[str, Any]] = []

    def step(self) -> None:
        """
        Make one game step: each agent makes a move.
        """
        for agent in self.agents:
            observation = self._get_observation(agent)
            action = agent.act(observation)
            self._apply_action(agent, action)
            self.history.append({
                "agent": agent.name,
                "observation": observation,
                "action": action,
            })

    def _get_observation(self, agent: BaseAgent) -> Dict[str, Any]:
        """
        Get observation for an agent (can be overridden in subclasses).

        Args:
            agent (BaseAgent): Agent.

        Returns:
            Dict[str, Any]: Observation.
        """
        return {"state": self.state, "agent": agent.name}

    def _apply_action(self, agent: BaseAgent, action: str) -> None:
        """
        Apply agent's action to the game state (stub, can be overridden in subclasses).

        Args:
            agent (BaseAgent): Agent.
            action (str): Action.
        """
        print(f"{agent.name} made an action: {action}")

    def run(self, steps: int = 1) -> None:
        """
        Run the game for a given number of steps.

        Args:
            steps (int): Number of steps.
        """
        for _ in range(steps):
            self.step()