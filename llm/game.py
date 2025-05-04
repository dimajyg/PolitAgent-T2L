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
    Базовый класс игры с поддержкой LangChain-агентов.

    Args:
        agents (List[BaseAgent]): Список агентов.
        state (Optional[Dict[str, Any]]): Начальное состояние игры.

    Attributes:
        agents (List[BaseAgent]): Агенты.
        state (Dict[str, Any]): Текущее состояние игры.
        history (List[Dict[str, Any]]): История ходов.
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
        Совершить один шаг игры: каждый агент делает ход.
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
        Получить наблюдение для агента (можно переопределить в наследниках).

        Args:
            agent (BaseAgent): Агент.

        Returns:
            Dict[str, Any]: Наблюдение.
        """
        return {"state": self.state, "agent": agent.name}

    def _apply_action(self, agent: BaseAgent, action: str) -> None:
        """
        Применить действие агента к состоянию игры (заглушка, переопределять в наследниках).

        Args:
            agent (BaseAgent): Агент.
            action (str): Действие.
        """
        # Пример: просто логируем действие
        print(f"{agent.name} совершил действие: {action}")

    def run(self, steps: int = 1) -> None:
        """
        Запустить игру на заданное число шагов.

        Args:
            steps (int): Количество шагов.
        """
        for _ in range(steps):
            self.step()