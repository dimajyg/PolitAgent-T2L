from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class Agent(ABC):
    def __init__(self, chatbot, player_name: str, players: List[str]):
        self.chatbot = chatbot
        self.player_name = player_name
        self.players = players
        self.private_history = []
        self.role = self.__class__.__name__.replace('Agent', '')

    def update_history(self, message: Dict[str, str]):
        """Add a message to agent's private history"""
        self.private_history.append(message)

    @abstractmethod
    def chat(self, context: str) -> Tuple[str, Dict[str, Any]]:
        """Process a chat message and return response with chain of thought"""
        pass

    def get_history(self) -> List[Dict[str, str]]:
        """Get agent's message history"""
        return self.private_history

    def clear_history(self):
        """Clear agent's message history"""
        self.private_history = []

    @abstractmethod
    def get_role_description(self) -> str:
        """Get description of agent's role in the game"""
        pass