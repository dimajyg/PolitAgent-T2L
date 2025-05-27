from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseChat(ABC):
    """
    Abstract base class for LLM chat interactions.

    Args:
        system_prompt (str): System prompt to initialize the conversation.

    Attributes:
        system_prompt (str): System prompt for the conversation.
        history (List[Dict[str, str]]): Message history containing the conversation.
    """

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []

    @abstractmethod
    def send(self, user_message: str, **kwargs: Any) -> str:
        """
        Send a user message and get a response from the LLM.

        Args:
            user_message (str): The user's message to send.

        Returns:
            str: The LLM's response.
        """
        pass

    def reset_history(self) -> None:
        """Reset the conversation history."""
        self.history = [] 