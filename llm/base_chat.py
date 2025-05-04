from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseChat(ABC):
    """
    Абстрактный базовый класс для чата с LLM.

    Args:
        system_prompt (str): Системный промпт для инициализации диалога.

    Attributes:
        system_prompt (str): Системный промпт.
        history (List[Dict[str, str]]): История сообщений.
    """

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []

    @abstractmethod
    def send(self, user_message: str, **kwargs: Any) -> str:
        """
        Отправить сообщение пользователем и получить ответ LLM.

        Args:
            user_message (str): Сообщение пользователя.

        Returns:
            str: Ответ LLM.
        """
        pass

    def reset_history(self) -> None:
        """Сбросить историю диалога."""
        self.history = [] 