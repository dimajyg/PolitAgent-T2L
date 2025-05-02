from typing import Any, Dict, List, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from llm.base_chat import BaseChat

# Класс для обратной совместимости
class OpenAI_Base:
    def __init__(self) -> None:
        self.name = "openai"
        
    def single_chat(self, content, role=None):
        """Обратная совместимость со старым API"""
        if role is None:
            role = "You are an AI assistant that helps people find information."
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": content}
        ]
        return self.multi_chat(messages)
    
    def multi_chat(self, messages):
        """Обратная совместимость со старым API"""
        try:
            chat = OpenAIChat(system_prompt="")
            return chat.multi_chat(messages)
        except Exception as e:
            logging.error(f"OpenAI error: {e}")
            return "Error in OpenAI API call"

class OpenAIChat(BaseChat):
    """
    Класс для общения с OpenAI LLM через LangChain.

    Args:
        system_prompt (str): Системный промпт.
        model_name (str): Название модели OpenAI.
        temperature (float): Температура генерации.
        api_key (Optional[str]): Ключ OpenAI API.
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(system_prompt)
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{user_message}"),
        ])

    def send(self, user_message: str, **kwargs: Any) -> str:
        """
        Отправить сообщение пользователем и получить ответ LLM.

        Args:
            user_message (str): Сообщение пользователя.

        Returns:
            str: Ответ LLM.
        """
        try:
            # Для совместимости используем multi_chat
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            response = self.multi_chat(messages)
            self.history.append({"user": user_message, "assistant": response})
            return response
        except Exception as e:
            logging.error(f"OpenAIChat error: {e}")
            return "Ошибка генерации ответа."
            
    def multi_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Отправить список сообщений и получить ответ от OpenAI.
        
        Args:
            messages (List[Dict[str, str]]): Список сообщений в формате [{"role": "system", "content": "..."}, ...]
            
        Returns:
            str: Ответ от модели
        """
        try:
            # Используем langchain_openai для отправки запроса
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return f"Ошибка при обращении к OpenAI API: {str(e)}"