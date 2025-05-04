from typing import Any, Dict, Optional, List
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

class BaseAgent:
    """
    Базовый агент для игр с поддержкой LLM через LangChain.

    Args:
        name (str): Имя агента.
        llm (BaseLanguageModel): Языковая модель для генерации ходов.
        prompt_template (str): Шаблон промпта для генерации действий.

    Attributes:
        name (str): Имя агента.
        llm (BaseLanguageModel): Языковая модель.
        prompt (ChatPromptTemplate): Шаблон для промпта.
        chain (RunnableSequence): Цепочка LangChain для генерации действий.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLanguageModel,
        prompt_template: str,
    ) -> None:
        self.name = name
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        # Используем современный pipeline вместо устаревшего LLMChain
        self.chain = self.prompt | self.llm

    def act(self, observation: Dict[str, Any]) -> str:
        """
        Сгенерировать действие агента на основе наблюдения.

        Args:
            observation (Dict[str, Any]): Текущее состояние/наблюдение.

        Returns:
            str: Сгенерированное действие.
        """
        response = self.chain.invoke(observation)
        # Извлекаем содержимое из ответа LangChain
        return response.content if hasattr(response, 'content') else str(response)
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Отправить сообщения LLM и получить ответ.

        Args:
            messages (List[Dict[str, str]]): Сообщения в формате [{"role": "system", "content": "..."}, ...]

        Returns:
            str: Ответ от модели
        """
        return self.llm.invoke(messages).content

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name})"