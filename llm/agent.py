from typing import Any, Dict, Optional
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

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
        chain (LLMChain): Цепочка LangChain для генерации действий.
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
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def act(self, observation: Dict[str, Any]) -> str:
        """
        Сгенерировать действие агента на основе наблюдения.

        Args:
            observation (Dict[str, Any]): Текущее состояние/наблюдение.

        Returns:
            str: Сгенерированное действие.
        """
        return self.chain.run(observation)

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name})"