from typing import Any, Dict, Optional, List
try:
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableSequence
except ImportError:
    # Fallback for older langchain versions
    from langchain.llms.base import BaseLLM as BaseLanguageModel
    from langchain.prompts import ChatPromptTemplate
    RunnableSequence = None

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
        try:
            self.prompt = ChatPromptTemplate.from_template(prompt_template)
            # Используем современный pipeline если доступен
            if RunnableSequence:
                self.chain = self.prompt | self.llm
            else:
                self.chain = None
        except:
            self.prompt = None
            self.chain = None

    def act(self, observation: Dict[str, Any]) -> str:
        """
        Сгенерировать действие агента на основе наблюдения.

        Args:
            observation (Dict[str, Any]): Текущее состояние/наблюдение.

        Returns:
            str: Сгенерированное действие.
        """
        try:
            if self.chain:
                response = self.chain.invoke(observation)
                # Извлекаем содержимое из ответа LangChain
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback для старых версий
                return self.llm(str(observation))
        except Exception as e:
            return f"Error generating action: {e}"
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Отправить сообщения LLM и получить ответ.

        Args:
            messages (List[Dict[str, str]]): Сообщения в формате [{"role": "system", "content": "..."}, ...]

        Returns:
            str: Ответ от модели
        """
        try:
            if hasattr(self.llm, 'invoke'):
                result = self.llm.invoke(messages)
                return result.content if hasattr(result, 'content') else str(result)
            else:
                # Fallback для старых версий
                combined = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                return self.llm(combined)
        except Exception as e:
            return f"Error in chat: {e}"

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name})"