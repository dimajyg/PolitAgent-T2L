from typing import Any, Dict, Optional, List
try:
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableSequence
except ImportError:
    from langchain.llms.base import BaseLLM as BaseLanguageModel
    from langchain.prompts import ChatPromptTemplate
    RunnableSequence = None

class BaseAgent:
    """
    Base agent for games with LLM support through LangChain.

    Args:
        name (str): Name of the agent.
        llm (BaseLanguageModel): Language model for generating moves.
        prompt_template (str): Prompt template for generating actions.

    Attributes:
        name (str): Name of the agent.
        llm (BaseLanguageModel): Language model.
        prompt (ChatPromptTemplate): Prompt template.
        chain (RunnableSequence): LangChain pipeline for generating actions.
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
                return response.content if hasattr(response, 'content') else str(response)
            else:
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
                combined = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                return self.llm(combined)
        except Exception as e:
            return f"Error in chat: {e}"

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name})"