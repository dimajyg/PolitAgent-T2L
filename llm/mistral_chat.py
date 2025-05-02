from mistralai import Mistral
from func_timeout import func_set_timeout
from time import sleep
from typing import Any, List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from llm.base_chat import BaseChat
import logging

from llm.config import key_mistral, temperature_mistral, model_mistral

@func_set_timeout(15)
def get_response(messages):
    client = Mistral(api_key=key_mistral)
    response = client.chat.complete(
        model=model_mistral,
        temperature=temperature_mistral,
        messages=messages
    )
    return response

class Mistral_Base:
    def __init__(self) -> None:
        self.name = "mistral"

    def single_chat(self, content, role=None):
        if role is None:
            role = "You are an AI assistant that helps people find information."
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": content}
        ]
        res = None
        cnt = 0
        while True:
            try:
                response = get_response(messages)
                res = response.choices[0].message.content
                break
            except:
                cnt += 1
            if cnt >= 5:
                break
        return res

    def multi_chat(self, messages):
        res = None
        cnt = 0
        while True:
            sleep(5)
            try:
                response = get_response(messages)
                res = response.choices[0].message.content
                break
            except Exception as e:
                print(e)
                cnt += 1
            if cnt >= 3:
                break
        return res

class MistralChat(BaseChat):
    """
    Класс для общения с Mistral LLM через LangChain.

    Args:
        system_prompt (str): Системный промпт.
        model_name (str): Название модели.
        temperature (float): Температура генерации.
        api_key (Optional[str]): Ключ API.
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "mistral-medium",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(system_prompt)
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or key_mistral
        # self.llm = ChatMistral(model_name=model_name, temperature=temperature, api_key=api_key)
        self.llm = None  # Заглушка, заменяется на прямой вызов API в multi_chat и send
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
            # Преобразуем в формат, ожидаемый multi_chat
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            response = self.multi_chat(messages)
            self.history.append({"user": user_message, "assistant": response})
            return response
        except Exception as e:
            logging.error(f"MistralChat error: {e}")
            return "Ошибка генерации ответа."
    
    def multi_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Отправить список сообщений и получить ответ от Mistral.
        
        Args:
            messages (List[Dict[str, str]]): Список сообщений в формате [{"role": "system", "content": "..."}, ...]
            
        Returns:
            str: Ответ от модели
        """
        res = None
        cnt = 0
        while True:
            sleep(1)  # Уменьшенная задержка для лучшего UX
            try:
                response = get_response(messages)
                res = response.choices[0].message.content
                break
            except Exception as e:
                logging.error(f"Mistral API error: {e}")
                cnt += 1
                sleep(2)  # Экспоненциальный backoff можно добавить здесь
            if cnt >= 3:
                res = "Ошибка при обращении к Mistral API после 3 попыток"
                break
        return res