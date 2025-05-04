from typing import Any, Dict, List, Optional, Tuple, Union
import json
import random

from llm.base_chat import BaseChat
from llm.openai_chat import OpenAIChat
from llm.mistral_chat import MistralChat

from environments.spyfall.utils.prompt import (
    game_prompt_en, game_prompt_template, 
    describe_prompt_template, vote_prompt_template,
    describe_parser, vote_parser
)
from environments.spyfall.utils.utils import create_message

class BaseAgent:
    """
    Базовый агент Spyfall, использующий LangChain-совместимый чат.

    Args:
        chatbot (BaseChat): Объект чата (OpenAIChat, MistralChat и др.).
        player_name (str): Имя игрока.
        players (List[str]): Список всех игроков.
        phrase (str): Слово/фраза для роли.
        is_spy (bool): Является ли агент шпионом.

    Attributes:
        chatbot (BaseChat): Чат-LLM.
        player_name (str): Имя игрока.
        players (List[str]): Список игроков.
        phrase (str): Слово для роли.
        is_spy (bool): Флаг шпиона.
        private_history (List[Dict[str, str]]): Приватная история сообщений.
    """

    def __init__(
        self,
        chatbot: BaseChat,
        player_name: str,
        players: List[str],
        phrase: str,
        is_spy: bool = False,
    ) -> None:
        self.chatbot = chatbot
        self.player_name = player_name
        self.players = players
        self.phrase = phrase
        self.is_spy = is_spy
        self.private_history: List[Dict[str, str]] = []

    def get_role_description(self) -> str:
        """Возвращает строку-описание роли."""
        role = "spy" if self.is_spy else "villager"
        return f"A {role} in the Spyfall game with the phrase: {self.phrase}"

    def chat(self, context: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Общение с LLM по текущему контексту.

        Args:
            context (str): Контекст для LLM.

        Returns:
            Tuple[Optional[str], Optional[Dict[str, Any]]]: Ответ и цепочка рассуждений.
        """
        # Формируем историю сообщений для LLM
        messages = [
            {"role": "system", "content": game_prompt_en},
            {"role": "user", "content": context},
            *self.private_history,
        ]
        
        try:
            # Use the invoke method consistently across all methods
            response = self.chatbot.invoke(messages).content
                
            # Парсим ответ как JSON
            cot = json.loads(response)
            return cot.get("answer", ""), cot
        except Exception as e:
            print(f"Error in chat: {e}")
            return None, None

    def describe(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Генерирует описание для текущей роли, используя структурированный вывод.

        Returns:
            Tuple[Optional[str], Optional[Dict[str, Any]]]: Описание и цепочка рассуждений.
        """
        # Используем LangChain PromptTemplate со структурированным выводом
        role = "spy" if self.is_spy else "villager"
        prompt = describe_prompt_template.format(
            game_prompt=game_prompt_en,
            players=json.dumps(self.players),
            player_name=self.player_name,
            role=role,
            phrase=self.phrase
        )
        
        try:
            # Create a message format and use the invoke method
            messages = [{"role": "user", "content": prompt}]
            response = self.chatbot.invoke(messages).content
            
            # Используем структурированный парсер вместо ручного парсинга JSON
            parsed_output = describe_parser.parse(response)
            # Преобразуем Pydantic модель в dict для совместимости
            res = parsed_output.model_dump()
            speak = res.get("speak")
            return speak, res
        except Exception as e:
            print(f"Error parsing describe response: {e}")
            # Пробуем резервный метод в случае ошибки
            try:
                response = self._strip_json_markers(response)
                res = json.loads(response)
                speak = res.get("speak")
                return speak, res
            except Exception:
                return None, None

    def vote(self) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Генерирует голос агента, используя структурированный вывод.

        Returns:
            Tuple[str, str, Optional[Dict[str, Any]]]: Имя выбранного игрока, речь, цепочка рассуждений.
        """
        # Используем LangChain PromptTemplate со структурированным выводом
        role = "spy" if self.is_spy else "villager"
        living_players = self._extract_living_players_from_history()
        prompt = vote_prompt_template.format(
            game_prompt=game_prompt_en,
            player_name=self.player_name,
            role=role,
            living_players=json.dumps(living_players)
        )
        
        try:
            # Create a message format and use the invoke method
            messages = [{"role": "user", "content": prompt}]
            response = self.chatbot.invoke(messages).content
            
            # Используем структурированный парсер вместо ручного парсинга JSON
            parsed_output = vote_parser.parse(response)
            # Преобразуем Pydantic модель в dict для совместимости
            res = parsed_output.model_dump()
            thought = res.get("thought")
            speak = res.get("speak")
            name = res.get("name")
            
            # Проверка валидности голоса
            if living_players and name not in living_players:
                name = random.choice(living_players)
                thought = (thought or "") + f" [NOTE: Original vote was invalid, randomly selected {name} from living players]"
                res["name"] = name
                res["thought"] = thought
            
            return name, speak, res
        except Exception as e:
            print(f"Error parsing vote response: {e}")
            # Пробуем резервный метод в случае ошибки
            try:
                response = self._strip_json_markers(response)
                res = json.loads(response)
                thought = res.get("thought")
                speak = res.get("speak")
                name = res.get("name")
                
                # Проверка валидности голоса
                if living_players and name not in living_players:
                    name = random.choice(living_players)
                    thought = (thought or "") + f" [NOTE: Original vote was invalid, randomly selected {name} from living players]"
                
                return name, speak, res
            except Exception:
                # В случае ошибки парсинга, если есть живые игроки, выбираем случайного
                if living_players:
                    name = random.choice(living_players)
                    thought = f"Failed to parse my response, randomly selecting {name}"
                    speak = f"I think {name} is suspicious"
                    return name, speak, {"thought": thought, "speak": speak, "name": name}
                else:
                    return "", "", None

    def _extract_living_players_from_history(self) -> List[str]:
        """
        Извлекает список живых игроков из истории сообщений.

        Returns:
            List[str]: Список живых игроков.
        """
        living_players = []
        for message in reversed(self.private_history):
            if message.get("role") == "user" and "Host: The living players are:" in message.get("content", ""):
                try:
                    content = message["content"]
                    json_start = content.find('[')
                    json_end = content.rfind(']') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        living_players = json.loads(json_str)
                        break
                except Exception:
                    living_players = self.players.copy()
                    break
        if not living_players:
            living_players = self.players.copy()
        return living_players

    @staticmethod
    def _strip_json_markers(text: str) -> str:
        """Удаляет markdown-ограничители из json-ответа."""
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Преобразует список сообщений в строку-промпт."""
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += "system: " + message["content"] + "\n"
            elif message["role"] == "assistant":
                prompt += "assistant: " + message["content"] + "\n"
            else:
                prompt += message["content"] + "\n"
        return prompt