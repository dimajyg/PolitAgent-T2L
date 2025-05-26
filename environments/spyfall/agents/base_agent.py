from typing import Any, Dict, List, Optional, Tuple
import json
import random
import re

from llm.base_chat import BaseChat

from environments.spyfall.utils.prompt import (
    game_prompt_en, game_prompt_template, 
    describe_prompt_template, vote_prompt_template,
    describe_parser, vote_parser
)

class BaseAgent:
    """
    Базовый агент Spyfall, использующий LangChain-совместимый чат.

    Args:
        chatbot (BaseChat): Объект чата (OpenAIChatModel, MistralChatModel и др.).
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
            {"role": "user", "content": context + "\n\nGame history so far:"},
            *self.private_history,  # Complete dialogue history of the game
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
        Генерирует описание для текущей роли, используя обычное JSON парсирование.

        Returns:
            Tuple[Optional[str], Optional[Dict[str, Any]]]: Описание и цепочка рассуждений.
        """
        role = "spy" if self.is_spy else "villager"
        prompt = f"""
        {game_prompt_en}
        
        Players: {json.dumps(self.players)}
        Your name: {self.player_name}
        Your role: {role}
        Your phrase: {self.phrase}
        
        Generate a description about your word/phrase without directly saying it.
        
        Respond with a JSON object in this exact format:
        {{
            "thought": "Private reasoning about the strategy and role",
            "speak": "Public statement about the word without directly saying it"
        }}
        """
        
        try:
            # Create a message format and use the invoke method
            messages = [{"role": "user", "content": prompt}]
            # Add private history to the messages
            messages.extend(self.private_history)
            response = self.chatbot.invoke(messages).content
            
            # Add small delay to simulate thinking time
            import time
            time.sleep(0.1)
            
            # Try to parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_output = json.loads(json_str)
                speak = parsed_output.get("speak", "")
                return speak, parsed_output
            
            # If JSON parsing fails, extract speak content from response
            speak = response.strip()
            if len(speak) > 200:  # If too long, truncate
                speak = speak[:200] + "..."
            
            return speak, {"thought": "Could not parse JSON", "speak": speak}
            
        except Exception as e:
            print(f"Error parsing describe response: {e}")
            # Fallback description
            if self.is_spy:
                speak = "It's something you might encounter in daily life."
            else:
                speak = f"It's related to {self.phrase[0] if self.phrase else 'something familiar'}."
            return speak, {"thought": "Error in processing", "speak": speak}

    def vote(self) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Генерирует голос агента, используя обычное JSON парсирование.

        Returns:
            Tuple[str, str, Optional[Dict[str, Any]]]: Имя выбранного игрока, речь, цепочка рассуждений.
        """
        role = "spy" if self.is_spy else "villager"
        living_players = self._extract_living_players_from_history()
        
        prompt = f"""
        {game_prompt_en}
        
        Your name: {self.player_name}
        Your role: {role}
        Living players: {json.dumps(living_players)}
        
        Vote for who you think is the spy. You must choose from the living players.
        
        Respond with a JSON object in this exact format:
        {{
            "thought": "Private reasoning about who might be the spy",
            "speak": "Public statement explaining vote",
            "name": "Name of player being voted for (must be in living_players)"
        }}
        """
        
        try:
            # Create a message format and use the invoke method
            messages = [{"role": "user", "content": prompt}]
            # Add private history to the messages
            messages.extend(self.private_history)
            response = self.chatbot.invoke(messages).content
            
            # Add small delay to simulate thinking time
            import time
            time.sleep(0.1)
            
            # Try to parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_output = json.loads(json_str)
                
                thought = parsed_output.get("thought", "")
                speak = parsed_output.get("speak", "")
                name = parsed_output.get("name", "")
                
                # Validate vote target
                if name not in living_players and living_players:
                    # Try to find a mentioned player in the response
                    for player in living_players:
                        if player.lower() in response.lower():
                            name = player
                            break
                    else:
                        name = random.choice(living_players)
                        thought += f" [NOTE: Original vote was invalid, randomly selected {name}]"
                
                return name, speak, {"thought": thought, "speak": speak, "name": name}
            
            # If JSON parsing fails, try to extract information from text
            target = None
            for player in living_players:
                if player.lower() in response.lower():
                    target = player
                    break
            
            if not target and living_players:
                target = random.choice(living_players)
            
            speak = response.strip()
            if len(speak) > 100:
                speak = speak[:100] + "..."
            
            thought = "Could not parse JSON response"
            return target or "", speak, {"thought": thought, "speak": speak, "name": target}
            
        except Exception as e:
            print(f"Error parsing vote response: {e}")
            # Fallback vote
            if living_players:
                name = random.choice(living_players)
                thought = f"Failed to parse response, randomly selecting {name}"
                speak = f"I think {name} is suspicious"
                return name, speak, {"thought": thought, "speak": speak, "name": name}
            else:
                return "", "", {"thought": "No living players", "speak": "", "name": ""}

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
                    if json_start != -1:
                        json_end = content.find(']', json_start) + 1
                        json_str = content[json_start:json_end]
                        living_players = json.loads(json_str)
                        break
                except Exception:
                    continue
        
        # Fallback to all players except self if no living players found
        if not living_players:
            living_players = [p for p in self.players if p != self.player_name]
        
        return living_players

    @staticmethod
    def _strip_json_markers(text: str) -> str:
        """
        Удаляет маркеры JSON блоков из текста.

        Args:
            text (str): Исходный текст.

        Returns:
            str: Очищенный текст.
        """
        return text.replace('```json', '').replace('```', '').strip()

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Преобразует список сообщений в один промпт.

        Args:
            messages (List[Dict[str, str]]): Список сообщений.

        Returns:
            str: Объединенный промпт.
        """
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])