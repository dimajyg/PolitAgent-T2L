from typing import Any, Dict, List, Optional, Tuple
import json
import random
import re

from llm.base_chat import BaseChat

from environments.spyfall.utils.prompt import game_prompt_en

class BaseAgent:
    """
    Base Spyfall agent using LangChain-compatible chat.

    Args:
        chatbot (BaseChat): Chat object (OpenAIChatModel, MistralChatModel, etc.).
        player_name (str): Player name.
        players (List[str]): List of all players.
        phrase (str): Word/phrase for the role.
        is_spy (bool): Whether the agent is a spy.

    Attributes:
        chatbot (BaseChat): Chat-LLM.
        player_name (str): Player name.
        players (List[str]): List of players.
        phrase (str): Word for the role.
        is_spy (bool): Spy flag.
        private_history (List[Dict[str, str]]): Private message history.
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
        """Returns a string-description of the role."""
        role = "spy" if self.is_spy else "villager"
        return f"A {role} in the Spyfall game with the phrase: {self.phrase}"

    def chat(self, context: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Communicating with LLM on the current context.

        Args:
            context (str): Context for LLM.

        Returns:
            Tuple[Optional[str], Optional[Dict[str, Any]]]: Response and reasoning chain.
        """
        messages = [
            {"role": "system", "content": game_prompt_en},
            {"role": "user", "content": context + "\n\nGame history so far:"},
            *self.private_history, 
        ]
        
        try:
            response = self.chatbot.invoke(messages).content
                
            cot = json.loads(response)
            return cot.get("answer", ""), cot
        except Exception as e:
            print(f"Error in chat: {e}")
            return None, None

    def describe(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generates a description for the current role, using regular JSON parsing.

        Returns:
            Tuple[Optional[str], Optional[Dict[str, Any]]]: Description and reasoning chain.
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
            messages = [{"role": "user", "content": prompt}]
            messages.extend(self.private_history)
            response = self.chatbot.invoke(messages).content
            
            import time
            time.sleep(0.1)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_output = json.loads(json_str)
                speak = parsed_output.get("speak", "")
                return speak, parsed_output
            
            speak = response.strip()
            if len(speak) > 200:
                speak = speak[:200] + "..."
            
            return speak, {"thought": "Could not parse JSON", "speak": speak}
            
        except Exception as e:
            print(f"Error parsing describe response: {e}")
            if self.is_spy:
                speak = "It's something you might encounter in daily life."
            else:
                first_letter = self.phrase[0] if self.phrase and len(self.phrase) > 0 else "i"
                speak = f"It's related to something that starts with '{first_letter}'."
            return speak, {"thought": "Error in processing", "speak": speak}

    def vote(self) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Generates an agent's vote, using regular JSON parsing.

        Returns:
            Tuple[str, str, Optional[Dict[str, Any]]]: Name of the chosen player, speech, reasoning chain.
        """
        role = "spy" if self.is_spy else "villager"
        living_players = self._extract_living_players_from_history()
        
        if not living_players and self.players:
            living_players = self.players.copy()
            print(f"Warning: No living players found in history for {self.player_name}. Using all players.")
        
        prompt = f"""
        {game_prompt_en}
        
        Your name: {self.player_name}
        Your role: {role}
        Living players: {json.dumps(living_players)}
        
        Vote for who you think is the spy. You MUST choose from the living players list.
        
        Respond with a JSON object in this exact format:
        {{
            "thought": "Private reasoning about who might be the spy",
            "speak": "Public statement explaining vote",
            "name": "Name of player being voted for (must be in living_players)"
        }}
        """
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                messages = [{"role": "system", "content": "You are playing Spyfall. Always respond with valid JSON."}, 
                           {"role": "user", "content": prompt}]
                
                if self.private_history:
                    for msg in self.private_history[-5:]: 
                        messages.append(msg)
                
                import time
                start_time = time.time()
                timeout = 20
                
                try:
                    response = self.chatbot.invoke(messages).content
                    elapsed = time.time() - start_time
                    print(f"Vote response received in {elapsed:.2f}s")
                except Exception as timeout_err:
                    if attempt < max_retries:
                        print(f"Timeout or error in vote attempt {attempt+1}, retrying: {timeout_err}")
                        time.sleep(1)
                        continue
                    else:
                        raise timeout_err
                
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_output = json.loads(json_str)
                    
                    thought = parsed_output.get("thought", "")
                    speak = parsed_output.get("speak", "")
                    name = parsed_output.get("name", "")
                    
                    if name not in living_players and living_players:
                        for player in living_players:
                            if player.lower() in response.lower():
                                name = player
                                break
                        else:
                            name = random.choice(living_players)
                            thought += f" [NOTE: Original vote was invalid, randomly selected {name}]"
                    
                    return name, speak, {"thought": thought, "speak": speak, "name": name}
                
                target = None
                for player in living_players:
                    if player.lower() in response.lower():
                        target = player
                        break
                
                if not target and living_players:
                    target = random.choice(living_players)
                
                speak = "I suspect " + target if target else "I'm not sure who to vote for"
                if "suspicious" in response.lower():
                    speak = f"I think {target} is suspicious"
                
                thought = "Could not parse JSON response"
                return target or "", speak, {"thought": thought, "speak": speak, "name": target}
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"Error in vote attempt {attempt+1}, retrying: {e}")
                    time.sleep(1)
                    continue
                else:
                    print(f"Error parsing vote response after {max_retries+1} attempts: {e}")
                    if living_players:
                        if self.is_spy:
                            name = random.choice(living_players)
                            thought = "As the spy, I'm randomly selecting someone to avoid suspicion"
                            speak = f"I think {name} is acting very suspicious. Their description seemed off."
                        else:
                            suspicious_players = []
                            for msg in self.private_history:
                                content = msg.get("content", "")
                                if "it's your turn" not in content.lower() and ":" in content:
                                    parts = content.split(":", 1)
                                    if len(parts) == 2 and parts[0] in living_players:
                                        suspicious_players.append(parts[0])
                            
                            if suspicious_players:
                                name = random.choice(suspicious_players)
                            else:
                                name = random.choice(living_players)
                            
                            thought = f"Failed to parse response. Selected {name} based on available information."
                            speak = f"I think {name} is suspicious based on their vague description"
                        
                        return name, speak, {"thought": thought, "speak": speak, "name": name}
                    else:
                        return "", "I don't know who to vote for", {"thought": "No living players found", "speak": "I don't know who to vote for", "name": ""}

    def _extract_living_players_from_history(self) -> List[str]:
        """
        Extracts a list of living players from the message history.

        Returns:
            List[str]: List of living players.
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
        
        if not living_players:
            living_players = [p for p in self.players if p != self.player_name]
        
        return living_players

    @staticmethod
    def _strip_json_markers(text: str) -> str:
        """
        Removes JSON block markers from the text.

        Args:
            text (str): Original text.

        Returns:
            str: Cleaned text.
        """
        return text.replace('```json', '').replace('```', '').strip()

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Converts a list of messages into a single prompt.

        Args:
            messages (List[Dict[str, str]]): List of messages.

        Returns:
            str: Combined prompt.
        """
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])