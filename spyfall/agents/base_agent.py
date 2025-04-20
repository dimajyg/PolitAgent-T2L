from base.agent import Agent
from spyfall.utils.prompt import game_prompt_en
from spyfall.utils.utils import create_message, print_messages
import json
import random

class BaseAgent(Agent):
    def __init__(self, chatbot, player_name, players, phrase, is_spy=False) -> None:
        self.phrase = phrase
        self.is_spy = is_spy
        super().__init__(chatbot, player_name, players)
        self.game_prompt = game_prompt_en
        self.role_prompt = self.get_role_prompt()
        self.role_messages = self.get_role_messages()
        self.vote_messages = self.get_vote_messages()
        
    def get_role_description(self) -> str:
        role = "spy" if self.is_spy else "villager"
        return f"A {role} in the Spyfall game with the phrase: {self.phrase}"

    def chat(self, context: str):
        # Implement the abstract chat method
        messages = []
        messages.append(create_message("system", self.game_prompt))
        messages.append(create_message("user", context))
        messages += self.private_history
        
        response = self.chatbot.multi_chat(messages)
        try:
            cot = json.loads(response)
            return cot.get("answer", ""), cot
        except:
            return None, None

    def get_role_prompt(self):
        role = "spy" if self.is_spy else "villager"
        role_prompt = (
            f'''{self.game_prompt}'''
            f'''The players involved in the game are: {json.dumps(self.players)}.'''
            f'''You are {self.player_name} \n'''
            f'''Your role is {role} \n'''
            f'''Your given phrase is {self.phrase} \n'''
        )

        return role_prompt

    def get_role_messages(self):
        role = "spy" if self.is_spy else "villager"
        
        messages = []
        messages.append(create_message("system", self.game_prompt))
        temp = f"Now I have read the rules and I know how to play the game. I'm a {role}. Can you offer me some key strategy to win the game with this role?"
        messages.append(create_message("assistant", temp))
        
        if self.is_spy:
            temp = f"As a spy, your main strategy is to blend in without revealing your identity. Since you don't know the common word, you need to use vague descriptions that could apply to many things."
            messages.append(create_message("user", temp))
            temp = f"Listen carefully to other players' descriptions to try and guess what the common word might be. Then try to give descriptions that seem compatible with what you're hearing."
            messages.append(create_message("user", temp))
            temp = f"When others get suspicious, don't be afraid to cast suspicion on someone else to divert attention. But be careful not to be too aggressive as it might backfire."
            messages.append(create_message("user", temp))
        else:
            temp = f"As a villager, your goal is to identify the spy. Give clear but not too obvious descriptions about your word to help other villagers recognize you're on the same team."
            messages.append(create_message("user", temp))
            temp = f"Pay close attention to vague or inconsistent descriptions - these could indicate the spy who doesn't know the common word."
            messages.append(create_message("user", temp))
            temp = f"Be careful not to make your descriptions too obvious, or you'll help the spy figure out the word too quickly."
            messages.append(create_message("user", temp))
        
        temp = f"I understand these strategies. I'll use them to play effectively."
        messages.append(create_message("assistant", temp))
        
        temp = f"Now you are {self.player_name}, you are a {role}, and the word you got is {self.phrase}."
        messages.append(create_message("user", temp))
        
        temp = f"Received. I'll play as {self.player_name} the {role} with the word '{self.phrase}'."
        messages.append(create_message("assistant", temp))
        
        temp = (
            f'''Your reply should be a string in the json format as follows:\n'''
            '''{"thought":{your though},"speak":{your speak}}\n '''
            f''' "thought" represent your thinking, which can be seen only by your self. \n'''
            f''' "speak" represent your speak in this round, which can been seen by all the other players. \n'''
        )
        messages.append(create_message("user", temp))
        temp = (
            '''Your speak should only contain the few words about the word you received, you should not speak like 'i agree with {player_name}' or other thing irrelevant to the word you received. '''
        )
        messages.append(create_message("user", temp))
        temp = f"I understand. I will reply with a json string, and I will not repeat other players' speak or my own speak in the previous round."
        messages.append(create_message("assistant", temp))
        
        return messages
    
    def get_vote_messages(self):
        role = "spy" if self.is_spy else "villager"
        
        messages = []
        messages.append(create_message("system", self.game_prompt))
        temp = f"Now I have read the rules and I'm ready for the voting phase. I'm a {role} in this game. What strategy should I use?"
        messages.append(create_message("assistant", temp))
        
        if self.is_spy:
            temp = f"As a spy in the voting phase, your goal is to avoid being detected. Pay attention to which players seem confident about the word, they're likely villagers."
            messages.append(create_message("user", temp))
            temp = f"You should vote for someone who seems most suspicious of you, or someone who has been particularly clear in their descriptions (thus clearly a villager)."
            messages.append(create_message("user", temp))
            temp = f"In your speech before voting, try to sound confident and perhaps cast suspicion on a villager who seems most vocal against you."
            messages.append(create_message("user", temp))
        else:
            temp = f"As a villager in the voting phase, your goal is to identify the spy. Look for players whose descriptions seem vague or inconsistent with the common understanding."
            messages.append(create_message("user", temp))
            temp = f"Vote for the player you believe is most likely the spy based on their descriptions and behavior."
            messages.append(create_message("user", temp))
            temp = f"In your speech, explain your reasoning to help convince other villagers. Be clear about why you suspect certain players."
            messages.append(create_message("user", temp))
        
        temp = f"I understand these voting strategies and will apply them."
        messages.append(create_message("assistant", temp))
        
        temp = f"Now you are {self.player_name}, you are a {role}, and the word you got is {self.phrase}."
        messages.append(create_message("user", temp))
        
        temp = f"Received. I'll approach the voting as {self.player_name} the {role} with the word '{self.phrase}'."
        messages.append(create_message("assistant", temp))
        
        temp = (
             f'''In the voting stage, your reply should be a string in the json format as follows:\n'''
             '''{"thought":{your though},"speak":{your speak},"name":{voted name}} \n '''
            f''' "thought" represent your thinking, which can be seen only by your self. \n'''
            f''' "speak" represent your speak in the game, which can be seen for all the players. \n'''
            f''' "name" can be only select from the living players. \n '''
        )
        messages.append(create_message("user", temp))
        temp = f"I understand. I will reply with a json string containing my thought process, what I'll say publicly, and who I'm voting for from the living players."
        messages.append(create_message("assistant", temp))
        return messages
        
    def describe(self):
        messages = self.role_messages + self.private_history
        role_reminder = f"You are a {'spy' if self.is_spy else 'villager'}. Remember to describe your word accordingly."
        messages.append(create_message("system", role_reminder))
        messages.append(create_message("system", "Remember, you must reply a json string as required. And you must not repeat the statements of other players and your own past statement."))
        speak = None
        res = None
        res = self.chatbot.multi_chat(messages)
        # Remove any markdown code block markers if present
        if res.startswith('```json'):
            res = res[7:]
        if res.endswith('```'):
            res = res[:-3]
        res = res.strip()
        try:
            res = json.loads(res)
            speak = res["speak"]
        except:
            pass
        return speak, res
    
    def vote(self):
        messages = self.vote_messages + self.private_history
        
        # Извлекаем информацию о живых игроках из истории сообщений
        living_players = self.extract_living_players_from_history()
        
        # Добавляем явное указание голосовать только за живых игроков
        if living_players:
            role_instruction = f"You are a {'spy' if self.is_spy else 'villager'}. "
            if self.is_spy:
                instructions = (
                    f"IMPORTANT: {role_instruction}The current living players are: {json.dumps(living_players)}. "
                    f"You MUST vote only for someone from this list. Try to avoid suspicion and vote strategically."
                )
            else:
                instructions = (
                    f"IMPORTANT: {role_instruction}The current living players are: {json.dumps(living_players)}. "
                    f"You MUST vote only for someone from this list. Try to identify who is the spy based on their descriptions."
                )
            messages.append(create_message("system", instructions))
        
        messages.append(create_message("system", "Remember, you must reply a json string as required, and the 'speak' must not repeat with the statements of other players or your own past statement. The 'name' must be the same string chosen from the given list 'living players'."))
        res = self.chatbot.multi_chat(messages)
        thought = None 
        speak = None 
        name = None
        # Remove any markdown code block markers if present
        if res.startswith('```json'):
            res = res[7:]
        if res.endswith('```'):
            res = res[:-3]
        res = res.strip()
        try:
            res = json.loads(res)
            thought = res["thought"]
            speak = res["speak"]
            name = res["name"]
            
            # Дополнительная проверка: если имя не в списке живых игроков, выбираем случайное
            if living_players and name not in living_players:
                name = random.choice(living_players)
                thought += f" [NOTE: Original vote was invalid, randomly selected {name} from living players]"
        except:
            # В случае ошибки парсинга, если есть живые игроки, выбираем случайного
            if living_players:
                name = random.choice(living_players)
                thought = f"Failed to parse my response, randomly selecting {name}"
                speak = f"I think {name} is suspicious"
        
        return name, speak, res

    def extract_living_players_from_history(self):
        """Извлекает список живых игроков из истории сообщений"""
        living_players = []
        
        # Ищем в истории сообщений указания о живых игроках
        for message in reversed(self.private_history):
            if message["role"] == "user" and "Host: The living players are:" in message["content"]:
                try:
                    # Извлекаем JSON часть сообщения
                    content = message["content"]
                    json_start = content.find('[')
                    json_end = content.rfind(']') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        living_players = json.loads(json_str)
                        break
                except:
                    # Если не удалось извлечь, используем исходный список игроков
                    living_players = self.players.copy()
                    break
        
        # Если не нашли в истории, используем полный список игроков
        if not living_players:
            living_players = self.players.copy()
        
        return living_players
        
    def convert_messages_to_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt +=  "system: "
                prompt += message["content"]
                prompt += "\n"
            elif message["role"] == "assistant":
                prompt += f"{self.player_name}: "
                prompt += message["content"]
                prompt += "\n"
            else:
                prompt += message["content"]
                prompt += "\n"
        prompt += f"{self.player_name}: "
        return prompt