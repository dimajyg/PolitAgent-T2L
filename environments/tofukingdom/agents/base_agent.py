from llm.agent import Agent
from environments.tofukingdom.utils.prompt import game_prompt_en
from environments.tofukingdom.utils.utils import create_message, print_messages
import json

class BaseAgent(Agent):
    def __init__(self, chatbot, player_name, all_players) -> None:
        super().__init__(chatbot, player_name, all_players)
        self.game_prompt = game_prompt_en
        self.role_prompt = self.get_role_prompt()

    def get_role_description(self) -> str:
        return f"A {self.role} in the Tofu Kingdom game"

    def get_role_prompt(self):
        role_prompt = '''
        You now need to play the role of the Maid.
        For the Prince's question, you can choose to say the truth or lie.

        '''
        return role_prompt
    
    def chat(self,identities):
        role_prompt = (f"{self.game_prompt} \n"
                f"Now, you are player {self.player_name} "
                f"{self.role_prompt} \n"
                f"This is the identity information of other players: {identities} \n "
                )
        last_prompt = (f'''Your reply must be a JSON string in the following format: \n'''
                '''{"thought":{your thought},"answer":"your answer"} \n'''
                f''' 'thought' represent your thought of how to answer the question according to the rule and your goal. '''
                f''' 'answer' represent your reply to the Prince. ''')
        messages = []
        first_message = create_message("system",role_prompt)
        messages.append(first_message)
        messages += self.private_history
        last_message = create_message("system",last_prompt)
        messages.append(last_message)

        cnt = 0
        while True:
            try:
                res = self.chatbot.multi_chat(messages)
                # Remove markdown code block markers if present
                res = res.strip()
                if res.startswith('```json') and res.endswith('```'):
                    res = res[7:-3].strip()
                if res.startswith('```') and res.endswith('```'):
                    res = res[3:-3].strip()
                res = json.loads(res)
                break
            except:
                cnt += 1
            if cnt >= 3:
                return None, None
            
        answer = res["answer"]
        return answer, res
        
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
