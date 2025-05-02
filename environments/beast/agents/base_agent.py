from base.agent import Agent
from beast.utils.utils import create_message
from beast.utils.prompt import get_role_prompt, get_choose_conv_prompt, get_conv_prompt
import json

class BeastAgent(Agent):
    def __init__(self, chatbot, player_name, players, wealth) -> None:
        super().__init__(chatbot, player_name, players)
        self.wealth = wealth
        self.role_prompt = self.get_role_prompt()

        role_message = create_message("system", self.role_prompt)
        self.private_history.append(role_message)
        
    def get_role_description(self) -> str:
        return f"A player in the Beast game with {self.wealth} wealth"
    
    def get_role_prompt(self):
        return get_role_prompt(self.player_name, self.wealth)
    
    def choose_opponents(self, players_remaining):
        messages = self.private_history.copy()
        messages.append(create_message("user", get_choose_conv_prompt(players_remaining)))
        messages.append(create_message("user","Remember, you must reply as list of string oponnent names as required. Also you couldn't choose your own name"))
        response = self.chatbot.multi_chat(messages)
        try:
            valid_players = [p for p in self.players if p != self.player_name and p in response]
            return valid_players
        except Exception as e:
            print(e)

        return []
    
    def chat(self, context: str) -> tuple[str, dict[str | None]]:
        """Process a chat message and return response with chain of thought"""
        messages = self.private_history.copy()
        messages.append({"role": "user", "content": context})
        response = self.chatbot.multi_chat(messages)
        return response, {"thought": "Processed chat message and generated response"}
    
    def bargain(self, opponent_name):
        # Prepare conversation context

        messages = self.private_history.copy()

        messages.append(create_message("user", get_conv_prompt(opponent_name)))

        messages.append(create_message("user",'Remember, you must reply a json string in form of {"message":{your reply},"offer":{your offer(if you want to make one, should always be integer)}}, FOLLOW THIS RULE STRICTLY!!! ONLY JSON!!!'))

        
        res = self.chatbot.multi_chat(messages)

        
        try:
            # Clean and parse response for message and offer
            # Remove any markdown code block markers and clean the string
            if res.startswith('```json'):
                res = res[7:]
            if res.endswith('```'):
                res = res[:-3]
            # Clean the string by removing extra whitespace and newlines
            res = res.strip()
            response_data = json.loads(res)
            message = response_data.get("message", "")
            offer = response_data.get("offer") or 0
            
            # Record the conversation
            self.private_history.append(
                create_message(
                    "assistant", 
                    f"{message} with offer: {offer}"
                )
            )
            
            return message, offer
        except Exception as e:
            print(e)
            return None, 0
    
    def handle_offer(self, opponent_name, amount):
        messages = self.private_history.copy()

        context = (
            f"{opponent_name} has offered you {amount} wealth.\n"
            f"Your current wealth is {self.wealth}.\n"
            f"Decide whether to accept the offer (true/false).\n"
        )
        
        messages.append(create_message("user", context))
        
        response = self.chatbot.multi_chat(messages)
        try:
            return (True if 'true' in response.lower() else False)
        except Exception as e:
            print(e)
            return False
    
    def vote(self):
        messages = self.private_history.copy()

        context = "It's time to vote. Choose opponent which you want to get 250000 this round. As answer just write name of your opponent and nothing else.\n\n"

        messages.append(create_message("user", context))
        
        response = self.chatbot.multi_chat(messages)
        try:
            # Expect response to be a player name
            if response in self.players and response != self.player_name:
                return response
        except Exception as e:
            print(e)
            return None