from base.agent import Agent
from askguess.utils.prompt import get_questioner_role
from askguess.utils.utils import create_message, convert_messages_to_prompt, print_messages

class QuestionAgent(Agent):

    def __init__(self, chatbot, object_name, args) -> None:
        # Initialize with empty players list since this game doesn't use multiple players
        super().__init__(chatbot, "Questioner", [])
        self.object_name = object_name
        self.role_easy, self.role_hard = get_questioner_role()
        
        if args.mode == "easy":
            role_message = create_message("system", self.role_easy)
        else:
            role_message = create_message("system", self.role_hard)
        self.private_history.append(role_message)
    
    def get_role_description(self) -> str:
        return "The questioner in the Ask & Guess game"

    def chat(self, context: str):
        # Basic chat implementation
        messages = self.private_history + [create_message("user", context)]
        response = self.chatbot.multi_chat(messages)
        return response, {"answer": response}

    def play(self):
        response = self.chatbot.multi_chat(self.private_history)
        return response



