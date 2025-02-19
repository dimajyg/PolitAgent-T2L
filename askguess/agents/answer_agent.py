from base.agent import Agent
from askguess.utils.prompt import get_answerer_role, get_questioner_role
from askguess.utils.utils import create_message, print_messages, convert_messages_to_prompt

class AnswerAgent(Agent):

    def __init__(self, chatbot, object_name, args) -> None:
        # Initialize with empty players list since this game doesn't use multiple players
        super().__init__(chatbot, "Answerer", [])
        self.object_name = object_name
        self.role_easy, self.role_hard = get_answerer_role(object_name)
        
        if args.mode == "easy":
            role_message = create_message("system", self.role_easy)
        else:
            role_message = create_message("system", self.role_hard)
        self.private_history.append(role_message)
 
    def get_role_description(self) -> str:
        return "The answerer in the Ask & Guess game"

    def chat(self, context: str):
        # Basic chat implementation
        messages = self.private_history + [create_message("user", context)]
        response = self.chatbot.multi_chat(messages)
        return response, {"answer": response}

    def play(self):
        response = self.chatbot.multi_chat(self.private_history)
        return response
    
    def answer(self):
        response = self.chatbot.multi_chat(self.private_history)
        return response

    def get_answer_prompt(self):
        prompt = "##system##"
        prompt += self.answer_role + "\n"
        flag = (len(self.private_history) + 1) % 2
        for i in range(len(self.private_history)):
            if i % 2 == flag:
                prompt += "##questioner##: "
            else: 
                prompt += "##answerer##:"
            prompt += self.private_history[i]["content"] + "\n\n"
        prompt +=  "##answerer##:"

        return prompt
    
    def get_describe_prompt(self):
        
        prompt = "##system##"
        prompt += self.describe_role + "\n"
        flag = (len(self.private_history) + 1) % 2
        for i in range(len(self.private_history)):
            if i % 2 == flag:
                prompt += "##questioner##: "
            else: 
                prompt += "##answerer##:"
            prompt += self.private_history[i]["content"] + "\n\n"
        prompt += "##answerer##:"

        return prompt


