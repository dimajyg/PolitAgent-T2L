from base.game import Game
from askguess.agents.answer_agent import AnswerAgent
from askguess.agents.question_agent import QuestionAgent
from askguess.utils.utils import create_message

from askguess.utils.prompt import host_description_prompt, host_qa_prompt

from time import sleep

import re
import json

class AskGuessGame(Game):
    def __init__(self, args, model):
        super().__init__(args)
        self.model = model
        self.word = None
        self.answer_agent = None
        self.question_agent = None
        self.max_rounds = 1000

    def init_game(self, word):
        self.word = word.replace("_", " ")
        self.answer_agent = AnswerAgent(self.model, self.word, self.args)
        self.question_agent = QuestionAgent(self.model, self.word, self.args)
        self.agents = [self.answer_agent, self.question_agent]

    def check_word_mention(self, text):
        """Check whether the answerer directly uses the word as hint"""
        pattern = r'[^\w\s]'
        replaced_text = re.sub(pattern, ' ', text)
        return (" " + self.word + " ") in replaced_text

    def game_loop(self, log_file):
        # Describing Stage
        if self.args.mode == "easy" and self.game_round == 0:
            # Add host describing prompt
            host_message = create_message("user", host_description_prompt)
            
            self.answer_agent.update_history(host_message)
            self.question_agent.update_history(host_message)

            # Get description from answer agent
            sleep(2)
            description = self.answer_agent.play()


            if description is None:
                return {"object": self.word, "round": -1, "error_type": "ChatError"}

            if self.check_word_mention(description.lower()):
                return {"object": self.word, "round": -1, "error_type": "AnswerMentionedError"}

            self.log_message(log_file, f"description: {description}")
            questioner_message = create_message("user", description)
            answerer_message = create_message("assistant", description)
            self.answer_agent.update_history(answerer_message)
            self.question_agent.update_history(questioner_message)

            self.game_round += 1
            return None

        # Q&A Stage
        # Add host Q&A prompt
        host_message = create_message("user", host_qa_prompt)
        self.answer_agent.update_history(host_message)
        self.question_agent.update_history(host_message)

        result = self.handle_qa_stage(log_file)
        if result is False:
            return {"object": self.word, "round": -1, "error_type": "ChatError"}
        elif isinstance(result, dict):
            return result

        self.game_round += 1
        if self.game_round > self.max_rounds:
            return {"object": self.word, "round": -1, "error_type": "RoundLimitError"}

        return None

    def handle_description_stage(self, log_file):
        sleep(2)
        description = self.answer_agent.play()
        if description is None:
            return False

        if self.check_word_mention(description.lower()):
            return {"object": self.word, "round": -1, "error_type": "AnswerMentionedError"}

        self.log_message(log_file, f"description: {description}")
        questioner_message = create_message("user", description)
        answerer_message = create_message("assistant", description)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)
        return True

    def handle_qa_stage(self, log_file):
        # Get question
        sleep(2)
        question = self.question_agent.play()
        if question is None:
            return False

        self.log_message(log_file, f"question: {question}")
        questioner_message = create_message("assistant", question)
        answerer_message = create_message("user", question)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)

        # Get answer
        sleep(2)
        answer = self.answer_agent.play()
        if answer is None:
            return False

        self.log_message(log_file, f"answer: {answer}")
        questioner_message = create_message("user", answer)
        answerer_message = create_message("assistant", answer)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)

        # Check game end conditions
        if "gameover" in answer.lower() or "game over" in answer.lower():
            if self.word.lower() in question.lower():
                return {"object": self.word, "round": self.game_round, "error_type": "SuccessfulTrial"}
            else:
                return {"object": self.word, "round": -1, "error_type": "EndingError"}

        if self.check_word_mention(answer.lower()):
            return {"object": self.word, "round": -1, "error_type": "AnswerMentionedError"}

        return True