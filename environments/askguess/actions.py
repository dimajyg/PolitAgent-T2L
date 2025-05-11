from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
import json

class AskGuessBaseAction(BaseModel):
    """Base class for all actions in the AskGuess game."""
    action_type: str = Field(..., description="The specific type of action being performed.")
    rationale: Optional[str] = Field(None, description="The agent's reasoning or thought process behind this action.")
    
    def get_serialized(self) -> str:
        """Return JSON string representation, compatible with both Pydantic v1 and v2"""
        try:
            # Try Pydantic v2 method first
            if hasattr(self, "model_dump_json"):
                return self.model_dump_json()
            # Fallback to Pydantic v1
            elif hasattr(self, "json"):
                return self.json()
            # Ultimate fallback
            else:
                return json.dumps({
                    "action_type": self.action_type,
                    "rationale": self.rationale,
                    **{k: v for k, v in self.__dict__.items() 
                       if k not in ["action_type", "rationale"]}
                })
        except Exception as e:
            print(f"Error serializing action: {e}")
            return str(self)

class AskQuestionAction(AskGuessBaseAction):
    """Action for the Questioner to ask a question."""
    action_type: Literal["ask_question"] = "ask_question"
    question_text: str = Field(..., description="The content of the question being asked.")

class ProvideAnswerAction(AskGuessBaseAction):
    """Action for the Answerer to provide an answer or description."""
    action_type: Literal["provide_answer"] = "provide_answer"
    answer_text: str = Field(..., description="The content of the answer or description.")
    # In easy mode, the first action is a description. This can be handled by the same model.

class MakeGuessAction(AskGuessBaseAction):
    """Action for the Questioner to make a final guess."""
    action_type: Literal["make_guess"] = "make_guess"
    guessed_word: str = Field(..., description="The word the Questioner is guessing.")
    is_correct: Optional[bool] = Field(None, description="To be filled by game logic: whether the guess was correct.")

class SignalEndGameAction(AskGuessBaseAction):
    """Action to signal that the game should end based on agent's observation or decision."""
    action_type: Literal["signal_end_game"] = "signal_end_game"
    reason: str = Field(..., description="Reason for ending the game (e.g., 'Successfully guessed the word', 'Answerer mentioned the word', 'Max rounds reached and word not guessed by me').")
    # This can be used by the Questioner if it thinks it guessed, or by Answerer if it breaks a rule, or if Questioner gives up.

# Union type for actions QuestionAgent can take
QuestionerAction = Union[AskQuestionAction, MakeGuessAction, SignalEndGameAction]

# Union type for actions AnswerAgent can take
AnswererAction = Union[ProvideAnswerAction, SignalEndGameAction] 