from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

class DescribeOutput(BaseModel):
    """Model for structured output of description."""
    thought: str = Field(description="Private reasoning about the strategy and role")
    speak: str = Field(description="Public statement about the word without directly saying it")

class VoteOutput(BaseModel):
    """Model for structured output of voting."""
    thought: str = Field(description="Private reasoning about who might be the spy")
    speak: str = Field(description="Public statement explaining vote")
    name: str = Field(description="Name of player being voted for (must be in living_players)")

# Создание парсеров
describe_parser = PydanticOutputParser(pydantic_object=DescribeOutput)
vote_parser = PydanticOutputParser(pydantic_object=VoteOutput)

def read_prompt_file(file_path):
    """Reads a prompt file."""
    with open(file_path, 'r') as f:
        return f.read()
    
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

_game_prompt_content = read_prompt_file(os.path.join(PROMPTS_DIR, "game_prompt.txt"))
_describe_prompt_content = read_prompt_file(os.path.join(PROMPTS_DIR, "describe_prompt.txt"))
_vote_prompt_content = read_prompt_file(os.path.join(PROMPTS_DIR, "vote_prompt.txt"))
_announce_prompt_content = read_prompt_file(os.path.join(PROMPTS_DIR, "announce_prompt.txt"))

game_prompt_template = PromptTemplate.from_template(_game_prompt_content)

describe_prompt_template = PromptTemplate(
    template=_describe_prompt_content,
    input_variables=["game_prompt", "players", "player_name", "role", "phrase"],
    partial_variables={"format_instructions": describe_parser.get_format_instructions()}
)

vote_prompt_template = PromptTemplate(
    template=_vote_prompt_content,
    input_variables=["game_prompt", "player_name", "role", "living_players"],
    partial_variables={"format_instructions": vote_parser.get_format_instructions()}
)

announce_prompt_template = PromptTemplate(
    template=_announce_prompt_content,
    input_variables=["game_prompt", "announcement"]
)

game_prompt_en = _game_prompt_content
