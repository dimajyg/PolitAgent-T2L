"""
Module for loading and providing AskGuess game prompts through LangChain PromptTemplate.
"""
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from typing import Any

PROMPT_DIR = Path(__file__).parent.parent / "prompts"

def load_prompt(filename: str) -> str:
    """Loads prompt from text file.

    Args:
        filename (str): Name of the prompt file.

    Returns:
        str: Content of the prompt.
    """
    with open(PROMPT_DIR / filename, encoding="utf-8") as f:
        return f.read()

# Примеры функций для получения PromptTemplate

def get_answerer_prompt_template(mode: str) -> PromptTemplate:
    """Returns PromptTemplate for the answerer role.

    Args:
        mode (str): Game mode ('easy' or 'hard').

    Returns:
        PromptTemplate: Prompt template for the answerer.
    """
    filename = f"answerer_{mode}.txt"
    template = load_prompt(filename)
    return PromptTemplate.from_template(template)

def get_questioner_prompt_template(mode: str) -> PromptTemplate:
    """Returns PromptTemplate for the questioner role.

    Args:
        mode (str): Game mode ('easy' or 'hard').

    Returns:
        PromptTemplate: Prompt template for the questioner.
    """
    filename = f"questioner_{mode}.txt"
    template = load_prompt(filename)
    return PromptTemplate.from_template(template)

def get_host_description_prompt() -> str:
    """Returns the prompt for the host's word description.

    Returns:
        str: Word description prompt.
    """
    return load_prompt("host_description.txt")

def get_host_qa_prompt() -> str:
    """Returns the prompt for starting the Q&A round.

    Returns:
        str: Q&A round prompt.
    """
    return load_prompt("host_qa.txt")