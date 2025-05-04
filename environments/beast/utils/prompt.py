"""
Module for loading and providing prompts for the Beast game using LangChain PromptTemplate.
"""
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Optional, Union
import logging

PROMPT_DIR = Path(__file__).parent.parent / "prompts"

def load_prompt(filename: str) -> str:
    """Loads a prompt from a text file.

    Args:
        filename (str): Name of the file containing the prompt.

    Returns:
        str: The prompt content.
    """
    prompt_path = PROMPT_DIR / filename
    
    # Check if file exists
    if not prompt_path.exists():
        logging.error(f"Prompt file '{filename}' not found in {PROMPT_DIR}")
        raise FileNotFoundError(f"Prompt file '{filename}' not found in {PROMPT_DIR}")
    
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()

def get_role_prompt_template() -> PromptTemplate:
    """Returns PromptTemplate for the player role.

    Returns:
        PromptTemplate: The role prompt template.
    """
    template = load_prompt("role_prompt.txt")
    return PromptTemplate.from_template(template)

def get_choose_conv_prompt_template() -> PromptTemplate:
    """Returns PromptTemplate for choosing conversation partners.

    Returns:
        PromptTemplate: The prompt template for choosing conversation partners.
    """
    template = load_prompt("choose_conversation_prompt.txt")
    return PromptTemplate.from_template(template)

def get_conv_prompt_template() -> PromptTemplate:
    """Returns PromptTemplate for a conversation with an opponent.

    Returns:
        PromptTemplate: The conversation prompt template.
    """
    template = load_prompt("conversation_prompt.txt")
    return PromptTemplate.from_template(template)

def get_wealth_status_template() -> PromptTemplate:
    """Returns PromptTemplate for displaying current wealth status.

    Returns:
        PromptTemplate: The wealth status prompt template.
    """
    template = load_prompt("wealth_status_prompt.txt")
    return PromptTemplate.from_template(template)

def get_voting_results_template() -> PromptTemplate:
    """Returns PromptTemplate for displaying voting results.

    Returns:
        PromptTemplate: The voting results prompt template.
    """
    template = load_prompt("voting_results_prompt.txt")
    return PromptTemplate.from_template(template)

def format_prompt(prompt_template: Union[PromptTemplate, str], **kwargs: Any) -> str:
    """Format a prompt template with the given variables.
    
    Args:
        prompt_template: Either a PromptTemplate or a template filename
        **kwargs: Variables to format the template with
        
    Returns:
        str: The formatted prompt
    """
    if isinstance(prompt_template, str):
        template = load_prompt(prompt_template)
        prompt_template = PromptTemplate.from_template(template)
    
    return prompt_template.format(**kwargs)

def get_current_wealth_prompt(wealth: Dict[str, int]) -> str:
    """Returns a prompt showing the current wealth of all players.

    Args:
        wealth (Dict[str, int]): A dictionary mapping player names to their wealth.

    Returns:
        str: The formatted wealth prompt.
    """
    # Format the wealth status as a string
    wealth_status = ''.join(f"{player} has {amount}\n" for player, amount in wealth.items())
    
    # Use the template
    return format_prompt(get_wealth_status_template(), wealth_status=wealth_status)

def get_voting_prompt(voting_results: Dict[str, int]) -> str:
    """Returns a prompt showing the voting results.

    Args:
        voting_results (Dict[str, int]): A dictionary mapping player names to their vote counts.

    Returns:
        str: The formatted voting results prompt.
    """
    # Format the voting results as a string
    voting_results_str = ''.join(f"{player} has {votes}\n" for player, votes in voting_results.items())
    
    # Use the template
    return format_prompt(get_voting_results_template(), voting_results=voting_results_str)
    